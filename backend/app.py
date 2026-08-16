import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to sys.path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from ml.audio_processor import extract_features_from_wav, extract_features_from_pcm, extract_vector, FEATURE_COLUMNS

app = Flask(__name__, static_folder=os.path.join(PROJECT_DIR, "frontend"), static_url_path="")

MODELS_DIR = os.path.join(PROJECT_DIR, "ml", "models")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
AUDIO_SAMPLES_DIR = os.path.join(DATA_DIR, "audio_samples")
DATASET_PATH = os.path.join(DATA_DIR, "gtzan_features.csv")

# Global variables for loaded models
loaded_models = {}
scaler = None
label_encoder = None
benchmarks = {}
df_gtzan = None
pca_model = None

def load_ml_assets():
    global loaded_models, scaler, label_encoder, benchmarks, df_gtzan, pca_model
    
    if not os.path.exists(MODELS_DIR) or not os.path.exists(os.path.join(MODELS_DIR, "benchmarks.json")):
        print("ML models not yet trained. Run ml/train_model.py first.")
        return

    try:
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
        label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
        
        for name in ["randomforest", "xgboost", "svm", "neuralnetwork"]:
            model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
            if os.path.exists(model_path):
                loaded_models[name] = joblib.load(model_path)

        with open(os.path.join(MODELS_DIR, "benchmarks.json"), "r") as f:
            benchmarks = json.load(f)
            
        if os.path.exists(DATASET_PATH):
            df_gtzan = pd.read_csv(DATASET_PATH)
            X_all = df_gtzan[FEATURE_COLUMNS].values
            X_scaled = scaler.transform(X_all)
            pca_model = PCA(n_components=3, random_state=42)
            coords = pca_model.fit_transform(X_scaled)
            df_gtzan["pca_x"] = coords[:, 0]
            df_gtzan["pca_y"] = coords[:, 1]
            df_gtzan["pca_z"] = coords[:, 2]

        print(f"Loaded ML assets: {list(loaded_models.keys())}")
    except Exception as e:
        print(f"Error loading ML assets: {e}")

load_ml_assets()

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/audio_samples/<path:filename>")
def serve_audio_sample(filename):
    return send_from_directory(AUDIO_SAMPLES_DIR, filename)

@app.route("/api/models", methods=["GET"])
def get_models_info():
    if not benchmarks:
        load_ml_assets()
    return jsonify({
        "status": "success",
        "available_models": list(loaded_models.keys()),
        "benchmarks": benchmarks.get("results", {}),
        "genres": benchmarks.get("genres", []),
        "feature_columns": FEATURE_COLUMNS
    })

@app.route("/api/samples", methods=["GET"])
def get_samples():
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    samples = []
    for g in genres:
        samples.append({
            "genre": g.capitalize(),
            "filename": f"{g}_sample.wav",
            "url": f"/audio_samples/{g}_sample.wav"
        })
    return jsonify({"status": "success", "samples": samples})

@app.route("/api/predict", methods=["POST"])
def predict():
    if not loaded_models:
        load_ml_assets()

    chosen_model = request.form.get("model", "ensemble").lower()
    features = None
    file_name = "Uploaded Audio Track"

    try:
        if "file" in request.files:
            audio_file = request.files["file"]
            file_name = audio_file.filename
            file_ext = os.path.splitext(file_name)[1].lower() or ".wav"
            temp_path = os.path.join(DATA_DIR, f"temp_upload{file_ext}")
            audio_file.save(temp_path)
            try:
                features = extract_features_from_wav(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        elif request.is_json:
            data = request.get_json()
            if "pcm" in data:
                pcm_data = np.array(data["pcm"], dtype=np.float32)
                sr = data.get("sample_rate", 22050)
                features = extract_features_from_pcm(pcm_data, sr)
            else:
                features = data.get("features")
                file_name = data.get("filename", "Custom Feature Vector")
            if "model" in data:
                chosen_model = data["model"].lower()

        if not features:
            return jsonify({"status": "error", "message": "No audio file or feature vector provided"}), 400

        feature_vec = np.array([extract_vector(features)])
        feature_vec_scaled = scaler.transform(feature_vec)

        genres = list(label_encoder.classes_)

        if chosen_model in loaded_models:
            mdl = loaded_models[chosen_model]
            vec = feature_vec_scaled if chosen_model in ["svm", "neuralnetwork"] else feature_vec
            probs = mdl.predict_proba(vec)[0]
        else:
            prob_list = []
            for name, mdl in loaded_models.items():
                vec = feature_vec_scaled if name in ["svm", "neuralnetwork"] else feature_vec
                prob_list.append(mdl.predict_proba(vec)[0])
            probs = np.mean(prob_list, axis=0)

        genre_predictions = []
        for idx, prob in enumerate(probs):
            genre_predictions.append({
                "genre": genres[idx].capitalize(),
                "confidence": float(round(prob * 100, 2))
            })
        genre_predictions.sort(key=lambda x: x["confidence"], reverse=True)

        top_genre = genre_predictions[0]["genre"]
        top_confidence = genre_predictions[0]["confidence"]

        acoustic_profile = {
            "Energy": float(features.get("energy", 0.5)),
            "Tempo / 180": float(min(1.0, features.get("tempo", 120) / 180.0)),
            "Brightness": float(min(1.0, features.get("spectral_centroid_mean", 2000) / 4000.0)),
            "Danceability": float(features.get("danceability", 0.5)),
            "Harmonics": float(features.get("chroma_stft_mean", 0.35) * 2.0),
            "Dynamic Range": float(min(1.0, features.get("rms_mean", 0.1) * 5.0))
        }

        return jsonify({
            "status": "success",
            "filename": file_name,
            "predicted_genre": top_genre,
            "confidence": top_confidence,
            "model_used": chosen_model.capitalize(),
            "predictions": genre_predictions,
            "features": features,
            "acoustic_profile": acoustic_profile
        })
    except Exception as e:
        print(f"Prediction handler error: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error analyzing audio: {str(e)}"
        }), 400

@app.route("/api/eda", methods=["GET"])
def get_eda():
    if df_gtzan is None:
        load_ml_assets()

    if df_gtzan is None:
        return jsonify({"status": "error", "message": "Dataset not available"}), 500

    genre_means = df_gtzan.groupby("label")[["spectral_centroid_mean", "tempo", "energy", "danceability", "zero_crossing_rate_mean"]].mean().to_dict(orient="index")
    corr_cols = ["chroma_stft_mean", "rms_mean", "spectral_centroid_mean", "spectral_bandwidth_mean", "rolloff_mean", "zero_crossing_rate_mean", "tempo", "energy", "danceability"]
    corr_matrix = df_gtzan[corr_cols].corr().round(3).to_dict()
    pca_samples = df_gtzan.sample(n=min(300, len(df_gtzan)), random_state=42)[["filename", "label", "pca_x", "pca_y", "pca_z", "tempo", "energy"]].to_dict(orient="records")

    return jsonify({
        "status": "success",
        "total_samples": len(df_gtzan),
        "genres": list(df_gtzan["label"].unique()),
        "genre_means": genre_means,
        "correlation_matrix": corr_matrix,
        "pca_scatter": pca_samples
    })

@app.route("/api/recommend", methods=["POST"])
def recommend():
    if df_gtzan is None:
        load_ml_assets()

    data = request.get_json() or {}
    features = data.get("features")
    
    if not features:
        idx = np.random.randint(0, len(df_gtzan))
        features = df_gtzan.iloc[idx][FEATURE_COLUMNS].to_dict()

    target_vec = np.array([extract_vector(features)])
    dataset_vecs = df_gtzan[FEATURE_COLUMNS].values

    sims = cosine_similarity(target_vec, dataset_vecs)[0]
    top_indices = np.argsort(sims)[::-1][1:6]

    recommendations = []
    for idx in top_indices:
        row = df_gtzan.iloc[idx]
        recommendations.append({
            "filename": row["filename"],
            "genre": row["label"].capitalize(),
            "similarity": float(round(sims[idx] * 100, 1)),
            "tempo": float(round(row["tempo"], 1)),
            "energy": float(round(row["energy"], 2)),
            "audio_url": f"/audio_samples/{row['label']}_sample.wav"
        })

    return jsonify({
        "status": "success",
        "recommendations": recommendations
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
