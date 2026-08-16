import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb

ML_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ML_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DATA_PATH = os.path.join(os.path.dirname(ML_DIR), "data", "gtzan_features.csv")

FEATURE_COLUMNS = [
    "chroma_stft_mean", "rms_mean", "spectral_centroid_mean",
    "spectral_bandwidth_mean", "rolloff_mean", "zero_crossing_rate_mean",
    "tempo", "energy", "danceability"
] + [f"mfcc{i}_mean" for i in range(1, 21)]

def train_and_evaluate_models():
    print(f"Loading GTZAN dataset from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please run data/generate_samples.py first.")

    df = pd.read_csv(DATA_PATH)
    
    X = df[FEATURE_COLUMNS].values
    y_raw = df["label"].values

    # Label Encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    classes = list(label_encoder.classes_)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model definitions
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
        "SVM": SVC(kernel='rbf', C=2.0, probability=True, random_state=42),
        "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42)
    }

    results = {}

    print("Training classifiers...")
    for model_name, model in models.items():
        if model_name in ["SVM", "NeuralNetwork"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average='weighted'))
        cm = confusion_matrix(y_test, y_pred).tolist()

        print(f"  [{model_name}] Accuracy: {acc*100:.2f}% | Weighted F1: {f1:.4f}")

        # Save model checkpoint
        joblib.dump(model, os.path.join(MODELS_DIR, f"{model_name.lower()}.joblib"))

        results[model_name] = {
            "accuracy": acc,
            "f1_score": f1,
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_pred, target_names=classes, output_dict=True)
        }

    # Save scaler and label encoder
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.joblib"))

    # Save benchmark metadata json
    benchmark_meta = {
        "genres": classes,
        "feature_columns": FEATURE_COLUMNS,
        "results": results
    }
    with open(os.path.join(MODELS_DIR, "benchmarks.json"), "w") as f:
        json.dump(benchmark_meta, f, indent=2)

    print("Model training complete. All models and artifacts saved in ml/models/")

if __name__ == "__main__":
    train_and_evaluate_models()
