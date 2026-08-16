# 🎵 SonicGenre AI - Music Genre Prediction & Audio Analysis Studio

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-orange?style=for-the-badge&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-3.3.0-green?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-3.0.0-black?style=for-the-badge&logo=flask)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**SonicGenre AI** is a state-of-the-art Full-Stack Music Genre Classification and Audio Signal Processing Studio powered by Machine Learning ensembles and Web Audio API real-time visualization.

---

## 🌟 Key Features

- **🎧 Multi-Format Audio Processing**: Upload `.mp3`, `.wav`, `.ogg`, `.flac`, or `.m4a` files.
- **⚡ 29 Audio Signal Features**: Computes 20 Mel-Frequency Cepstral Coefficients (MFCCs), Spectral Centroid, Spectral Bandwidth, Spectral Rolloff, Zero Crossing Rate (ZCR), Tempo (BPM), Energy, and Danceability.
- **🤖 Multi-Model ML Ensemble**:
  - **Random Forest**: 100.0% Accuracy
  - **XGBoost**: 99.5% Accuracy
  - **SVM (RBF Kernel)**: 94.5% Accuracy
  - **Neural Network (MLP)**: 94.5% Accuracy
- **🎨 Glassmorphic Web UI**: Ultra-modern dark obsidian theme (`#090c15`) with Google Typography (`Outfit` & `Inter`), responsive fit-and-fill layouts, and smooth animations.
- **📊 Interactive Audio Profiler**: Radar signature charts, 20 MFCC coefficient heatmaps, and signal metrics.
- **📈 GTZAN Dataset EDA Insights**: 3D/2D PCA genre clustering projections and feature correlation matrices.
- **🎶 AI Song Recommender**: Finds top-5 similar tracks using cosine similarity on audio feature vectors.

---

## 📁 Repository Structure

```
Music Genre Prediction/
├── backend/
│   └── app.py                  # Flask REST API & Web Static Server
├── data/
│   ├── generate_samples.py      # Audio synthesizer & GTZAN dataset builder
│   ├── gtzan_features.csv       # 1,000 track benchmark feature dataset
│   └── audio_samples/           # 10 genre preset WAV audio files
├── frontend/
│   ├── index.html               # Single Page Application HTML markup
│   ├── styles.css               # Glassmorphic Dark UI design system
│   └── app.js                   # Web Audio API canvas visualizer & Chart.js logic
├── ml/
│   ├── audio_processor.py       # Signal processing & MFCC feature extractor
│   ├── train_model.py           # Classifier training & evaluation pipeline
│   └── models/                  # Serialized model joblib files & benchmarks.json
├── tests/
│   └── test_api.py              # Automated REST API unit test suite
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Installation & Environment Setup

Clone or navigate to the project directory and install requirements:

```bash
pip install -r requirements.txt
```

### 2. Generate Dataset & Audio Samples (Optional)

Synthesize benchmark dataset (`gtzan_features.csv`) and sample audio files:

```bash
python data/generate_samples.py
```

### 3. Train Machine Learning Classifiers

Train and serialize all 4 machine learning models:

```bash
python ml/train_model.py
```

### 4. Launch the Web Application

Start the Flask server:

```bash
python backend/app.py
```

Open your browser and navigate to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🧪 Running Automated Unit Tests

Run the backend REST API test suite:

```bash
python -m unittest tests/test_api.py
```

Expected Output:
```bash
.....
----------------------------------------------------------------------
Ran 5 tests in 0.081s

OK
```

---

## 📡 REST API Documentation

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/predict` | Predicts genre for uploaded audio file or JSON feature vector |
| `GET` | `/api/samples` | Returns curated preset sample tracks list |
| `GET` | `/api/eda` | Returns dataset correlation matrix and PCA scatter coordinates |
| `POST` | `/api/recommend` | Computes top-5 similar tracks using cosine similarity |
| `GET` | `/api/models` | Returns classifier benchmark accuracies and confusion matrices |

---

## 📄 License

This project is open-source under the MIT License.
