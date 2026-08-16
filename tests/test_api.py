import os
import sys
import unittest
import json
import numpy as np

# Ensure parent directory is in python path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from backend.app import app
from ml.audio_processor import FEATURE_COLUMNS

class TestMusicGenreAPI(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_01_models_endpoint(self):
        response = self.app.get('/api/models')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('randomforest', data['available_models'])
        self.assertIn('xgboost', data['available_models'])
        self.assertIn('svm', data['available_models'])
        self.assertIn('neuralnetwork', data['available_models'])

    def test_02_samples_endpoint(self):
        response = self.app.get('/api/samples')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(len(data['samples']), 10)

    def test_03_predict_endpoint_json(self):
        # Create a sample feature dictionary for Rock
        sample_features = {
            "chroma_stft_mean": 0.38,
            "rms_mean": 0.15,
            "spectral_centroid_mean": 2400.0,
            "spectral_bandwidth_mean": 2350.0,
            "rolloff_mean": 5000.0,
            "zero_crossing_rate_mean": 0.102,
            "tempo": 128.0,
            "energy": 0.72,
            "danceability": 0.75
        }
        for i in range(1, 21):
            sample_features[f"mfcc{i}_mean"] = float(-50 + i * 2)

        payload = {
            "model": "ensemble",
            "features": sample_features,
            "filename": "test_rock_track.wav"
        }

        response = self.app.post(
            '/api/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('predicted_genre', data)
        self.assertIn('confidence', data)
        self.assertIn('predictions', data)
        self.assertEqual(len(data['predictions']), 10)

    def test_04_eda_endpoint(self):
        response = self.app.get('/api/eda')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['total_samples'], 1000)
        self.assertIn('correlation_matrix', data)
        self.assertIn('pca_scatter', data)

    def test_05_recommend_endpoint(self):
        sample_features = {col: 0.5 for col in FEATURE_COLUMNS}
        response = self.app.post(
            '/api/recommend',
            data=json.dumps({"features": sample_features}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(len(data['recommendations']), 5)

if __name__ == '__main__':
    unittest.main()
