from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import joblib
import io
import os

app = Flask(__name__)
CORS(app)

# Load the saved model and emotions mapping
model_path = "./saved_models/emotion_model.joblib"
emotions_path = "./saved_models/emotions.joblib"

if not os.path.exists(model_path) or not os.path.exists(emotions_path):
    raise FileNotFoundError("Model or emotions file not found. Please train and save the model first.")

model = joblib.load(model_path)
emotions = joblib.load(emotions_path)

AVAILABLE_EMOTIONS = emotions

# Feature extraction function (same as in the model training code)
def extract_features(audio_data):
    try:
        y, sr = librosa.load(io.BytesIO(audio_data), duration=3, sr=22050)
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_scaled = np.mean(chroma.T, axis=0)
        
        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_scaled = np.mean(mel.T, axis=0)
        
        # RMS Energy
        rmse = librosa.feature.rms(y=y)
        rmse_scaled = np.mean(rmse.T, axis=0)
        
        features = np.concatenate([
            mfcc_scaled,
            chroma_scaled,
            mel_scaled,
            rmse_scaled,
        ])
        
        return features
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    audio_data = audio_file.read()

    # Extract features from the audio data
    features = extract_features(audio_data)
    if features is None:
        return jsonify({'error': 'Failed to extract features from audio data'}), 500

    try:
        # Reshape features and make predictions
        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        emotion_scores = {
            emotion: float(prob)
            for emotion, prob in zip(AVAILABLE_EMOTIONS, probabilities)
        }

        return jsonify({
            'predicted_emotion': prediction,
            'confidence_scores': emotion_scores
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
