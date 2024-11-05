import numpy as np
import librosa
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import glob
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import os
import joblib

emotions = {
    "01": "angry",
    "02": "fearful",
    "03": "happy",
    "04": "sad",
    "05": "surprise",
    "06": "neutral"
}

AVAILABLE_EMOTIONS = ["angry", "fearful", "surprise", "neutral"]

def load_audio_data_files() -> list[str]:
    return glob.glob("./Audio_Speech_Actors/Actor_0*/*")

def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, duration=3, offset=0.5)
        
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
        print(f"Error extracting features from {audio_path}: {str(e)}")
        return None

def get_emotion_from_filename(filename):
    emotion_code = filename.split('-')[2]
    return emotions.get(emotion_code)

class SpeechRecognitionModel():
    def __init__(self):
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=500
        )
        self.emotions = AVAILABLE_EMOTIONS
    
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.classifier.fit(X_train, y_train)
        return self.classifier.score(X_test, y_test)
    
    def predict_emotion(self, audio_path):
        features = extract_features(audio_path)
        features = features.reshape(1, -1)
        prediction = self.classifier.predict(features)
        probabilities = self.classifier.predict_proba(features)
        
        emotion_scores = {
            emotion: float(prob) 
            for emotion, prob in zip(self.emotions, probabilities[0])
        }
        
        return {
            'predicted_emotion': prediction[0],
            'confidence_scores': emotion_scores
        }
    def save_model(self, model_path="saved_models"):
        os.makedirs(model_path, exist_ok=True)
        
        model_file = os.path.join(model_path, "emotion_model.joblib")
        emotions_file = os.path.join(model_path, "emotions.joblib")
        
        joblib.dump(self.classifier, model_file)
        joblib.dump(self.emotions, emotions_file)
        print(f"Model saved to {model_file}")
        

def prepare_dataset():
    X = []
    y = []
    
    audio_files = load_audio_data_files()
    print(f"Found {len(audio_files)} audio files")
    
    for audio_path in audio_files:
        # Get emotion from filename
        filename = os.path.basename(audio_path)
        emotion = get_emotion_from_filename(filename)
        
        # skip if the emotion is not in our target emotions
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        
        # Extract features
        features = extract_features(audio_path)
        if features is not None:
            X.append(features)
            y.append(emotion)
    
    return np.array(X), np.array(y)


def main():
    print("Preparing dataset...")
    X, y = prepare_dataset()
    print(f"Dataset prepared with {len(X)} samples")
    
    if len(X) == 0:
        print("Error: No features were extracted from the audio files")
        return
    
    print("Training model...")
    ser = SpeechRecognitionModel()
    
    ser.save_model()
    
    accuracy = ser.train(X, y)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Test prediction on first file
    test_file = "./Audio_Speech_Actors/Actor_01/03-01-01-01-01-02-01.wav"
    
    if os.path.exists(test_file):
        result = ser.predict_emotion(test_file)
        print(f"\nPredicted emotion: {result['predicted_emotion']}")
        print("Confidence scores:")
        for emotion, score in result['confidence_scores'].items():
            print(f"{emotion}: {score:.2f}")
    else:
        print(f"Test file not found: {test_file}")

main()