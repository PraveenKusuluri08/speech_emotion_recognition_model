import numpy as np
import librosa
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import glob
import os

emotions = {
    "01": "angry",
    "02": "fearful",
    "03": "happy",
    "04": "sad",
    "05": "surprise",
    "06": "neutral"
}
AVAILABLE_EMOTIONS = ["angry", "fearful", "surprise", "neutral"]

def load_audio_data_files() -> list:
    return glob.glob("./Audio_Speech_Actors/Actor_0*/*")

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    features = np.concatenate([
        np.mean(mfcc.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(mel.T, axis=0),
        np.mean(rmse.T, axis=0),
    ])
    return features

def get_emotion_from_filename(filename):
    emotion_code = filename.split('-')[2]
    return emotions.get(emotion_code)

X, y = [], []
audio_files = load_audio_data_files()
for audio_path in audio_files:
    emotion = get_emotion_from_filename(os.path.basename(audio_path))
    if emotion in AVAILABLE_EMOTIONS:
        features = extract_features(audio_path)
        X.append(features)
        y.append(emotion)

X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam', max_iter=500)
model.fit(X_train, y_train)

os.makedirs("saved_models", exist_ok=True)
joblib.dump(model, "./saved_models/emotion_model.joblib")
joblib.dump(AVAILABLE_EMOTIONS, "./saved_models/emotions.joblib")

print("Model training complete and saved.")
