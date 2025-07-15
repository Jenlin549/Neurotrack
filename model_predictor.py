import os
os.environ["PATH"] += os.pathsep + r"C:\Users\jenli\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"

from pydub import AudioSegment
AudioSegment.converter = r"C:\Users\jenli\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\Users\jenli\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffprobe.exe"

import numpy as np
import librosa
from tensorflow.keras.models import load_model

model = load_model("voice_emotion.h5")
emotion_labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

def predict_voice_emotion(audio_path):
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=22050, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0).reshape(1, -1)

        predictions = model.predict(mfcc_scaled)
        predicted_index = np.argmax(predictions)

        return emotion_labels[predicted_index]

    except Exception as e:
        print("❌ Internal error:", e)
        return "undefined"
