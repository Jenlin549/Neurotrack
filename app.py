import os
os.environ["PATH"] += os.pathsep + r"C:\Users\jenli\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"

from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ ADD THIS
from pydub import AudioSegment
from tensorflow.keras.models import load_model
import numpy as np
import librosa

app = Flask(__name__)
CORS(app)  # ✅ ADD THIS TOO — this solves your fetch blocked error

model = load_model("voice_emotion.h5")
emotion_labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

@app.route('/predict-voice', methods=['POST'])
def predict_voice():
    try:
        audio_file = request.files['audio']
        audio_path = "temp_voice.wav"
        audio_file.save(audio_path)

        # Process audio
        y, sr = librosa.load(audio_path, sr=22050, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0).reshape(1, -1)

        # Predict
        predictions = model.predict(mfcc_scaled)
        predicted_index = np.argmax(predictions)
        emotion = emotion_labels[predicted_index]

        print("🧠 Predicted:", emotion)
        return jsonify({'emotion': emotion})

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({'emotion': "undefined"}), 500

if __name__ == '__main__':
    app.run(debug=True)
