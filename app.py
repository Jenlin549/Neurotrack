from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import pandas as pd
from datetime import datetime
import os

# ✅ Initialize Flask app and CORS once
app = Flask(__name__)
CORS(app)

# ✅ Load models
face_model = load_model("emotion_model.h5")
voice_model = load_model("voice_emotion.h5")

emotion_labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

# 🎙️ Voice Emotion Prediction Route
@app.route("/predict-voice", methods=["POST"])
def predict_voice():
    try:
        audio_file = request.files['audio']
        audio_path = "temp_voice.wav"
        audio_file.save(audio_path)

        y, sr = librosa.load(audio_path, sr=22050, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0).reshape(1, -1)

        predictions = voice_model.predict(mfcc_scaled)
        predicted_index = np.argmax(predictions)
        emotion = emotion_labels[predicted_index]

        return jsonify({'emotion': emotion})
    except Exception as e:
        print("❌ Voice Emotion Error:", e)
        return jsonify({'emotion': "undefined"}), 500

# 🔐 Login Logging Route
@app.route("/log-login", methods=["POST"])
def log_login():
    try:
        data = request.json
        username = data.get("username")
        email = data.get("email", "")
        login_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        file_path = "login_log.xlsx"

        # ✅ Create or load Excel file
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
        else:
            df = pd.DataFrame(columns=["Username", "Email", "Login Time"])

        # ✅ Add new login row
        new_entry = {"Username": username, "Email": email, "Login Time": login_time}
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

        df.to_excel(file_path, index=False)
        return {"status": "success"}

    except Exception as e:
        print("❌ Login Log Error:", e)
        return {"status": "error"}, 500

# ✅ Start the Flask server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

