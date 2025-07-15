from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
from model_predictor_face import predict_face_emotion

# Create Flask app
app = Flask(__name__)
CORS(app)  # Allow requests from your HTML page

@app.route('/predict-face', methods=['POST'])
def predict_face():
    try:
        print("🚀 /predict-face endpoint was hit")

        data = request.get_json()
        base64_img = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        img_bytes = base64.b64decode(base64_img)

        nparr = np.frombuffer(img_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_np is None:
            print("❌ Error decoding image.")
            return jsonify({"emotion": "Error decoding image"})

        print("📷 Image shape:", img_np.shape)

        emotion = predict_face_emotion(img_np)
        print("🧠 Emotion Detected:", emotion)

        return jsonify({"emotion": emotion})

    except Exception as e:
        print("🔥 Error detecting emotion:", e)
        return jsonify({"emotion": "Error"})


# ✅ This line starts the Flask server!
if __name__ == "__main__":
    app.run(debug=True)
