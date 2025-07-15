import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_face(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(gray, (48, 48))
        face_array = face_resized.astype("float32") / 255.0
        face_array = np.expand_dims(face_array, axis=0)
        face_array = np.expand_dims(face_array, axis=-1)
        return face_array
    except Exception as e:
        print("Preprocessing error:", e)
        return None

def predict_face_emotion(img):
    face = preprocess_face(img)
    if face is None:
        return "No face"
    try:
        prediction = model.predict(face)[0]
        return emotion_labels[np.argmax(prediction)]
    except Exception as e:
        print("Prediction error:", e)
        return "Error"
