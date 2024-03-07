from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)


# path setups.....
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
model_path = os.path.join(app.root_path, 'emoji_model_v4.h5')
emotion_model = load_model(model_path)


# preprocessing the image for the model.....
def load_and_preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (48, 48))
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY) 
    img_array = np.expand_dims(gray_frame, axis=0) 
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array


# an optional code to detect and cut faces from the js.....
def detect_face(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        return (x, y, w, h), face_roi
    else:
        return None, None


# Model in the working......
def predict_emotion(model, frame):
    img_array = load_and_preprocess_frame(frame)
    predictions = model.predict(img_array)
    emotion_index = np.argmax(predictions[0])
    emotions = ['Angry','Disgust', 'Fear','Happy', 'neutral', 'Sad', 'Surprise']
    predicted_emotion = emotions[emotion_index]
    return predicted_emotion


# detection and output.......
def detect_emotion(file_path):
    global emotion_model
    # Load the image using OpenCV
    image = cv2.imread(file_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_rect, face_roi = detect_face(image, face_cascade)
    if face_roi is not None:
        predicted_emotion = predict_emotion(emotion_model, face_roi)
        return predicted_emotion
    else:
        return 'No face detected'


#main html....
@app.route('/')
def index():
    return render_template('index.html')

# Api......
@app.route('/upload', methods=['POST'])
def upload_file():
    # Error gracefully handled....
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']

    # error due to no file upload....
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400


    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    #Output as JSON.....
    try:
        predicted_emotion = detect_emotion(filename)
        return jsonify({'message': 'File successfully uploaded', 'filename': filename, 'emotion': predicted_emotion}), 201
    except Exception as e:
        return jsonify({'message': f'Error detecting emotion: {str(e)}'}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
