import sys, traceback
from flask import Flask, request, jsonify
import os
import hashlib
from pymongo import MongoClient
from werkzeug.utils import secure_filename
from deepface import DeepFace
from bson.binary import Binary
import numpy as np
from PIL import Image
import io
from flask_cors import CORS, cross_origin
import cv2
from scipy.signal import find_peaks
import werkzeug
import tempfile


# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client['facialAuth']
collection = db["facialData"]

app = Flask(__name__)

CORS(app)

# Create an 'images' directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

@app.route("/register", methods=["POST"])
def register():
    username = request.form.get("username").lower()
    image_file = request.files.get("image")

    if not username or not image_file:
        return jsonify({"error": "Missing username or image"}), 400

    if not image_file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return jsonify({"error": "Invalid image format"}), 400

    # Check if username already exists
    if collection.find_one({"username": username}):
        return jsonify({"error": "Username already exists"}), 400

    # Save image in the images folder
    filename = secure_filename(image_file.filename)
    image_path = os.path.join('images', f"{username}_{filename}")
    image_file.save(image_path)

    image_content = open(image_path, 'rb').read()
    image_hash = hashlib.sha256(image_content).hexdigest()

    user_data = {
        "username": username,
        "image_hash": image_hash,
        "image_data": Binary(image_content),
        "image_path": image_path  # Store image path for later reference
    }
    collection.insert_one(user_data)

    return jsonify({"message": "Registration successful"}), 201



@app.route("/login", methods=["POST"])
def login():
    models = DeepFace.build_model('VGG-Face')

    username = request.form.get("username")
    uploaded_image_file = request.files.get("image")
    heart_rate = request.form.get("heart_rate")
    if not username or not uploaded_image_file:
        return jsonify({"error": "Missing username or image"}), 400
    user = collection.find_one({"username": username})
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Save the uploaded image temporarily for processing
    filename = secure_filename(uploaded_image_file.filename)
    temp_upload_path = os.path.join('images', f"temp_{username}_{filename}")
    uploaded_image_file.save(temp_upload_path)

    try:
        result = DeepFace.verify(img1_path=temp_upload_path, img2_path=user["image_path"]) # verify if image provided is same as image in database
        if result["verified"]: # if image is verified
            if int(heart_rate) <= 100: # check if heart rate provided is less than 100
                try:
                    analysis = DeepFace.analyze(img_path=temp_upload_path, actions=['emotion'], enforce_detection=False) #check image provided for dominant emotion
                    emotion = analysis[0]['emotion']
                    dominant_emotion = max(emotion, key=emotion.get) # check for emotion that has the highest value 
                    if dominant_emotion in ['happy', 'neutral']:
                        return jsonify({"message": "Access granted", "verified": True, "dominant_emotion": dominant_emotion}), 200
                    else:
                        return jsonify({"message": "Access denied due to unsuitable emotional state", "verified": False, "dominant_emotion": dominant_emotion}), 403
                except Exception as e:
                    traceback.print_exc()
                    return jsonify({"error": "Failed to analyze emotion", "details": str(e)}), 500
            else:
                return jsonify({"message": "Heart rate check failed", "verified": False}), 403
        else:
            return jsonify({"error": "Access denied due to facial mismatch", "verified": False}), 403
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Failed to analyze image", "details": str(e)}), 500


UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def estimate_heart_rate(video_path, fps=30):
    # get and open uploaded video using opencv
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mean_intensities = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray_frame)
            mean_intensities.append(mean_intensity)
    finally:
        cap.release()

    intensities = np.array(mean_intensities)
    intensities = intensities - np.mean(intensities)
    peaks, _ = find_peaks(intensities, distance=fps//2)
    duration_in_seconds = frame_counts / fps
    heart_rate = len(peaks) * (60 / duration_in_seconds)

    return heart_rate

@app.route('/get_heart_rate', methods=['POST'])
def get_heart_rate():
    # check for video in request body
    if 'video' not in request.files:
        return jsonify({'error': 'No video part'}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected video'}), 400

    if video:
        filename = werkzeug.utils.secure_filename(video.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)

        video.save(video_path)

        heart_rate = estimate_heart_rate(video_path)
        if heart_rate is not None:
            os.remove(video_path)  # Clean up after processing
            return jsonify({'heart_rate': heart_rate}), 200
        else:
            os.remove(video_path)  # Clean up if processing failed
            return jsonify({'error': 'Could not process video'}), 500


if __name__ == "__main__":
    app.run(debug=True)