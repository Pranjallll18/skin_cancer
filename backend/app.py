from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
model = tf.keras.models.load_model("../models/skin_cancer_cnn.h5")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128)) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    result = "Malignant" if prediction[0] > 0.5 else "Benign"
    confidence = float(prediction[0]) if prediction[0] > 0.5 else 1 - float(prediction[0])
    return {"prediction": result, "confidence": confidence}

@app.route("/predict", methods=["POST"])
def predict():
    print("📥 Prediction request received")
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result = predict_image(filepath)

    if "error" in result:
        print("❌ Prediction error:", result["error"])
        return jsonify(result), 500

    return jsonify(result)
