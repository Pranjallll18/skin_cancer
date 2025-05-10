import tensorflow as tf
from preprocess import preprocess_image
import cv2
import numpy as np



# Load the trained model
model = tf.keras.models.load_model("../../models/skin_cancer_cnn.h5")

def predict_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image could not be read. Is the file corrupted?")
        image = cv2.resize(image, (128, 128)) / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        result = "Malignant" if prediction[0] > 0.5 else "Benign"
        confidence = float(prediction[0]) if prediction[0] > 0.5 else 1 - float(prediction[0])
        return {"prediction": result, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

