import cv2
import numpy as np

def preprocess_image(image_path, img_size=(128, 128)):
    """
    Reads and preprocesses an image for prediction.
    - Loads the image using OpenCV
    - Resizes and scales it
    - Adds a batch dimension
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unreadable: " + image_path)

    image = cv2.resize(image, img_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image
