import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import os

# Define constants
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Get the absolute path to the dataset and model directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATASET_DIR = os.path.join(BASE_DIR, "../../dataset/test")
MODEL_DIR = os.path.join(BASE_DIR, "../../models/skin_cancer_cnn.h5")

# Load test data
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Load trained model
model = tf.keras.models.load_model(MODEL_DIR)

# Evaluate model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
