import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2


# Load the saved model
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# Preprocess the image
def preprocess_image(img_path, target_size=(224, 224)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


# Make a prediction
def predict_image(model, img_path):
    if model is None:
        print("Model is not loaded.")
        return None, None

    # Preprocess the image
    img_array = preprocess_image(img_path)
    if img_array is None:
        return None, None

    # Make a prediction
    try:
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = class_names[predicted_class_index]
        confidence = np.max(predictions)
        return predicted_class, confidence
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None


# Define the class names
class_names = ["FMD", "IBK", "LSD"]

# Path to the model and image
model_path = "./model_mobilenetV2"  # Use the directory if saved in SavedModel format
image_path = "fmd.jpg"  # Replace with the path to your image

# Load the model
model = load_model(model_path)

# Make a prediction
if model is not None:
    predicted_class, confidence = predict_image(model, image_path)
    if predicted_class is not None:
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}")
