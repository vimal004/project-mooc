import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('custom_model.h5')

# Load and preprocess the image
img_path = 'leaf.jpg'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(224, 224))  # Adjust target_size as per your model's input size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize the image

# Make a prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

# Print the result
print(f'Predicted class: {predicted_class}')