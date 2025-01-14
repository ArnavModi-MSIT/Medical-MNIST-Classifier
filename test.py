from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
import os

# Load the model
model = load_model('medical_mnist_cnn_model.keras')

# Define your class labels (same order as during training)
class_labels = ['AbdomenCT', 'BreastMRI', 'ChestCT', 'Hand', 'HeadCT']  # Update with your actual class names
label_encoder = LabelEncoder()
label_encoder.fit(class_labels)  # Fit the encoder with the correct labels

# Function to preprocess an image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    return img

# Test on a new image
image_path = r"C:\Coding\Medical MNIST\Images\ChestCT\001246.jpeg"  # Update with your image path
new_image = preprocess_image(image_path)

# Make a prediction
prediction = model.predict(new_image)
predicted_class_index = np.argmax(prediction)
predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]

print(f"Predicted Label: {predicted_label}")
