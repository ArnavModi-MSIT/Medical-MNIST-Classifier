from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
import os

model = load_model('medical_mnist_cnn_model.keras')

class_labels = ['AbdomenCT', 'BreastMRI', 'ChestCT', 'Hand', 'HeadCT']
label_encoder = LabelEncoder()
label_encoder.fit(class_labels)

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

image_path = r"C:\Coding\Medical MNIST\Images\ChestCT\001246.jpeg"
new_image = preprocess_image(image_path)

prediction = model.predict(new_image)
predicted_class_index = np.argmax(prediction)
predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]

print(f"Predicted Label: {predicted_label}")