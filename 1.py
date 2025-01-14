from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import matplotlib.pyplot as plt

# Directory containing image folders
image_dir = r"C:\Coding\Medical MNIST\Images"

def load_small_sample(image_dir, sample_fraction=0.1, augment_count=5):
    images = []
    labels = []
    
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    for folder in os.listdir(image_dir):
        folder_path = os.path.join(image_dir, folder)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            sample_size = max(1, int(len(files) * sample_fraction))  # Use 1% or at least 1 image
            sampled_files = files[:sample_size]
            
            for img_file in sampled_files:
                if img_file.endswith(".jpeg"):  # Assuming .jpeg format
                    img_path = os.path.join(folder_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Grayscale
                    img = img.astype("float32") / 255.0  # Normalize
                    img = np.expand_dims(img, axis=-1)  # Add channel dimension
                    
                    images.append(img)
                    labels.append(folder)  # Use folder name as label
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Augmentation
    augmented_images = []
    augmented_labels = []
    for img, label in zip(images, labels):
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        aug_iter = datagen.flow(img, batch_size=1)
        for _ in range(augment_count):  # Generate `augment_count` images per sample
            augmented_image = next(aug_iter)[0].squeeze()  # Remove batch dimension
            augmented_images.append(augmented_image)
            augmented_labels.append(label)
    
    return np.array(augmented_images), np.array(augmented_labels)

# Load and augment small sample
augmented_images, augmented_labels = load_small_sample(image_dir)

# Encode labels
label_encoder = LabelEncoder()
augmented_labels = label_encoder.fit_transform(augmented_labels)

# Split into train/validation sets
X_train, X_val, y_train, y_val = train_test_split(
    augmented_images, augmented_labels, test_size=0.2, random_state=42
)

# Convert labels to one-hot encoding
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the model
model.save('medical_mnist_cnn_model.keras')
