import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Task 1: Data Preparation (from Step 2) ---

base_folder = r'C:\Appl\fyp\data\cad-cardiac-mri-dataset'
IMG_SIZE = 128

normal_path = os.path.join(base_folder, 'Normal')
sick_path = os.path.join(base_folder, 'Sick')

image_data = []
labels = []

def process_images_from_path(path, label):
    print(f"\nProcessing images from: {path}")
    for dirpath, _, filenames in tqdm(list(os.walk(path))):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(dirpath, filename)
                    img = Image.open(img_path).convert('L').resize((IMG_SIZE, IMG_SIZE))
                    image_data.append(np.array(img))
                    labels.append(label)
                except Exception as e:
                    print(f"\nWarning: Could not process file {img_path}. Error: {e}")

process_images_from_path(normal_path, 0) # 0 for Normal
process_images_from_path(sick_path, 1)   # 1 for Sick

image_data = np.array(image_data)
labels = np.array(labels)

# --- Task 2: Split the Data ---

# Splitting data into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)

# --- Task 3: Prepare Data for the Model ---

# Normalize pixel values from 0-255 to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape data to include a color channel dimension (1 for grayscale)
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print("\n-------------------------")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print("-------------------------")


# --- Task 4: Build the CNN Model ---

model = models.Sequential([
    # 1st Convolutional Layer: Learns basic features like edges.
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D((2, 2)),

    # 2nd Convolutional Layer: Learns more complex features from the first layer.
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten the 2D image data into a 1D list to feed into the "brain" part.
    layers.Flatten(),
    
    # The "Brain" part of the network.
    layers.Dense(64, activation='relu'),
    # The Output Layer: Makes the final decision (0 for Normal, 1 for Sick).
    layers.Dense(1, activation='sigmoid')
])

model.summary()


# --- Task 5: Train the Model ---

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\nStarting model training...")
# We'll train for 10 "epochs" (passes through the data)
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
print("Model training complete!")


# --- Task 6: Evaluate the Model ---

print("\nEvaluating model performance...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

model.save('cardiac_mri_model.h5')
print("\nModel saved successfully as cardiac_mri_model.h5")