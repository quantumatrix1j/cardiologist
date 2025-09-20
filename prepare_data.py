import os
import numpy as np
from PIL import Image
from tqdm import tqdm # For the progress bar

# --- Configuration ---
# Your main project folder containing the dataset
base_folder = r'C:\Appl\fyp\data\cad-cardiac-mri-dataset'
# We will resize all images to this square dimension
IMG_SIZE = 128 
# --------------------

normal_path = os.path.join(base_folder, 'Normal')
sick_path = os.path.join(base_folder, 'Sick')

# This is where we will store our processed data
# A list for the image data (as arrays) and a list for the labels (0 or 1)
image_data = []
labels = []

def process_images_from_path(path, label):
    """
    This function walks through a directory, processes each image, 
    and adds the data and label to our master lists.
    """
    print(f"\nProcessing images from: {path}")
    # Use tqdm to show a progress bar
    for dirpath, _, filenames in tqdm(list(os.walk(path))):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    # Get the full path of the image
                    img_path = os.path.join(dirpath, filename)
                    
                    # Open, convert to grayscale, and resize the image
                    img = Image.open(img_path).convert('L').resize((IMG_SIZE, IMG_SIZE))
                    
                    # Add the image data and label to our lists
                    image_data.append(np.array(img))
                    labels.append(label)
                except Exception as e:
                    print(f"\nWarning: Could not process file {img_path}. Error: {e}")

# Process the 'Normal' images (label = 0)
process_images_from_path(normal_path, 0)

# Process the 'Sick' images (label = 1)
process_images_from_path(sick_path, 1)

# --- Convert our Python lists into NumPy arrays ---
# This is the standard format for machine learning frameworks like TensorFlow/Keras
image_data = np.array(image_data)
labels = np.array(labels)

# --- Final Check ---
print("\n-------------------------")
print("Data Preparation Complete!")
print(f"Total images processed: {len(image_data)}")
print(f"Shape of the image data array: {image_data.shape}")
print(f"Shape of the labels array: {labels.shape}")
print("-------------------------")

# The shape (Num_Images, Height, Width) tells us our data is ready!
# For example, (500, 128, 128) means 500 images, each 128x128 pixels.