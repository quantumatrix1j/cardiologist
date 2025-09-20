import numpy as np
import tensorflow as tf
from PIL import Image
import os

# --- Configuration ---
IMG_SIZE = 128
MODEL_PATH = 'cardiac_mri_model.h5'
# --------------------

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

def predict_image(image_path):
    """
    This function takes a path to an image, preprocesses it, 
    and returns the model's prediction.
    """
    try:
        # 1. Open and preprocess the image
        img = Image.open(image_path).convert('L').resize((IMG_SIZE, IMG_SIZE))
        
        # 2. Convert to NumPy array and normalize
        img_array = np.array(img) / 255.0
        
        # 3. Reshape for the model (add batch and channel dimensions)
        img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        
        # 4. Make a prediction
        prediction = model.predict(img_array)
        
        # 5. Interpret the result
        # The output is a probability. > 0.5 means 'Sick', <= 0.5 means 'Normal'
        confidence = prediction[0][0]
        if confidence > 0.5:
            return f"Prediction: Sick (Confidence: {confidence*100:.2f}%)"
        else:
            return f"Prediction: Normal (Confidence: {(1-confidence)*100:.2f}%)"
            
    except FileNotFoundError:
        return "Error: Image file not found."
    except Exception as e:
        return f"An error occurred: {e}"

# --- Use the function ---
test_image_path = r'C:\Appl\fyp\data\cad-cardiac-mri-dataset\Sick\Directory_17\SR_29\IM00001.jpg' 

# Get the prediction
result = predict_image(test_image_path)
print(f"\nAnalyzing image: {os.path.basename(test_image_path)}")
print(result)