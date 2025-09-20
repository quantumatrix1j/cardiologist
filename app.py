import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- Configuration ---
IMG_SIZE = 128
MODEL_PATH = 'cardiac_mri_model.h5'

# --- Page Setup ---
st.set_page_config(
    page_title="Cardiac MRI Analyzer",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Cardiac MRI Analyzer")
st.write("Upload a cardiac MRI scan (JPG or PNG) to predict if the heart is normal or sick. This tool is a demonstration and not for medical diagnosis.")

# --- Model Loading ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_keras_model():
    """
    Loads the pre-trained Keras model.
    """
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except (OSError, IOError):
        st.error(f"Error: Model file not found at '{MODEL_PATH}'. Make sure the model is in the same directory as this script.")
        return None

model = load_keras_model()

# --- Prediction Function ---
def predict_image(image_file, loaded_model):
    """
    This function takes an uploaded image file, preprocesses it,
    and returns the model's prediction.
    """
    if loaded_model is None:
        return "Model not loaded.", 0.0

    try:
        # 1. Open and preprocess the image
        img = Image.open(image_file).convert('L').resize((IMG_SIZE, IMG_SIZE))

        # 2. Convert to NumPy array and normalize
        img_array = np.array(img) / 255.0

        # 3. Reshape for the model (add batch and channel dimensions)
        img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        # 4. Make a prediction
        prediction = loaded_model.predict(img_array)

        # 5. Interpret the result
        confidence = prediction[0][0]
        return "Sick", confidence
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return "Error", 0.0

# --- UI Elements ---
uploaded_file = st.file_uploader("Choose a cardiac MRI scan...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded MRI Scan', use_column_width=True)

    # Make prediction on button click
    if st.button('Analyze Scan'):
        with st.spinner('Analyzing...'):
            label, confidence = predict_image(uploaded_file, model)

            if label == "Sick":
                if confidence > 0.5:
                    st.error(f"Prediction: Sick (Confidence: {confidence*100:.2f}%)")
                    st.warning("This model suggests a potential abnormality. Please consult a cardiologist for a proper diagnosis.")
                else:
                    st.success(f"Prediction: Normal (Confidence: {(1-confidence)*100:.2f}%)")
                    st.info("This model suggests the scan is normal.")
            else:
                st.error("Could not process the image for prediction.")