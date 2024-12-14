import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('plant_disease_model.keras')

# Class labels for the model's output
class_names = ['Early Blight', 'Healthy', 'Late Blight']

# Define the image size
IMG_SIZE = 224

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f0f0f5;
            color: #333;
            font-family: 'Arial', sans-serif;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 40px;
            text-align: center;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .stWrite {
            font-size: 18px;
            text-align: center;
            margin-bottom: 30px;
            color: #555;
        }
        .upload-box {
            border: 2px dashed #4CAF50;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
            background-color: #ffffff;
            margin-bottom: 20px;
        }
        .predict-button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
        }
        .predict-button:hover {
            background-color: #45a049;
        }
        .footer {
            position: fixed;
            bottom: 10px;
            right: 10px;
            color: #999;
            font-size: 16px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI components
st.markdown("<h1 class='stTitle'>Plant Disease Classification</h1>", unsafe_allow_html=True)
st.markdown("<p class='stWrite'>Upload an image of a plant leaf to detect its disease.</p>", unsafe_allow_html=True)

# File upload component with custom box
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image for prediction
    img = img.resize((IMG_SIZE, IMG_SIZE))  # Resize to model input size
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction when the button is clicked
    if st.button("Predict", key="predict", help="Click to predict the disease of the plant"):
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        # Display the results with some styling
        st.markdown(f"### **Prediction**: {predicted_class}", unsafe_allow_html=True)
        st.markdown(f"### **Confidence**: {confidence * 100:.2f}%", unsafe_allow_html=True)

# Footer with the name "Renuka" at the bottom-right
st.markdown("<p class='footer'>Renuka </p>", unsafe_allow_html=True)
