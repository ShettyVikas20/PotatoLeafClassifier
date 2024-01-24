import streamlit as st
import joblib
from PIL import Image
import numpy as np

# Load the pickled model
model = joblib.load('model.pkl')

# Streamlit app
st.title("Potato Plant Leaf Classification")

# File upload widget
uploaded_file = st.file_uploader("Choose a potato leaf image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image (modify based on your model's preprocessing steps)
    # Example: Convert the image to a numpy array
    image_array = np.array(image)
    # Add more preprocessing steps as needed

    # Make predictions using the loaded model
    prediction = model.predict(image_array.reshape(1, -1))[0]

    # Display the prediction
    class_labels = ['Healthy', 'Early Blight', 'Late Blight']
    st.write(f"Prediction: {class_labels[prediction]}")
