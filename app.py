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

    # Preprocess the image
    # Resize the image to match the expected input shape of the model
    image = image.resize((256, 256))
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Normalize the pixel values (optional but often necessary)
    image_array = image_array / 255.0

    # Make predictions using the loaded model
    prediction = model.predict(np.expand_dims(image_array, axis=0))[0]

    # Display the prediction
    class_labels = ['Healthy', 'Early Blight', 'Late Blight']
    st.subheader(f"Prediction: {class_labels[np.argmax(prediction)]}")
    # st.subheader(prediction)
    st.markdown("---")
