import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load model

model = load_model("imageclassifier3_with_history.keras")
classes = ['Bengin', 'Malignant', 'Normal']

# Function to make prediction
def predict(image):
    # Preprocess image
    image = image.resize((224, 224))
    image = np.expand_dims(np.array(image), axis=0)
    # Make prediction
    prediction = model.predict(image)[0]
    predicted_class = classes[np.argmax(prediction)]
    return predicted_class

# Streamlit app
st.title("Image Classifier")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Make prediction if file is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    prediction = predict(image)
    st.write("Prediction:", prediction)
