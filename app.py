import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load a pre-trained model or your custom model
@st.cache_resource
def load_model():
    # For example, using MobileNetV2 pre-trained on ImageNet
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    return model

model = load_model()

# Preprocess the image
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # Resize for MobileNetV2
    image_array = np.array(image)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Decode predictions to get labels
def decode_predictions(predictions):
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)
    return decoded

# Streamlit app
st.title("Bird Classification App 🐦")
st.write("Upload an image of a bird, and the app will classify it for you!")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    st.write("Classifying...")
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    decoded_predictions = decode_predictions(predictions)

    # Display predictions
    st.write("### Top Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
        st.write(f"{i+1}. {label}: {score:.4f}")
