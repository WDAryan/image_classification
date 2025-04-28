import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
import os

# Set up Streamlit app
st.set_page_config(page_title="Fruit & Vegetable Classifier", page_icon="🍏", layout="centered")
st.title("Fruit & Vegetable Image Classifier")
st.write("Upload an image of a fruit or vegetable, and our model will classify it with an accuracy score!")

# Cache the model to avoid reloading it
@st.cache_resource
def load_model_cached():
    try:
        return load_model('Image_classify.keras')
    except Exception as e:
        st.error("Error loading the model. Please check the model file.")
        st.stop()

model = load_model_cached()

# List of categories (fruits and vegetables)
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
    'grapes', 'jalapeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange',
    'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'radish',
    'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# Image settings
img_height = 180
img_width = 180

# Folder to save feedback images
feedback_folder = "fruit_vegetable_feedback"
os.makedirs(feedback_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Image uploader widget
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    try:
        # Open the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Resize and normalize the image
        image = image.resize((img_height, img_width))
        img_arr = np.array(image) / 255.0  # Normalize pixel values
        img_bat = np.expand_dims(img_arr, axis=0)

        # Model prediction
        predict = model.predict(img_bat)
        score = tf.nn.softmax(predict)

        # Display prediction result
        st.subheader("Prediction Results")
        predicted_class = data_cat[np.argmax(score)]
        confidence = np.max(score) * 100
        st.write(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f}%)")

        # Ensure feedback mechanism works
        st.subheader("Feedback on Prediction")
        is_correct = st.radio("Is the prediction correct?", ["Yes", "No"], key="feedback")  # Radio button

        if is_correct == "No":
            correct_label = st.text_input("What is the correct label?", key="correct_label")  # Text input
            if correct_label:
                # Save the image to the feedback folder with the correct label
                save_path = os.path.join(feedback_folder, f"{correct_label}_{uploaded_image.name}")
                image.save(save_path)
                st.success(f"Image saved to {save_path} with the label '{correct_label}'!")

    except Exception as e:
        st.error(f"Error processing the image: {e}")
else:
    st.write("Please upload an image to begin!")
