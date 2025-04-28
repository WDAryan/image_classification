import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
import os

# Streamlit page configuration
st.set_page_config(page_title="Fruit & Vegetable Classifier", page_icon="🍏", layout="centered")
st.title("Fruit & Vegetable Image Classifier")
st.write("Upload an image of a fruit or vegetable, and our model will classify it with an accuracy score!")

# Load the trained model with error handling
try:
    model = load_model('Image_classify.keras')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# List of categories
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
    'grapes', 'jalapeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange',
    'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'radish',
    'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# Image uploader widget
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    try:
        # Open and display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Preprocess the image
        img_height, img_width = 180, 180
        image = image.resize((img_height, img_width))
        img_arr = np.array(image)
        img_bat = np.expand_dims(img_arr, axis=0)

        # Model prediction
        predict = model.predict(img_bat)
        score = tf.nn.softmax(predict)

        # Display prediction results
        predicted_class = data_cat[np.argmax(score)]
        confidence = np.max(score) * 100

        st.subheader(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")

        # Feedback section
        st.markdown("### Feedback")
        feedback = st.radio("Is the prediction correct?", ["Yes", "No"])

        if feedback == "No":
            correct_label = st.text_input("What is the correct label?")
            if st.button("Submit Feedback"):
                # Feedback directory setup
                feedback_dir = "feedback_images"
                if not os.path.exists(feedback_dir):
                    os.makedirs(feedback_dir)  # Create the feedback directory
                    st.write(f"Directory '{feedback_dir}' created!")

                # Save the image with the correct label
                save_path = os.path.join(feedback_dir, f"{correct_label}_{uploaded_image.name}")
                image.save(save_path)
                st.success("Thank you for your feedback! The image has been saved for future improvements.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
else:
    st.write("Please upload an image to begin!")

# Footer
st.markdown("---")
st.markdown("Developed by Aryan | Powered by Streamlit & TensorFlow")
