from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('Image_classify.keras')  # Replace with the path to your saved model

# Define image dimensions and class labels
IMG_WIDTH = 180
IMG_HEIGHT = 180
CLASS_NAMES = ['class1', 'class2', 'class3']  # Replace with your actual class labels

UPLOAD_FOLDER = 'uploads'  # Directory to save uploaded images
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Preprocess the image
        image = load_img(file_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        image = img_to_array(image) / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict using the model
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        label = CLASS_NAMES[predicted_class]
        confidence = np.max(predictions) * 100

        return render_template('result.html', label=label, confidence=confidence, file_path=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
