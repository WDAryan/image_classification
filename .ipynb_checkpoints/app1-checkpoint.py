from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join('model', 'Image_classify.keras')
model = load_model(model_path)

# Class labels
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalapeno', 'kiwi', 'lemon', 'lettuce',
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'radish', 'soy beans', 'spinach', 'sweetcorn',
    'sweet potato', 'tomato', 'turnip', 'watermelon'
]

# Image dimensions
img_height = 180
img_width = 180

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    try:
        # Load and preprocess image
        image = load_img(file, target_size=(img_height, img_width))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, 0)  # Create batch axis

        # Predict
        predictions = model.predict(img_array)
        score = predictions[0]

        # Return the result as JSON
        return jsonify({
            "prediction": data_cat[np.argmax(score)],
            "confidence": float(np.max(score))
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
