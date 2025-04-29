# Image Classification of Fruits and Vegetables

## Overview
This project leverages **TensorFlow**, **Streamlit**, and **Keras** to build a machine learning model for classifying images of fruits and vegetables. It includes a trained model, a data pipeline for handling training, validation, and test datasets, and a user-friendly web app using **Streamlit** that enables users to upload images and classify them in real-time.

---
## Folder Structure
image_classification/
├── Fruits_Vegetables/       # Dataset directories (train, validation, test)
├── app.py                   # Streamlit web app script
├── model_train.py           # Script for training the model
├── Image_classify.keras     # Trained model file
├── requirements.txt         # Dependencies
├── feedback_images/         # Stores user feedback images


## Features
- **Trainable Neural Network**:
  - A Convolutional Neural Network (CNN) model is used for classification.
  - The model is trained on a custom dataset of fruits and vegetables.
- **Streamlit Web App**:
  - A web interface where users can upload an image and view the classification along with confidence scores.
  - Includes a feedback system to allow users to report misclassifications and provide correct labels.
- **Visualization**:
  - Displays training/validation accuracy and loss graphs for better understanding of the model's performance.
- **Feedback Handling**:
  - Saves incorrect classifications and correct labels for further improvements.

---

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.9+
- Pip (Python package manager)

### Steps to Set Up
1. Clone this repository:
    ```bash
    git clone https://github.com/WDAryan/image_classification.git
    cd image_classification
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Verify the `Image_classify.keras` model file is present in the project directory. If it's missing, train the model using the training script provided in this repository.

4. Launch the Streamlit web app:
    ```bash
    streamlit run app.py
    ```

---

## Dataset
The dataset contains images of fruits and vegetables and is organized into three directories:
- `train`: Training data
- `validation`: Validation data
- `test`: Test data

Images are resized to **180x180** pixels for uniformity.

---

## Model Architecture
The model is a **Sequential CNN** and consists of:
- Rescaling Layer
- Convolutional Layers with ReLU Activation
- MaxPooling Layers
- Dropout for Regularization
- Fully Connected Dense Layers
- Final Dense Layer for classification into `len(data_cat)` categories.

The model is compiled with:
- **Optimizer**: Adam
- **Loss Function**: SparseCategoricalCrossentropy
- **Metrics**: Accuracy

---

## How to Use
1. Open the Streamlit web app.
2. Upload an image of a fruit or vegetable (supported formats: `.jpg`, `.jpeg`, `.png`).
3. View the predicted label and confidence score.
4. Provide feedback if the prediction is incorrect:
   - Specify the correct label.
   - The app saves the image and label in a feedback directory for future improvements.

---

## Results
### Training and Validation
- **Accuracy**: Achieved high accuracy on training and validation datasets.
- **Loss**: Reduced significantly over epochs.

### Example Prediction:
Upload `Image_2.jpg`:
- Predicted Label: **Apple**
- Confidence: **98.45%**

---

## Feedback System
The web app features a feedback mechanism where users can:
- Confirm whether the prediction is correct.
- Provide the correct label if the prediction is wrong.
- Save the feedback for retraining or analysis.

---

## Future Work
- Model Improvement: Use feedback images to retrain and improve the model.
- Dataset Expansion: Add more fruits and vegetables to the dataset.
- Scalability: Optimize the app for deployment on cloud platforms.

---

## Technologies Used
- TensorFlow: Deep learning framework for training and inference.
- Keras: High-level API for building and training neural networks.
- Streamlit: Framework for building web apps.
- Python: Programming language.
- Matplotlib: Library for data visualization.

----
