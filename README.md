<div align="center">

# 🍎🥦 Fruit & Vegetable Image Classifier

### Deep Learning · Computer Vision · Real-Time Web App

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.5%2B-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Classify 36 fruits & vegetables instantly** — upload any image and get a real-time prediction with a confidence score, powered by a custom-trained Convolutional Neural Network.

</div>

---

## 📌 Table of Contents

- [✨ Features](#-features)
- [🧠 Model Architecture](#-model-architecture)
- [📂 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [🖼️ How to Use](#️-how-to-use)
- [🍓 Supported Categories](#-supported-categories)
- [📊 Results](#-results)
- [🔁 Feedback System](#-feedback-system)
- [🔮 Future Work](#-future-work)
- [🛠️ Tech Stack](#️-tech-stack)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Real-Time Classification** | Upload a `.jpg`, `.jpeg`, or `.png` image and get an instant prediction |
| 📈 **Confidence Score** | See how confident the model is in its prediction (e.g., 98.45%) |
| 💬 **Feedback System** | Report wrong predictions and submit the correct label for future retraining |
| 📊 **Training Visualization** | Accuracy and loss graphs showing model performance over epochs |
| 🌐 **Web App** | Clean, user-friendly Streamlit interface — no coding required to use |

---

## 🧠 Model Architecture

The classifier uses a **Sequential Convolutional Neural Network (CNN)** built with Keras:

```
Input (180×180 RGB Image)
        ↓
Rescaling Layer          — Normalize pixel values to [0, 1]
        ↓
Conv2D + ReLU            — Feature extraction
        ↓
MaxPooling2D             — Spatial downsampling
        ↓
Conv2D + ReLU            — Deeper feature extraction
        ↓
MaxPooling2D
        ↓
Conv2D + ReLU
        ↓
MaxPooling2D
        ↓
Dropout                  — Regularization to reduce overfitting
        ↓
Flatten → Dense (ReLU)   — Fully connected layers
        ↓
Dense (36 classes)       — Final classification output
```

**Compilation settings:**
- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Metric:** Accuracy

---

## 📂 Project Structure

```
image_classification/
│
├── 📄 app.py                    # Streamlit web application
├── 📄 app1.py                   # Alternate app configuration
├── 📄 Image_Class_Model.ipynb   # Model training notebook
├── 🧠 Image_classify.keras      # Pre-trained model weights
│
├── 📁 Fruits_Vegetables/        # Full dataset
│   ├── train/                   # Training images
│   ├── validation/              # Validation images
│   └── test/                    # Test images
│
├── 📁 feedback_images/          # User-submitted feedback images
├── 📁 fruit_vegetable_feedback/ # Processed feedback storage
│
├── 📄 requirements.txt          # Python dependencies
├── 📄 runtime.txt               # Python runtime version
└── 📄 procfile.txt              # Process configuration
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9+**
- pip package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/WDAryan/image_classification.git
cd image_classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the web app
streamlit run app.py
```

> ✅ The pre-trained model file `Image_classify.keras` is included in the repository — no training required to get started!

---

## 🖼️ How to Use

1. **Launch** the Streamlit app with `streamlit run app.py`
2. **Upload** an image of any fruit or vegetable (`.jpg`, `.jpeg`, `.png`)
3. **View** the predicted label and confidence score instantly
4. **Give Feedback** if the prediction is wrong:
   - Select **"No"** under the feedback section
   - Enter the correct label
   - Click **"Submit Feedback"** — the image is saved for future model improvement

---

## 🍓 Supported Categories

The model can classify **36 fruits and vegetables**:

| 🍎 | 🍌 | 🫑 | 🥦 | 🌽 | 🍇 |
|---|---|---|---|---|---|
| Apple | Banana | Bell Pepper | Cabbage | Corn | Grapes |
| Beetroot | Capsicum | Carrot | Cauliflower | Cucumber | Eggplant |
| Garlic | Ginger | Jalapeno | Kiwi | Lemon | Lettuce |
| Mango | Onion | Orange | Paprika | Pear | Peas |
| Pineapple | Pomegranate | Potato | Radish | Soy Beans | Spinach |
| Sweetcorn | Sweet Potato | Tomato | Turnip | Watermelon | Chilli Pepper |

---

## 📊 Results

### Training & Validation Performance

- ✅ **High accuracy** achieved on both training and validation datasets
- 📉 **Loss** reduced consistently over training epochs

### Example Prediction

| Input Image | Predicted Label | Confidence |
|---|---|---|
| `Image_2.jpg` | 🍎 **Apple** | **98.45%** |

---

## 🔁 Feedback System

The app includes a built-in feedback loop to continuously improve the model:

```
User uploads image
       ↓
Model predicts label
       ↓
User confirms or corrects prediction
       ↓
Feedback image saved to /feedback_images
       ↓
Use saved data to retrain & improve the model
```

---

## 🔮 Future Work

- [ ] **Model Retraining** — Use collected feedback images to retrain and improve accuracy
- [ ] **Dataset Expansion** — Add more categories beyond the current 36
- [ ] **Cloud Deployment** — Host the app on Streamlit Cloud / Heroku / AWS
- [ ] **Mobile Support** — Optimize the UI for mobile browsers
- [ ] **REST API** — Expose model inference as a REST API endpoint

---

## 🛠️ Tech Stack

<div align="center">

| Technology | Purpose |
|---|---|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | Core programming language |
| ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white) | Deep learning framework |
| ![Keras](https://img.shields.io/badge/-Keras-D00000?logo=keras&logoColor=white) | High-level model API |
| ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white) | Web application framework |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) | Numerical computing |
| ![Pillow](https://img.shields.io/badge/-Pillow-3776AB?logo=python&logoColor=white) | Image processing |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?logo=python&logoColor=white) | Data visualization |

</div>

---

<div align="center">

**Developed with ❤️ by [Aryan](https://github.com/WDAryan) · Powered by Streamlit & TensorFlow**

⭐ Star this repo if you found it useful!

</div>
