# Rice-Classification-Using-Artificial-Neural-Network-ANN
## 📌 Overview

This project implements a machine learning system to classify rice varieties based on morphological features using an Artificial Neural Network (ANN). The model is trained on a dataset containing measurements such as area, perimeter, axis lengths, and shape-related attributes extracted from rice grain images. The system achieves high classification accuracy and demonstrates the application of deep learning in agricultural quality analysis.

---

## 🚀 Features

* Exploratory Data Analysis (EDA) and visualization
* Data preprocessing and feature scaling
* ANN model built using TensorFlow/Keras
* Model evaluation using accuracy, confusion matrix, precision, recall, and F1-score
* Prediction on new user inputs
* Optional image-based feature extraction using OpenCV

---

## 📂 Dataset

The dataset contains morphological features of rice grains, including:

* Area
* Major Axis Length
* Minor Axis Length
* Eccentricity
* Convex Area
* Equivalent Diameter
* Extent
* Perimeter
* Roundness
* Aspect Ratio
* Class (Target)

---

## 🧠 Model Architecture

The Artificial Neural Network consists of:

* Input Layer (10 features)
* Hidden Layer 1 — 64 neurons (ReLU)
* Hidden Layer 2 — 32 neurons (ReLU)
* Output Layer — 1 neuron (Sigmoid)

Loss Function: Binary Crossentropy
Optimizer: Adam

---

## 📊 Results

The trained model achieved approximately **98.5% accuracy** on the test dataset, demonstrating strong classification performance.

Performance Metrics:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score

---

## 🖼️ Real-World Application

In practical applications, rice grain features such as area and perimeter can be extracted automatically from images using computer vision techniques. The trained ANN model can then classify the rice type based on these extracted features.

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* OpenCV (optional for image processing)

---

## ⚙️ Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python
## ▶️ Usage

### Train the Model

```python
python train_model.py
```

### Predict Using User Input

```python
python predict.py
```

### Predict Using Image

Upload a rice image and extract features using OpenCV before prediction.

## 📈 Workflow

Dataset → Preprocessing → Scaling → ANN Model → Training → Evaluation → Prediction

## 📌 Future Improvements

* CNN-based image classification directly from rice images
* Web application deployment using Streamlit or Flask
* Mobile-based rice classification system
* Real-time image capture integration

## ⭐ Contribution

Contributions, suggestions, and improvements are welcome!
