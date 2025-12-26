# Fashion-MNIST Model Demo Web App (Streamlit) — CNN vs VGG16 Transfer Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Computer%20Vision](https://img.shields.io/badge/Computer%20Vision-Image%20Classification-teal)
![Transfer%20Learning](https://img.shields.io/badge/Transfer%20Learning-VGG16-informational)

A lightweight **Streamlit web application** that visualizes and compares predictions from two trained models on **Fashion-MNIST**:
- **Custom CNN** (baseline model)
- **VGG16 Transfer Learning** (feature extraction / optional fine-tuning)

The app supports **image upload**, shows the input image, provides **class probabilities + predicted class**, and displays **training curves (loss/accuracy)** loaded from saved artifacts.

---

## Why this project
This project demonstrates practical skills in:
- Building a simple **ML web app** for inference and visualization
- Packaging and loading trained **TensorFlow/Keras** models (`.keras`)
- Handling **preprocessing** for different model input formats (28×28 grayscale vs 224×224 RGB for VGG16)
- Showing **probability distributions** and readable outputs for end users
- Logging and visualizing training history (**loss/accuracy**) from saved artifacts

---

## Features
Upload an image (jpg/png)  
Display the uploaded image  
Select model: **Custom CNN** vs **VGG16**  
Predict class + show probabilities (Top-K table + bar chart)  
Show training curves (loss/accuracy) from saved history JSON files  

---

## Project structure

