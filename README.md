# EUGuess 🌍
### Deep Learning for European Country Geolocation Using Random Images

This project tackles the challenge of image geolocation, specifically within Europe, by classifying images into one of 32 European countries using Convolutional Neural Networks (CNNs). The goal is to improve the model's ability to correctly predict the country based on randomly generated street-level images.

📅 **Project done at Georgia Tech (Fall 2022) as part of the CS4644 Deep Learning course**

---

## 📝 Project Overview

**EUGuess** approaches image geolocation by:
- **Dataset Generation**: Leveraging Google Street View API to capture images from random coordinates within the geographical bounds of European countries.
- **Classification Task**: Classifying these images using various CNN architectures.
- **Multi-Perspective Analysis**: Testing models with both single and multiple perspectives to gauge performance differences.

---

## 🔍 Problem Statement

Geolocating random images is complex due to the diversity of landscapes and similarities across borders. EUGuess simplifies this by focusing on a Europe-only dataset, aiming for high accuracy in country classification. Key challenges included:
- **Image Size**: High-resolution images required downscaling for manageable processing.
- **Dataset Size**: With limited resources, dataset size remained small (42,000 images), impacting training and generalization.

---

## 📊 Approach

1. **Image Processing**: 
   - Resize images to 64x64 pixels.
   - Convert images to tensors and store as pickle objects for faster loading.
2. **Model Selection**:
   - Evaluated several models: ResNet50, EfficientNetB4, GoogleNet, VGG16, and a custom CNN architecture.
3. **Optimization Techniques**:
   - Tuned hyperparameters, applied optimizers (e.g., Adam, SGD), and used schedulers to achieve improved accuracy.

---

## 🔬 Results

- Best performance achieved with **GoogleNet**, reaching a test accuracy of **25.1%** using three perspectives per location.
- Experimentation with new data, not seen in training, yielded an accuracy of **16%**.
- Demonstrated that multi-perspective inputs boost accuracy by approximately 50% compared to single perspectives.

---

## 🛠️ Technologies Used

- **Python** and **PyTorch**
- **Google Street View API** for dataset generation
- **torchvision** library for model experimentation

---

## 👥 Contributors

- Jorge Garcelán Gómez (jgarcelan3@gatech.edu)
- Luis Pérez Olid (lperez63@gatech.edu)
- Carlos Pérez Jiménez (cjimenez41@gatech.edu)

---

## 🎮 Interactive Game

Access our interactive game to experience the geolocation challenge with EUGuess images!

