# 🧠 Autism Detection Using Facial Images (VGG19 Model)

This project presents a deep learning approach for **autism detection** using **facial images**. It leverages a fine-tuned **VGG19** convolutional neural network model to classify images as **Autistic** or **Non-Autistic** with an achieved accuracy of **87%**.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Results](#results)

---

## 🔍 Overview

Autism Spectrum Disorder (ASD) is a developmental disorder that affects communication and behavior. Early diagnosis is crucial for timely intervention. In this project, a **facial image-based classifier** is developed using **transfer learning** with the VGG19 model to detect signs of autism.

---

## 📁 Dataset

The dataset used contains labeled facial images of:
- **Autistic children**
- **Non-Autistic (typical) children**

> 📌 *Dataset source is either publicly available or self-curated. (Update this section with the actual dataset link or description.)*

---

## 🧠 Model Architecture

- **Base model**: [VGG19](https://keras.io/api/applications/vgg/#vgg19-function) (pre-trained on ImageNet)
- **Custom layers**:
  - GlobalAveragePooling2D
  - Dense layers with ReLU activation
  - Dropout for regularization
  - Output layer with Sigmoid activation (for binary classification)

> Fine-tuning was applied to selected top layers of VGG19 to enhance domain-specific learning.

---

## ⚙️ Training Details

- **Loss Function**: Binary Crossentropy  
- **Optimizer**: Adam  
- **Metrics**: Accuracy  
- **Epochs**: _(e.g., 20)_  
- **Batch size**: _(e.g., 32)_  
- **Validation Split**: _(e.g., 20%)_

---

## 📈 Results

- **Training Accuracy**: ~87%
- **Validation Accuracy**: ~85–87%
- **Loss Curves**: (include graph if available)
- **Confusion Matrix**: (optional)

> The model shows promising results and generalizes well on unseen test data.

---


