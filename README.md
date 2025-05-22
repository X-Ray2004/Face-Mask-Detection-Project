# Face Mask Detection

## Overview
The **Face Mask Detection** project is a computer vision application that detects whether individuals in images or live video streams are wearing face masks. Built using Python, OpenCV, and scikit-learn, it combines feature extraction techniques (HOG, Chain Code, GLCM) with machine learning (SVM) for accurate classification. It supports batch image processing and real-time webcam detection, making it suitable for public health monitoring or compliance checking.

## Features
- **Batch Processing**: Detects and classifies mask-wearing in images, saving annotated results.
- **Real-Time Detection**: Identifies mask-wearing in live webcam feeds.
- **Feature Extraction**: Uses HOG, Chain Code, and GLCM for robust image analysis.
- **Model Training**: Compares SVM, Logistic Regression, Random Forest, and k-NN models.
- **Dimensionality Reduction**: Applies PCA to optimize feature vectors.

## Prerequisites
- Python 3.8+
- Libraries: Install via `pip`:
  ```bash
  pip install opencv-python numpy scikit-learn scikit-image joblib
