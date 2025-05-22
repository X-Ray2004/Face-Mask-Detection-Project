Face Mask Detection
Overview
The Face Mask Detection project is a computer vision application designed to detect whether individuals in images or live video streams are wearing face masks. It leverages machine learning techniques and image processing to extract features from facial images and classify them as "With Mask" or "Without Mask." The project is built using Python, OpenCV, and scikit-learn, and supports both batch image processing and real-time webcam detection.
This project is ideal for applications in public health monitoring, security systems, or automated compliance checking in environments requiring mask-wearing.
Features

Batch Image Processing: Processes a folder of images to detect faces and classify mask-wearing status, saving annotated results.
Real-Time Detection: Uses a webcam to detect and classify mask-wearing in real-time video streams.
Feature Extraction: Combines Histogram of Oriented Gradients (HOG), Chain Code, and Gray-Level Co-occurrence Matrix (GLCM) features for robust classification.
Model Training: Trains and compares multiple machine learning models (SVM, Logistic Regression, Random Forest, k-NN) to select the best performer.
Dimensionality Reduction: Applies Principal Component Analysis (PCA) to reduce feature dimensions for efficient processing.

Prerequisites
Before running the project, ensure you have the following installed:

Python 3.8 or higher
Required Python libraries (install via pip):pip install opencv-python numpy scikit-learn scikit-image joblib


A webcam (for real-time detection)
OpenCV's Haar Cascade XML file (haarcascade_frontalface_default.xml), included with OpenCV installation.

Project Structure
face-mask-detection/
├── dataset/
│   ├── with_mask/              # Images of people wearing masks
│   └── without_mask/           # Images of people not wearing masks
├── images/                     # Input images for batch processing
├── result/                     # Output images with annotations
├── files/                      # Saved models and features
│   ├── features_pca.npy        # PCA-reduced features
│   ├── labels.npy             # Labels for training data
│   ├── pca_model.pkl          # PCA model
│   └── svm_face_mask_model.pkl # Trained SVM model
├── preprocess.py               # Feature extraction functions
├── batch_processing.py         # Script for batch image processing
├── live_detection.py           # Script for real-time detection
├── feature_extraction.py       # Script for extracting and saving features
├── model_training.py           # Script for training and comparing models
└── README.md                   # Project documentation

Installation

Clone the Repository:
git clone https://github.com/your-username/face-mask-detection.git
cd face-mask-detection


Install Dependencies:
pip install -r requirements.txt

Alternatively, install the required libraries manually as listed in the Prerequisites section.

Prepare the Dataset:

Place images in the dataset/with_mask/ and dataset/without_mask/ folders for training.
Ensure images are in .jpg or .png format.


Download Haar Cascade:

The haarcascade_frontalface_default.xml file is included with OpenCV. Ensure it is accessible via cv2.data.haarcascades.



Usage
The project consists of four main scripts, each serving a specific purpose:
1. Feature Extraction (feature_extraction.py)
Extracts HOG, Chain Code, and GLCM features from the dataset, applies PCA, and saves the results.
python feature_extraction.py


Input: Images in dataset/with_mask/ and dataset/without_mask/.
Output: Saves features (features_pca.npy), labels (labels.npy), and PCA model (pca_model.pkl) in the files/ directory.

2. Model Training (model_training.py)
Trains and compares four machine learning models (SVM, Logistic Regression, Random Forest, k-NN) and saves the best model (SVM).
python model_training.py


Output: Saves the trained SVM model as svm_face_mask_model.pkl in the files/ directory.
Performance: Prints accuracy and training time for each model. Example:SVM Accuracy: 94.00% | Training Time: 12.3400 seconds
Logistic Regression Accuracy: 89.10% | Training Time: 8.7600 seconds
Random Forest Accuracy: 90.80% | Training Time: 15.6700 seconds
K-Nearest Neighbors Accuracy: 88.00% | Training Time: 0.0200 seconds



3. Batch Image Processing (batch_processing.py)
Processes images in the images/ folder, detects faces, classifies mask-wearing status, and saves annotated images.
python batch_processing.py


Input: Images in the images/ folder.
Output: Annotated images saved in the result/ folder with rectangles and labels ("Mask" or "No Mask").

4. Real-Time Detection (live_detection.py)
Uses a webcam to detect and classify mask-wearing in real-time.
python live_detection.py


Controls: Press q to exit the webcam feed.
Output: Displays a live video feed with rectangles and labels ("With Mask" or "Without Mask") around detected faces.

How It Works

Feature Extraction:

Images are preprocessed (converted to grayscale, resized to 64x64, normalized).
Features are extracted using:
HOG: Captures gradient directions and magnitudes.
Chain Code: Analyzes edge patterns.
GLCM: Computes texture features (contrast, dissimilarity, homogeneity, energy, correlation).


Features are combined into a single feature vector and reduced using PCA.


Model Training:

Four models are trained on PCA-reduced features.
The SVM model with RBF kernel is selected due to its high accuracy (94%).


Detection:

Faces are detected using OpenCV's Haar Cascade classifier.
Features are extracted from detected faces, reduced by PCA, and classified using the trained SVM model.
Results are visualized with bounding boxes and labels.



Performance
The project evaluates four machine learning models:

SVM (RBF Kernel): 94% accuracy, robust for complex data.
Logistic Regression: 89.1% accuracy, fast but less effective for non-linear data.
Random Forest: 90.8% accuracy, good for complex data but slower.
k-NN (k=5): 88% accuracy, fast training but slower prediction.

The SVM model is used for inference due to its superior accuracy.
Limitations

Requires clear, well-lit images for accurate face detection.
Performance may degrade with low-quality images or complex backgrounds.
Real-time detection depends on the webcam's quality and processing power.
The dataset should be diverse to ensure robust generalization.

Future Improvements

Add support for detecting multiple mask types (e.g., cloth, surgical, N95).
Implement data augmentation to improve model robustness.
Optimize real-time detection for lower-end hardware.
Explore deep learning models (e.g., CNNs) for potentially higher accuracy.

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes and commit (git commit -m "Add feature").
Push to the branch (git push origin feature-branch).
Create a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Built using OpenCV, scikit-learn, and scikit-image.
Inspired by public health needs for automated mask detection.
Thanks to the open-source community for providing robust libraries and tools.

