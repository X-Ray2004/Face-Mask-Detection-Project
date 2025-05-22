# 🧠 Face Mask Detection

**Face Mask Detection** is a computer vision application designed to detect whether individuals in images or live video streams are wearing face masks. It leverages machine learning techniques and image processing to extract facial features and classify them as **"With Mask"** or **"Without Mask"**.

The project supports **batch image processing** and **real-time webcam detection**, making it ideal for **public health monitoring**, **security systems**, or **compliance automation** in environments requiring mask usage.

---

## 🚀 Features

- **Batch Image Processing:** Detects and classifies mask-wearing status in folders of images.
- **Real-Time Detection:** Detects mask usage in live webcam video streams.
- **Feature Extraction:** Combines HOG, Chain Code, and GLCM features for robust facial representation.
- **Model Training:** Trains and compares SVM, Logistic Regression, Random Forest, and k-NN classifiers.
- **Dimensionality Reduction:** Uses PCA to enhance performance and reduce complexity.

---

## 📦 Prerequisites

- Python 3.8 or higher
- Webcam (for real-time detection)
- OpenCV's Haar Cascade file: `haarcascade_frontalface_default.xml`

Install required libraries:

```bash
pip install opencv-python numpy scikit-learn scikit-image joblib
📁 Project Structure
bash
Copy
Edit
face-mask-detection/
├── dataset/
│   ├── with_mask/              # Training images with masks
│   └── without_mask/           # Training images without masks
├── images/                     # Input images for batch processing
├── result/                     # Output images with annotations
├── files/                      # Saved features & models
│   ├── features_pca.npy
│   ├── labels.npy
│   ├── pca_model.pkl
│   └── svm_face_mask_model.pkl
├── preprocess.py               # Feature extraction functions
├── feature_extraction.py       # Feature extraction & PCA saving
├── model_training.py           # Train & evaluate classifiers
├── batch_processing.py         # Batch image detection
├── live_detection.py           # Real-time webcam detection
└── README.md                   # Project documentation
⚙️ Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/face-mask-detection.git
cd face-mask-detection
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Or install manually (as listed in prerequisites).

🧪 Usage
1. Feature Extraction
Extracts HOG, Chain Code, and GLCM features and applies PCA.

bash
Copy
Edit
python feature_extraction.py
Input: Images in dataset/with_mask/ and dataset/without_mask/

Output: features_pca.npy, labels.npy, pca_model.pkl in files/

2. Model Training
Trains 4 classifiers and saves the best (SVM by default).

bash
Copy
Edit
python model_training.py
Output: svm_face_mask_model.pkl in files/

Example:

yaml
Copy
Edit
SVM Accuracy: 94.00% | Time: 12.34s
Logistic Regression: 89.10% | Time: 8.76s
Random Forest: 90.80% | Time: 15.67s
k-NN: 88.00% | Time: 0.02s
3. Batch Image Processing
Annotates images in the images/ directory.

bash
Copy
Edit
python batch_processing.py
Output: Annotated images in result/ folder

4. Real-Time Detection
Performs real-time mask detection via webcam.

bash
Copy
Edit
python live_detection.py
Press q to quit the live feed.

🧠 How It Works
🔍 Feature Extraction
Grayscale + resize images to 64x64

Extract:

HOG: gradient directions/magnitudes

Chain Code: edge direction sequences

GLCM: texture metrics (contrast, energy, etc.)

Combine and reduce with PCA

🤖 Model Training
Classifiers: SVM (RBF), Logistic Regression, Random Forest, k-NN

Best performer (SVM, 94%) is saved for detection

📸 Detection Pipeline
Detect face using Haar Cascade

Extract + reduce features

Predict class using trained SVM

Display result with bounding boxes

📊 Performance Summary
Model	Accuracy	Training Time
SVM (RBF Kernel)	94.00%	12.34 s
Logistic Regression	89.10%	8.76 s
Random Forest	90.80%	15.67 s
k-NN (k=5)	88.00%	0.02 s

⚠️ Limitations
Requires clear, well-lit images.

Accuracy may drop with occlusions or poor lighting.

Webcam performance varies with hardware.

Requires a balanced, diverse dataset for generalization.

🌱 Future Improvements
Multi-mask-type classification (e.g., cloth, N95).

Use data augmentation to improve generalization.

Real-time performance tuning for low-end devices.

Deep learning integration (CNN, MobileNet, etc.).

🤝 Contributing
Contributions are welcome!
To contribute:

Fork the repo

Create a branch: git checkout -b feature-branch

Commit your changes: git commit -m "Add feature"

Push the branch: git push origin feature-branch

Open a Pull Request

📜 License
This project is licensed under the MIT License.

🙏 Acknowledgments
Built using: OpenCV, scikit-learn, scikit-image

Inspired by real-world public health applications

Thanks to the open-source community 💙

yaml
Copy
Edit

---

Let me know if you'd like this as a downloadable file or want help turning this into a published GitHub repository.
