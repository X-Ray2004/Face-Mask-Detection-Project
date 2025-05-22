import cv2
import joblib
import numpy as np
import os
from preprocess import compute_hog, compute_chain_code, compute_glcm_features

def extract_features_for_single_image(img):
    hog = compute_hog(img)
    chain = compute_chain_code(img)
    glcm = compute_glcm_features(img)
    return np.concatenate((hog, chain, glcm))

folder_path = 'images'
output_folder = 'result'
os.makedirs(output_folder, exist_ok=True)

model = joblib.load("files/svm_face_mask_model.pkl")
pca = joblib.load("files/pca_model.pkl")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {filename}")
            continue

        height, width = image.shape[:2]
        if width < 200 or height < 200:
            scale = 200 / min(width, height)
            new_size = (int(width * scale), int(height * scale))
            image = cv2.resize(image, new_size)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print(f"{filename}: No faces detected")
            continue

        for (x, y, w, h) in faces:
            x = max(0, x)
            y = max(0, y)
            w = max(1, w)
            h = max(1, h)

            if x + w > gray.shape[1]:
                w = gray.shape[1] - x
            if y + h > gray.shape[0]:
                h = gray.shape[0] - y

            if w <= 0 or h <= 0:
                print(f"Invalid face region for {filename}, skipping this face.")
                continue

            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))
            face = cv2.normalize(face, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            features = extract_features_for_single_image(face)
            features = features.reshape(1, -1)

            reduced = pca.transform(features)
            pred = model.predict(reduced)[0]

            label = "Mask" if pred == 1 else "No Mask"
            color = (0, 255, 0) if pred == 1 else (0, 0, 255)

            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            print(f"{filename}: {label}")

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image)
