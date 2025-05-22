import cv2
import numpy as np
import joblib
from preprocess import compute_hog, compute_chain_code, compute_glcm_features

def resize_with_padding(image, target_size=(64, 64)):
    h, w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_img = cv2.resize(image, (new_w, new_h))

    padded_img = np.zeros((target_h, target_w), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img

    return padded_img

def predict_mask_live(flip_image=False):
    pca_model = joblib.load("files/pca_model.pkl")
    svm_model = joblib.load("files/svm_face_mask_model.pkl")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if flip_image:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            if x < 0 or y < 0 or x+w > gray.shape[1] or y+h > gray.shape[0]:
                continue

            face_img = gray[y:y+h, x:x+w]
            face_img = resize_with_padding(face_img, (64, 64))
            norm_face = cv2.normalize(face_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            hog = compute_hog(norm_face)
            chain = compute_chain_code(norm_face)
            glcm = compute_glcm_features(norm_face)

            features = np.concatenate((hog, chain, glcm)).reshape(1, -1)
            features_pca = pca_model.transform(features)
            pred = svm_model.predict(features_pca)[0]

            label = "With Mask" if pred == 1 else "Without Mask"
            color = (0, 255, 0) if pred == 1 else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Mask Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

predict_mask_live(flip_image=True)
