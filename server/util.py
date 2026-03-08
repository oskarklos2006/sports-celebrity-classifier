import json
import joblib
import numpy as np
import cv2
import os
import pywt

# =========================
# Globals
# =========================
__model = None
__class_name_to_number = None
__class_number_to_name = None

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
CASCADE_DIR = os.path.join(
    PROJECT_ROOT,
    "model",
    "opencv",
    "haarcascades"
)

# =========================
# Load Haar Cascades
# =========================
face_cascade = cv2.CascadeClassifier(
    os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml")
)

eye_cascade = cv2.CascadeClassifier(
    os.path.join(CASCADE_DIR, "haarcascade_eye.xml")
)

if face_cascade.empty() or eye_cascade.empty():
    raise RuntimeError("Haar cascade files could not be loaded. Check paths.")

# =========================
# Load Model + Classes
# =========================
def load_saved_artifacts():
    global __model, __class_name_to_number, __class_number_to_name

    if __model is not None:
        return

    with open(os.path.join(ARTIFACTS_DIR, "class_dictionary.json"), "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    __model = joblib.load(os.path.join(ARTIFACTS_DIR, "saved_model.pkl"))


# =========================
# Face Cropping (EXACT LOGIC FROM NOTEBOOK)
# =========================
def get_cropped_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = img[y:y+h, x:x+w]
        return roi

    return None


# =========================
# Wavelet Transform (EXACT NOTEBOOK VERSION)
# =========================
def w2d(img, mode='db1', level=3):
    imArray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255.0

    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255.0
    imArray_H = np.uint8(imArray_H)

    return imArray_H


# =========================
# Image Preprocessing (MATCHES TRAINING)
# =========================
def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    # 🔴 FACE CROP (CRITICAL)
    cropped = get_cropped_face(img)
    if cropped is None:
        raise ValueError("No face detected")

    # Raw image
    scalled_raw_img = cv2.resize(cropped, (32, 32))

    # Wavelet image
    img_har = w2d(cropped, 'db1', 3)
    scalled_img_har = cv2.resize(img_har, (32, 32))

    combined = np.vstack((
        scalled_raw_img.reshape(32*32*3, 1),
        scalled_img_har.reshape(32*32, 1)
    ))

    return combined.reshape(1, -1).astype(float)


# =========================
# Prediction
# =========================
def predict_image(image_bytes):
    load_saved_artifacts()

    X = preprocess_image(image_bytes)
    probabilities = __model.predict_proba(X)[0]

    class_idx = int(np.argmax(probabilities))

    return {
        "player": __class_number_to_name[class_idx],
        "confidence": float(probabilities[class_idx]),
        "all_confidences": {
            __class_number_to_name[i]: float(probabilities[i])
            for i in range(len(probabilities))
        }
    }
