# 🏅 Sports Celebrity Image Classifier

A full-stack computer vision project that identifies football celebrities from images. Upload a photo on the website, and the model will predict which player it is. Built with OpenCV, scikit-learn, Flask, and a custom frontend.

---

## 📌 Overview

Given a photo of a person, the app detects their face, verifies both eyes are visible, extracts features using wavelet transforms, and classifies which football celebrity they are. The result is displayed instantly on the webpage.

**Celebrities the model can identify:**
- 🧤 Thibaut Courtois
- ⚽ Cristiano Ronaldo
- ⚽ Paulo Dybala
- ⚽ Toni Kroos
- ⚽ Lionel Messi
- ⚽ Mohamed Salah
- ⚽ Neymar Jr.
- ⚽ Paul Pogba

---

## 🏗️ Project Structure

```
sports-celebrity-classifier/
├── model/
│   ├── sports_person_clasifier.ipynb  # Training notebook
│   ├── saved_model.pkl                # Trained SVM model
│   ├── class_dictionary.json          # Label mappings
│   ├── images/                        # Training images
│   └── test_images/                   # Test images
├── server/
│   ├── server.py                      # Flask backend
│   ├── util.py                        # Helper functions
│   └── artifacts/                     # Model artifacts
└── UI/
    ├── index.html                     # Frontend webpage
    ├── script.js                      # JavaScript logic
    └── style.css                      # Styling
```

---

## 🔧 Tech Stack

- **Computer Vision** — OpenCV (Haar cascades for face & eye detection)
- **Feature Extraction** — PyWavelets (Haar wavelet transforms)
- **Machine Learning** — scikit-learn (SVM, Random Forest, Logistic Regression)
- **Backend** — Flask (Python)
- **Frontend** — HTML, CSS, JavaScript

---

## 🧪 ML Pipeline

1. **Face Detection** — OpenCV Haar cascades detect faces in uploaded images
2. **Eye Verification** — Only images with both eyes visible are accepted
3. **Feature Extraction** — Raw pixel data combined with wavelet-transformed features
4. **Model Training** — SVM, Random Forest and Logistic Regression compared
5. **Hyperparameter Tuning** — RandomizedSearchCV with 3-fold cross-validation
6. **Best Model** — SVM saved as `saved_model.pkl`

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/oskarklos2006/sports-celebrity-classifier.git
cd sports-celebrity-classifier
```

### 2. Install dependencies
```bash
pip install flask opencv-python scikit-learn numpy pywavelets joblib
```

### 3. Start the Flask server
```bash
cd server
python server.py
```

### 4. Open the frontend
Open `UI/index.html` in your browser and upload a photo to get a prediction.

---

## 🌐 How the App Works

1. User uploads an image on the webpage
2. JavaScript sends the image as base64 to the Flask API
3. Flask processes the image using OpenCV and the trained SVM model
4. The predicted celebrity name is returned and displayed on the page

---

## 👤 Author

**Oskar Klos**  
[GitHub](https://github.com/oskarklos2006)
