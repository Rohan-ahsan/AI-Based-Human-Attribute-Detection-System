import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque, Counter

# ---------------- 1. CONFIG & LOAD ----------------
IMG_SIZE = 96
# Load your model
model = load_model("gender_model_final.keras", compile=False)

# IMPORTANT: Check your Kaggle output for gender_threshold.npy
# If you can't find the file, use the number printed in your Kaggle logs.
try:
    BEST_THRESHOLD = np.load("gender_threshold.npy")[0]
    print(f"✅ Using loaded optimal threshold: {BEST_THRESHOLD:.2f}")
except:
    BEST_THRESHOLD = 0.50  # Default fallback
    print("⚠️ threshold file not found, using 0.50")

GENDER_LABELS = ["Male", "Female"]

# Stability buffer
prediction_buffer = deque(maxlen=10)

# ---------------- 2. DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ---------------- 3. WEBCAM ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 8, minSize=(80, 80))

    for (x, y, w, h) in faces:
        # Extract and convert to RGB
        face_roi = frame[y:y + h, x:x + w]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))

        # PREPROCESSING MATCH:
        # Your training code uses MobileNetV2's internal preprocess_input.
        # It expects raw float32 pixels (0-255).
        face_array = face_resized.astype("float32")
        input_data = np.expand_dims(face_array, axis=0)

        # Prediction
        preds = model.predict(input_data, verbose=0)
        raw_val = preds[0][0]

        # USE OPTIMAL THRESHOLD
        # 0 = Male, 1 = Female (according to your Kaggle distribution)
        current_idx = 1 if raw_val >= BEST_THRESHOLD else 0
        prediction_buffer.append(current_idx)

        # Stabilization
        stable_idx = Counter(prediction_buffer).most_common(1)[0][0]
        label = GENDER_LABELS[stable_idx]

        # UI
        color = (255, 0, 255) if stable_idx == 1 else (255, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} ({raw_val:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Optimized Gender Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()