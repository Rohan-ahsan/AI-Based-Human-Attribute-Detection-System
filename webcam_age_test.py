import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque, Counter

# ---------------- 1. CONFIG & LABELS ----------------
AGE_LABELS = [
    "Child-Teen (0-18)",
    "Young Adult (19-30)",
    "Adult (31-50)",
    "Senior (51+)"
]
IMG_SIZE = 128

# STABILITY SETTINGS
BUFFER_SIZE = 13  # Number of frames to average (higher = smoother, but slower to react)
prediction_buffer = deque(maxlen=BUFFER_SIZE)

# ---------------- 2. LOAD MODEL ----------------
print("🔄 Loading Stable EfficientNetB2 model...")
model = load_model("age_model_final_optimized.keras", compile=False)
print("✅ Model loaded.")

# ---------------- 3. INITIALIZE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ---------------- 4. WEBCAM SETUP ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
window_name = "Stable Age Detection"

# Default label before detection starts
stable_label = "Scanning..."
stable_conf = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror view
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Tuned detector: higher minNeighbors = less "jitter"
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        # Preprocessing
        face_roi = frame[y:y + h, x:x + w]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
        face_array = face_resized.astype("float32")
        input_data = np.expand_dims(face_array, axis=0)

        # Prediction
        preds = model.predict(input_data, verbose=0)
        idx = np.argmax(preds)
        conf = preds[0][idx] * 100

        # --- STABILIZATION LOGIC ---
        prediction_buffer.append(idx)

        # Get the most frequent index in the last 15 frames
        most_common_idx = Counter(prediction_buffer).most_common(1)[0][0]
        stable_label = AGE_LABELS[most_common_idx]
        stable_conf = conf  # We still show real-time confidence

        # Dynamic Color Logic
        color = (0, 255, 0) if conf > 70 else (0, 255, 255)

        # Draw UI
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, y - 35), (x + w, y), color, -1)
        cv2.putText(
            frame,
            f"{stable_label}",
            (x + 5, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )
        cv2.putText(
            frame,
            f"Inst. Conf: {conf:.1f}%",
            (x, y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()