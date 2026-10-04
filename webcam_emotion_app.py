import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import plot_results  # your custom file

# ---------------- LOAD MODEL ----------------
model = load_model("emotion_model.keras",compile=False)  # 👈 YOUR COLAB MODEL FILE

# ---------------- LABELS (MUST MATCH TRAINING) ----------------
emotion_labels = ["Pleasant", "Unpleasant", "Neutral"]

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---------------- WEBCAM ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

window_name = "Emotion Detection"
cv2.namedWindow(window_name)

# ---------------- TIME TRACKING ----------------
emotion_time = {emotion: 0 for emotion in emotion_labels}

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    delta_time = current_time - prev_time
    prev_time = current_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_emotion = None

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        face = gray[y:y+h, x:x+w]

        # preprocess SAME as training
        face = cv2.resize(face, (48, 48))
        face = face.astype("float32") / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        # prediction
        prediction = model.predict(face, verbose=0)
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]

        current_emotion = emotion

        cv2.putText(
            frame,
            emotion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    # accumulate time
    if current_emotion:
        emotion_time[current_emotion] += delta_time

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27 or key == ord('q'):
        break

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()

# ---------------- FINAL OUTPUT ----------------
name = input("Enter your name: ")

plot_results.plot_emotion_graph(name, emotion_time)