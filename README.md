# AI-Based Age, Gender & Mood Detection

Due to file size limits, large datasets and model files are hosted on Google Drive:
[Download Models & Datasets from Google Drive]
https://drive.google.com/drive/folders/1jYLabGMxjgBFyMvWnjy-zW88EnKBtIff?usp=sharing

A real-time computer vision application that uses **deep learning and OpenCV** to detect faces through a webcam and predict a person's **age group, gender, and mood**.

The project uses separate CNN-based models for each task and combines them into a single real-time application using **Python, TensorFlow, Keras, and OpenCV**.

---

## Features

* Real-time face detection through webcam
* Age group classification
* Gender classification
* Mood detection
* Multiple AI models integrated into one application
* Real-time prediction display
* CNN-based deep learning models
* Image preprocessing and real-time inference

---

## Predictions

### Age

The system classifies faces into four age groups:

* **0–18**
* **19–30**
* **31–50**
* **51+**

### Gender

* **Male**
* **Female**

### Mood

* **Pleasant**
* **Neutral**
* **Unpleasant**

---

## How It Works

```text
Webcam
   ↓
Face Detection
   ↓
Face Extraction & Preprocessing
   ↓
┌─────────────┬─────────────┬─────────────┐
│  Age Model  │Gender Model │ Mood Model  │
└─────────────┴─────────────┴─────────────┘
       ↓             ↓             ↓
   Age Group       Gender         Mood
       └─────────────┬─────────────┘
                     ↓
              Results Displayed
                on Webcam
```

The webcam continuously captures frames. Detected faces are extracted and preprocessed before being passed to the three trained models. The resulting predictions are displayed on the live video feed.

---

## Technologies

* **Python**
* **TensorFlow**
* **Keras**
* **OpenCV**
* **NumPy**
* **Convolutional Neural Networks (CNNs)**

---

## Models

| Task   | Model                             | Output                        |
| ------ | --------------------------------- | ----------------------------- |
| Age    | `age_model_final_optimized.keras` | 0–18, 19–30, 31–50, 51+       |
| Gender | `gender_model_final.keras`        | Male, Female                  |
| Mood   | `emotion_model_final.keras`       | Pleasant, Neutral, Unpleasant |

The age model achieved approximately **69% accuracy** during evaluation.

---

## Datasets

Different datasets were used for the individual models.

### UTKFace

Used for developing the age classification model. The original age labels were converted into the four age groups used by the application.

### Facial Expression Dataset

Used for training the mood detection model and adapted into the three classes:

```text
Pleasant
Neutral
Unpleasant
```

---

## Project Structure

```text
AI-Age-Gender-Mood-Detection/
│
├── models/
│   ├── age_model_final_optimized.keras
│   ├── gender_model_final.keras
│   └── emotion_model_final.keras
│
├── dataset/
├── notebooks/
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/AI-Age-Gender-Mood-Detection.git
cd AI-Age-Gender-Mood-Detection
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Make sure the trained models are placed in the correct directory, then run:

```bash
python main.py
```

The webcam will open and the system will begin detecting faces and displaying the predicted age group, gender, and mood.

Example:

```text
Age: 19–30
Gender: Male
Mood: Pleasant
```

---

## Limitations

* Age is predicted as a range rather than an exact age.
* Predictions may vary depending on lighting, camera quality, and face angle.
* Mood detection is based on visible facial expressions and may not represent a person's actual emotional state.
* Model accuracy depends on the quality and diversity of the training datasets.
* The system is intended for educational and experimental purposes.

---

## Future Improvements

* Improve model accuracy with larger and more diverse datasets
* Support multiple faces simultaneously
* Add prediction confidence scores
* Optimize models for faster real-time inference
* Add a graphical user interface
* Deploy the system as a web application
