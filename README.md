# Real-Time Object Detection and Emotion Classification

This project is a Python-based real-time detection tool that uses a webcam to identify objects and facial emotions. It combines two main components:

---

## 1. General Object Detection (YOLOv8)

The first part of the project uses the YOLOv8 nano model to detect general objects like cups, bottles, people, etc. The webcam feed is processed in real time and bounding boxes are drawn using YOLO's predictions.

### Features:
- Fast and lightweight YOLOv8n model
- Real-time detection from webcam using OpenCV
- Plug-and-play functionality (no training required)

### Dependencies:
- `ultralytics`
- `opencv-python`

---

## 2. Facial Emotion Classification (FER2013)

The second part of the project extends functionality to analyze human facial emotions from webcam input. A CNN model is trained on the FER2013 dataset to recognize the following emotional states:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

The model is trained using grayscale 48x48 face images and integrated into the webcam pipeline after detecting faces.

### Workflow:
1. Train a CNN model on FER2013 (see `notebooks/01_train_emotion_model.ipynb`)
2. Save the model to `models/emotion_cnn.pt`
3. Load it in a real-time webcam script to classify detected facial expressions

---

To install requirements:
```bash
pip install -r requirements.txt

