## Project Overview
This project focuses on classifying hand gestures using deep learning models.
The system is trained on hand gesture images and is capable of performing both
offline classification and real-time gesture recognition using a webcam.

Multiple CNN architectures were implemented and compared to evaluate accuracy,
robustness, and performance under different conditions.

---

## Project Goal
- Classify hand gestures from images and video streams
- Compare different deep learning architectures
- Handle lighting and pose variations using data augmentation
- Implement real-time gesture recognition
- Provide visual explanations using Grad-CAM
- Build a simple GUI for user interaction

---

## Dataset
- Name: ASL Alphabet Dataset  
- Source: Kaggle  
- Link: https://www.kaggle.com/datasets/grassknoted/asl-alphabet  
- Description:
  - RGB images representing American Sign Language letters
  - Multiple samples per class
  - Diverse backgrounds and hand orientations

---

## Data Preprocessing & Augmentation
To improve model generalization, the following techniques were applied:
- Image resizing
- Normalization
- Random rotation
- Horizontal flipping
- Zoom and shift
- Brightness and contrast adjustments

---

##  Implemented Models
The following deep learning models were implemented and evaluated:
- InceptionV3  (97%)
- ResNet50      (96%)
- EfficientNetB0  (92%)

---

##  Results & Discussion
- EfficientNetB0: achieved the highest accuracy and was selected as the final model.
- InceptionV3: showed strong performance with deep feature extraction.
- ResNet50: provided stable training with slightly lower accuracy.
-The results demonstrate that EfficientNetB0 offers the best trade-off between
accuracy and computational efficiency.


Resnet 

![WhatsApp Image 2025-12-22 at 10 14 49 AM](https://github.com/user-attachments/assets/1fa30564-d35c-4d1c-8d11-7a36a03d6140)

inception 

![WhatsApp Image 2025-12-22 at 10 14 50 AM inception ](https://github.com/user-attachments/assets/63309212-11ee-45b4-b3f8-05e8de7e563f)

Efficient

![WhatsApp Image 2025-12-22 at 10 14 51 AM  EFFETIONT](https://github.com/user-attachments/assets/06fa2eea-5a23-44db-b42a-15ab3ba086ff)

Models Comparison

![WhatsApp Image 2025-12-22 at 10 14 50 AM comparison](https://github.com/user-attachments/assets/1223705f-acba-4658-9fd0-2e981a37b0c1)

![WhatsApp Image 2025-12-22 at 10 14 52 AM comparison macros](https://github.com/user-attachments/assets/a69d478e-8f51-48c5-9b98-e613fd0d3b75)


---

## Real-Time Gesture Recognition
A real-time gesture recognition system was implemented using a webcam:
- Captures live video frames
- Applies preprocessing
- Predicts gestures in real time
- Displays predictions on screen

Notebook:
- `web-cam.ipynb`

---

## Model Explainability
Grad-CAM was used to visualize the regions of the image that contribute most
to the model’s predictions, improving transparency and interpretability.

---

##  GUI
A graphical user interface allows users to:
- Load images
- View predictions
- Run real-time webcam gesture recognition
- Interact easily with the system

---

##  Setup Instructions

### Requirements
- Python 3.8+
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

| Name            | Role                                 |
| --------------- | ------------------------------------ |
| Omar            | InceptionV3 model implementation     |
| Mostafa Fouad   | ResNet50 model implementation        |
| Mahmoud Ehab    | EfficientNetB0 model implementation  |
| Mahmoud Mohamed | Real-time webcam gesture recognition |
| Mostafa Mahmoud | Grad-CAM and Models evaluation       |
| Nour            | GUI development                      |
















