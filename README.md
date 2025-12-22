## Project Overview
This project focuses on American Sign Language (ASL) alphabet recognition using deep learning and transfer learning techniques.
The system classifies static hand gesture images into ASL letters and special symbols, and also supports real-time gesture recognition using a webcam.

Multiple Convolutional Neural Network (CNN) architectures were implemented, trained, and compared to evaluate their performance in terms of accuracy, robustness, and generalization. The project also integrates model explainability (Grad-CAM) and a graphical user interface (GUI) to demonstrate the results interactively.

---

## Project Goal
- Classify ASL alphabet hand gestures from images and live video
- Compare multiple CNN architectures using transfer learning
- Improve robustness to lighting, background, and pose variations
- Implement real-time gesture recognition using a webcam
- Provide visual explanations of model predictions using Grad-CAM
- Build a simple and user-friendly GUI for interaction

---

## Dataset
- Name: ASL Alphabet Dataset  
- Source: Kaggle  
- Link: https://www.kaggle.com/datasets/grassknoted/asl-alphabet  
- Description:
  - RGB images representing American Sign Language letters (A–Z) and special classes (space, delete, nothing)
  - Large number of samples per class
  - Variations in hand orientation, scale, and background

---

## Data Preprocessing & Augmentation
To enhance model generalization and reduce overfitting, the following preprocessing and augmentation techniques were applied:- Image resizing
- Image resizing to a fixed input size
- Pixel normalization
- Random rotation
- Horizontal flipping
- Random zoom and shift
- Brightness and contrast adjustments
These techniques help the models handle real-world variations in lighting and hand positioning

---

##  Implemented Models
The following deep learning models were implemented and evaluated:
- InceptionV3  (97%)
- ResNet50      (96%)
- EfficientNetB0  (98.70%)
All models were trained on the same dataset split and evaluated using identical metrics to ensure fair comparison

---

##  Results & Discussion
- InceptionV3 achieved the best overall performance, with the highest accuracy, precision, and recall, and the cleanest confusion matrix with minimal misclassifications.
- ResNet50 showed strong and stable performance, with very good generalization across all ASL classes.
- EfficientNet achieved competitive results while maintaining high efficiency, making it suitable for lightweight or resource-constrained deployments.
Overall, the results demonstrate the effectiveness of transfer learning and data augmentation for ASL gesture recognition.
Based on quantitative metrics and confusion matrix analysis, InceptionV3 was selected as the final deployment model..


Resnet

![1fa30564-d35c-4d1c-8d11-7a36a03d6140](https://github.com/user-attachments/assets/7d349bda-73ca-4e19-acbe-59e5fc8bdcfc)




inception 

![63309212-11ee-45b4-b3f8-05e8de7e563f](https://github.com/user-attachments/assets/4bac77c2-efd3-4e52-a8af-43d7c269bf69)



Efficient

![06fa2eea-5a23-44db-b42a-15ab3ba086ff](https://github.com/user-attachments/assets/360e0486-d212-46b6-8512-78833aa4cdb3)



Models Comparison

![1223705f-acba-4658-9fd0-2e981a37b0c1](https://github.com/user-attachments/assets/c9106e4b-0d2e-4c35-9fb8-2a8ca51384ee)



![a69d478e-8f51-48c5-9b98-e613fd0d3b75](https://github.com/user-attachments/assets/eef983b4-150e-4e69-bef2-0334254ae8be)



---

## Real-Time Gesture Recognition
A real-time gesture recognition system was implemented using a webcam:
- Captures live video frames
- Applies preprocessing in real time
- Performs model inference on each frame
- Displays predicted class and confidence on screen
Notebook:
- `web-cam.ipynb`

---

## Model Explainability
To improve transparency and trust in the system, Grad-CAM was applied to visualize the regions of the image that contribute most to the model’s predictions.
These heatmaps confirm that the models focus primarily on the hand and finger regions, which aligns with human intuition.

---

##  GUI
A graphical user interface (GUI) was developed to allow users to:- Load images
- Load and classify ASL images
- Run real-time webcam gesture recognition
- Interact with the system easily without coding knowledge

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

















