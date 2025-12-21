Project Title : Real-Time Hand Gesture Classification Using Deep Learning 
Description:
This project aims to classify hand gestures from images and live video streams using deep learning models
The system recognizes hand gestures such as letters and symbolic gestures and can be extended to commands like stop, help, or peace.
We compare multiple CNN architectures to evaluate performance and robustness under variations in lighting, hand pose, and background.
A real-time webcam-based gesture recognition module and a graphical user interface (GUI) are also implemented.
Dataset :
Name: ASL Alphabet Dataset
Link: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
Description: 
Images of hand gestures representing American Sign Language (A–Z)
RGB images
Multiple samples per class
Preprocessing: 
Image resizing
Normalization
Data augmentation to improve generalization
Data Augmentation:
To handle lighting and pose variability, we apply strong augmentations:
Random rotation 
Horizontal flipping
Zoom and shift 
