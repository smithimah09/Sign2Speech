# Sign2Speech

## Summary
This project is a deep learning-based hand gesture recognition system using CNNs. It supports real-time classification of 44 static hand signs, including all letters of the alphabet, digits 0–9, and several custom signs. The system enables gesture-controlled text generation and calculator functionality using a webcam.

## Overview
* Data Collection: Captured 1,200 grayscale 50x50 pixel images per gesture using OpenCV, then augmented the dataset by flipping the images vertically to create 2,400 samples per gesture.

* Preprocessing: Required users to calibrate a skin-color histogram to segment the hand from the background. This step must be repeated if lighting changes.

* Model Training: Developed a CNN using both TensorFlow and Keras, with architecture inspired by the MNIST classifier. The Keras model was used for real-time inference.

* Real-Time Inference: The system uses a webcam to detect hand gestures in a defined region of the frame. It supports two modes:
    * Text Mode: Converts gestures to text and uses text-to-speech output.
    * Calculator Mode: Recognizes digits and operators to perform arithmetic and bitwise operations.

## Technologies Used
* Python 3.x
* OpenCV 3.4
* TensorFlow 1.5
* Keras
* h5py
* pyttsx3

## Impact
This system recognizes static American Sign Language gestures and improves accessibility by allowing individuals with speech impairments to communicate through hand gestures translated into real-time spoken text.
