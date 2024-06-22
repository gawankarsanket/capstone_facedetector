# Computer Vision-Based Face Mask, Sunglasses, Caps, and Hoodies Detection

## Project Overview
This project aims to develop a real-time system to detect if a human face is covered with masks, sunglasses, caps, or hoodies. The system is designed for applications in security monitoring, ATM security, and access control.

## Technologies Used
- **TensorFlow/Keras**
- **OpenCV**
- **FastAPI**
- **AWS S3**
- **Ngrok**

## Problem Statement
Unauthorized individuals covering their faces in restricted areas pose security concerns. This project implements a computer vision system to identify and alert when faces are covered by masks, sunglasses, caps, or hoodies.

## Dataset Collection
### Image Sources
- Internet
- Custom datasets

### Categories
- **Acceptable**: Faces without any coverings.
- **Alarming**: Faces with masks, sunglasses, caps, or hoodies.

### Preprocessing
- Image resizing to 224x224 pixels.
- Normalization.

## Data Preparation
### Data Download
Using Python scripts to download and categorize images.

### Preprocessing
- Resizing images using OpenCV.
- Normalizing pixel values.

### Data Storage
Organized in folders ('Acceptable' and 'Alarming') and uploaded to AWS S3 for access.

## Model Development
### Transfer Learning
- **Base Model**: ResNet50 pre-trained on ImageNet.
- **Custom Layers**: Added custom dense and output layers for binary classification.

### Model Training
- Split dataset into training and testing sets.
- Used augmentation techniques for better generalization.

## Model Training and Evaluation
### Training
- **Metrics**: Accuracy, loss.
- **Tools**: TensorFlow/Keras.

### Evaluation
- Validation on test set.
- Metrics: Test accuracy, confusion matrix, and classification report.

## Model Deployment
### Framework
FastAPI for serving the model.

### Endpoints
- `/start_prediction`: Start video capture and prediction.
- `/stop_prediction`: Stop video capture.

### Ngrok
Used to expose the local server to the internet.

## Real-Time Prediction
### Video Capture
Using OpenCV to access webcam and display predictions on video frames.

### Prediction Logic
- Preprocessing each frame.
- Making predictions using the trained model.
- Displaying results on the video stream.

## Challenges and Solutions
- **Data Collection**: Sourcing a diverse dataset.
- **Model Accuracy**: Improving accuracy through data augmentation and fine-tuning.
- **Real-Time Processing**: Optimizing prediction speed for real-time use.

## Future Work
### Improvements
- Increase dataset size and diversity.
- Enhance model accuracy and speed.

### Additional Features
- Integrate with security systems.
- Develop a mobile application for on-the-go use.

## Conclusion
This project successfully implements a computer vision system for detecting face coverings in real-time, enhancing security by providing timely alerts.

## Contact Information
For any questions, feel free to reach out to me.
