import boto3
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import threading
import uvicorn
from pyngrok import ngrok
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BUCKET_NAME = os.getenv('cvBucket')
ACCESS_KEY = os.getenv('accessKey')
SECRET_KEY = os.getenv('secretKey')
NGROK_AUTHTOKEN = os.getenv('ngrockToken')

local_model_path = 'face_detector.h5'

# Object to access S3 resources.
s3 = boto3.resource('s3', aws_access_key_id = ACCESS_KEY, aws_secret_access_key= SECRET_KEY)

# face_detector.h5 is the file we should download
import botocore
# save the filename in KEY variable
KEY = 'face_detector.h5' # replace with your object key

s3.Bucket(BUCKET_NAME).download_file(KEY, 'face_detector.h5')

# Load the model
model = load_model(local_model_path)

# Function to preprocess each frame
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))  # Resize to 224x224
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Global variable to control the video capture
capture_video = False

def video_capture():
    global capture_video
    cap = cv2.VideoCapture(0)  # Use 0 for webcam

    while capture_video:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Make prediction
        prediction = model.predict(processed_frame)

        # Convert the prediction to a class label (binary classification assumed)
        label = 'Alarming' if prediction > 0.5 else 'Accepted'

        # Display the prediction on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Video', frame)

        # Exit if the user presses 'q'
        if cv2.waitKey(2) & 0xFF == ord('q'):
            capture_video = False
            break

    cap.release()
    cv2.destroyAllWindows()
    
app=FastAPI()

@app.get("/start_prediction")
def start_prediction():
    global capture_video
    capture_video = True
    thread = threading.Thread(target=video_capture)
    thread.start()
    return {"message": "Video capture started"}

@app.get("/stop_prediction")
def stop_prediction():
    global capture_video
    capture_video = False
    return {"message": "Video capture stopped"}

if __name__ == "__main__":
    # Start ngrok tunnel
    public_url = ngrok.connect(8000)
    print(f"ngrok tunnel created at {public_url}")

    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)