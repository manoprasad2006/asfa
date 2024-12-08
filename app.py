from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import onnxruntime as ort
import numpy as np
import cv2
import io
import os

# Initialize FastAPI app
app = FastAPI()

# Load your ONNX model
onnx_model_path = 'model.onnx'  # Replace with the path to your model
session = ort.InferenceSession(onnx_model_path)

# Define the input name (assuming a single input)
input_name = session.get_inputs()[0].name

# Initialize OpenCV for video capture
video_capture = cv2.VideoCapture(1,cv2.CAP_DSHOW)

def generate_frames():
    global video_capture
    while True:
        # Capture frame from the webcam
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Pre-process the image for the model
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (160, 160))  # Adjust to your model's input size
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Perform inference with the ONNX model
        result = session.run(None, {input_name: image})
        prediction = result[0][0]
        
        # Label based on prediction
        label = "Spoof" if prediction > 0.5 else "Real"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        color = (0, 0, 255) if label == "Spoof" else (0, 255, 0)

        # Draw label and confidence on the frame
        #label_text = f"{label} (Confidence: {confidence:.2f})"
        label_text = f"{label} (Confidence: {confidence.item():.2f})"
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Convert frame to byte format to stream it
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = io.BytesIO(buffer)
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes.read() + b'\r\n')

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Face Spoof Detection</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 20px;
                background-color: #f0f0f0;
            }}
            #video-container {{
                position: relative;
                max-width: 640px;
                width: 100%;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            #video-feed {{
                max-width: 100%;
                width: 100%;
            }}
            #prediction {{
                margin-top: 10px;
                font-size: 18px;
                font-weight: bold;
                text-align: center;
                padding: 10px;
                background-color: #e0e0e0;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <h1>Live Face Spoof Detection</h1>
        <div id="video-container">
            <img id="video-feed" src="/video_feed" alt="Video Feed">
            <div id="prediction">Prediction: Waiting...</div>
        </div>

        <script>
            // Get the video element
            const videoElement = document.getElementById('video-feed');
            const predictionEl = document.getElementById('prediction');

            // Set the video feed source to the backend URL
            videoElement.src = '/video_feed';
        </script>
    </body>
    </html>
    """

@app.get("/video_feed")
async def video_feed():
    # Stream the video feed in MJPEG format
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
