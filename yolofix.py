import cv2
import torch
import warnings
import os
from flask import Flask, Response
import threading
import time
warnings.filterwarnings('ignore')

# Flask app setup
app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", force_reload=True)

# Video source
video_path = "bayi.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(0)  # Use 0 for webcam
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Frame buffer for threading
frame_buffer = None
frame_lock = threading.Lock()

# Background thread to process video frames
def process_frames():
    global frame_buffer

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        time.sleep(max(0, 1/15 - (time.time() - start_time)))
        # Perform object detection on the frame
        results = model(frame)

        # Filter detections with confidence >= 0.5
        detections = results.pandas().xyxy[0]
        filtered_detections = detections[detections['confidence'] >= 0.1]

        # Annotate the frame with filtered detections
        annotated_frame = frame.copy()  # Create a copy of the frame for annotations
        for index, row in filtered_detections.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{row['name']} {row['confidence']:.2f}"
            # Draw bounding box and label on the frame
            cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Resize frame for faster browser rendering
        #resized_frame = cv2.resize(annotated_frame, (640, 480))
        resized_frame = cv2.resize(annotated_frame, (320, 240))
        
        # Update the frame buffer
        with frame_lock:
            frame_buffer = resized_frame

    cap.release()

# Flask route to stream video
@app.route("/video_feed")
def video_feed():
    def generate():
        global frame_buffer
        while True:
            with frame_lock:
                if frame_buffer is not None:
                    #_, encoded_image = cv2.imencode('.jpg', frame_buffer)
                    _, encoded_image = cv2.imencode('.jpg', frame_buffer, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

                    frame = encoded_image.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Flask route for homepage
@app.route("/")
def index():
    return "<h1>Video Stream</h1><img src='/video_feed' width='640' height='480'>"

if __name__ == "__main__":
    # Start the frame processing thread
    threading.Thread(target=process_frames, daemon=True).start()

    # Run Flask app
    app.run(host="0.0.0.0", port=8000, threaded=True)
