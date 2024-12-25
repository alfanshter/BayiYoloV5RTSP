import cv2
import torch
import subprocess
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", force_reload=True)

# Define your RTSP server details
rtsp_url = "rtsp://localhost:8554/live"
ffmpeg_command = [
    "ffmpeg",
    "-loglevel", "verbose",
    "-y",
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", "320x240",  # Resolution changed to 320x240
    "-r", "10",
    "-i", "-",
    "-c:v", "libx264",
    "-f", "rtsp",
    rtsp_url
]

# Function to start FFmpeg process
def start_ffmpeg():
    return subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

# Start FFmpeg process
process = start_ffmpeg()

# Open webcam or video
video_path = "bayi.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Resize the frame to a lower resolution
        frame = cv2.resize(frame, (320, 240))  # Reduce resolution to 320x240

        # Perform object detection on the frame
        results = model(frame)

        # Filter detections with confidence >= 0.5
        detections = results.pandas().xyxy[0]
        filtered_detections = detections[detections['confidence'] >= 0.5]

        # Annotate the frame with filtered detections
        annotated_frame = frame.copy()  # Create a copy of the frame for annotations
        for index, row in filtered_detections.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{row['name']} {row['confidence']:.2f}"
            # Draw bounding box and label on the frame
            cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check if FFmpeg process is alive
        if process.poll() is not None:
            print("FFmpeg process terminated unexpectedly, restarting...")
            process = start_ffmpeg()

        # Stream to RTSP server, with error handling for broken pipe
        try:
            process.stdin.write(annotated_frame.tobytes())
        except BrokenPipeError:
            print("Broken pipe detected, restarting FFmpeg process...")
            process = start_ffmpeg()
            process.stdin.write(annotated_frame.tobytes())

        # Exit loop when 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Streaming stopped")

finally:
    # Release resources
    cap.release()
    if process.stdin:
        process.stdin.close()
    if process.poll() is None:
        process.wait()
    cv2.destroyAllWindows()
