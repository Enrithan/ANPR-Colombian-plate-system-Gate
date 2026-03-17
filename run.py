from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture("anpr-demo-video.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, 
                                       cv2.CAP_PROP_FRAME_HEIGHT, 
                                       cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("anpr-output.avi", 
                               cv2.VideoWriter_fourcc(*"mp4v"), 
                               fps, (w, h))

# Load the Ultralytics YOLO license plate detection model
model = YOLO("models/anpr-demo-model.pt")

