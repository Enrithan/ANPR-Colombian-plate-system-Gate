"""Quick diagnostic: test what anpr-demo-model detects on actual cali_traffic frames."""
import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture("cali_traffic.mp4")
model = YOLO("models/anpr-demo-model.pt")

# Sample 5 frames spread throughout the video
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sample_positions = [total // 8, total // 4, total // 2, (total * 3) // 4, total - 100]

print(f"Total frames: {total}")
print(f"Testing at positions: {sample_positions}\n")

for pos in sample_positions:
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame, verbose=False, conf=0.01)[0]  # Very low confidence to see everything
    boxes = results.boxes
    
    print(f"Frame {pos}:")
    if boxes is None or len(boxes) == 0:
        print("  → No detections at all (even at 1% confidence)")
    else:
        data = boxes.data.cpu().numpy()
        for row in data:
            conf = float(row[4])
            cls = int(row[5])
            print(f"  → Class {cls}, Confidence {conf:.3f}")
    print()

cap.release()
print("Diagnostic complete.")
