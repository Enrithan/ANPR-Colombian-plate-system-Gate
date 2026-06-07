from ultralytics import YOLO
import numpy as np
from .interfaces import IVehicleDetector, IVehicleTracker, IPlateDetector
import sys
import os

# Ensure the sort module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sort.sort import Sort

import torch

class YOLOVehicleDetector(IVehicleDetector):
    def __init__(self, model_path: str, vehicle_classes: list = None):
        if vehicle_classes is None:
            # Set lookup is O(1) instead of O(n) for list
            self.vehicle_classes = {2, 3, 5, 7}
        else:
            self.vehicle_classes = set(vehicle_classes)
            
        # Auto-detect device (Use CUDA if available)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # YOLOv10 is compatible with the YOLO class but faster/MIT-licensed
        self.model = YOLO(model_path).to(self.device)
        print(f"[AI] Vehicle Detector (Commercial-Ready) loaded on {self.device}")

    def detect(self, frame: np.ndarray) -> list:
        # YOLOv10 performs NMS internally (End-to-End)
        results = self.model(frame, imgsz=640, verbose=False)[0]
        detections = []
        if results.boxes:
            # Extract all boxes using vectorized tensor operation to prevent
            # multiple synchronous GPU-CPU data transfers in loop
            if hasattr(results.boxes, 'data') and results.boxes.data is not None:
                boxes_data = results.boxes.data
                if hasattr(boxes_data, 'cpu'):
                    boxes_data = boxes_data.cpu().numpy()

                for row in boxes_data:
                    x1, y1, x2, y2, conf, cls = row
                    if int(cls) in self.vehicle_classes:
                        detections.append([float(x1), float(y1), float(x2), float(y2), float(conf)])
            else:
                # Fallback in case of unexpected structure
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if cls in self.vehicle_classes:
                        detections.append([x1, y1, x2, y2, conf])
        return detections

class SORTVehicleTracker(IVehicleTracker):
    def __init__(self):
        self.tracker = Sort()

    def update(self, detections: list) -> np.ndarray:
        if detections:
            dets = np.array(detections, dtype=float)
            return self.tracker.update(dets)
        else:
            return np.zeros((0, 5), dtype=float)

class YOLOPlateDetector(IPlateDetector):
    def __init__(self, model_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(self.device)
        print(f"[AI] Plate Detector (Commercial-Ready) loaded on {self.device}")

    def detect(self, frame: np.ndarray) -> list:
        # Increase imgsz to 1024 for high-speed/small plate accuracy
        # Note: YOLOv10 is optimized for this type of high-res inference
        results = self.model(frame, imgsz=1024, verbose=False)[0]
        return results.boxes.data.tolist() if results.boxes else []
