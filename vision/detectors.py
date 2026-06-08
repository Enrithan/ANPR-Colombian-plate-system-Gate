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
            # ⚡ Bolt: Use a set for O(1) membership lookups instead of a list
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
        if not results.boxes:
            return []

        # ⚡ Bolt: Extract all bounding box data in one vectorized tensor operation.
        # Iterating over results.boxes and accessing properties causes repeated synchronous
        # GPU-CPU data transfers, which is an enormous bottleneck.
        boxes_data = results.boxes.data.cpu().numpy()

        # ⚡ Bolt: Fast list comprehension filtering using the pre-computed set
        return [[row[0], row[1], row[2], row[3], float(row[4])]
                for row in boxes_data if int(row[5]) in self.vehicle_classes]

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
