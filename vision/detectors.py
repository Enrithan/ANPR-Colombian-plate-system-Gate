from ultralytics import YOLO
import numpy as np
from .interfaces import IVehicleDetector, IVehicleTracker, IPlateDetector
import sys
import os

# Ensure the sort module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sort.sort import Sort

class YOLOVehicleDetector(IVehicleDetector):
    def __init__(self, model_path: str, vehicle_classes: list = None):
        if vehicle_classes is None:
            # Default COCO classes for vehicles (car, motorcycle, bus, truck)
            self.vehicle_classes = [2, 3, 5, 7]
        else:
            self.vehicle_classes = vehicle_classes
            
        self.model = YOLO(model_path).to('cpu')

    def detect(self, frame: np.ndarray) -> list:
        results = self.model(frame, verbose=False)[0]
        detections = [
            [x1, y1, x2, y2, conf]
            for x1, y1, x2, y2, conf, class_id in results.boxes.data.tolist()
            if int(class_id) in self.vehicle_classes
        ]
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
        self.model = YOLO(model_path).to('cpu')

    def detect(self, frame: np.ndarray) -> list:
        # Returns [x1, y1, x2, y2, conf, cls]
        return self.model(frame, verbose=False)[0].boxes.data.tolist()
