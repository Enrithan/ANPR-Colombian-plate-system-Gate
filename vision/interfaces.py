from abc import ABC, abstractmethod
import numpy as np

class IVideoSource(ABC):
    @abstractmethod
    def read_frame(self):
        """Returns (success_bool, frame_array)"""
        pass
        
    @abstractmethod
    def release(self):
        """Releases the camera resource"""
        pass

class IVehicleDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list:
        """Returns list of bounding boxes [x1, y1, x2, y2, conf]"""
        pass

class IVehicleTracker(ABC):
    @abstractmethod
    def update(self, detections: list) -> np.ndarray:
        """Returns tracked objects [x1, y1, x2, y2, track_id]"""
        pass

class IPlateDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list:
        """Returns list of plate bounding boxes [x1, y1, x2, y2, conf, cls]"""
        pass

class IPlateReader(ABC):
    @abstractmethod
    def read_text(self, cropped_plate: np.ndarray) -> tuple[str, float]:
        """Returns (plate_text_string, confidence_score)"""
        pass
