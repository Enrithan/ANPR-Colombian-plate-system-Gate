import numpy as np
import time
from unittest.mock import MagicMock, patch
from vision.detectors import YOLOVehicleDetector

# Create dummy frame and mock YOLO results
dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

class MockTensor:
    def __init__(self, data):
        self.data = data
    def tolist(self):
        return self.data
    def cpu(self):
        return self
    def numpy(self):
        return np.array(self.data)

class MockBox:
    def __init__(self, cls, xyxy, conf):
        self.cls = [cls]
        self.xyxy = [MockTensor(xyxy)]
        self.conf = [conf]

class MockBoxes:
    def __init__(self, boxes_list):
        self._boxes = boxes_list
        # simulate results.boxes.data for vectorized operations
        self.data = MockTensor([[*b.xyxy[0].data, b.conf[0], b.cls[0]] for b in boxes_list])

    def __iter__(self):
        return iter(self._boxes)

    def __bool__(self):
        return bool(self._boxes)

class MockResults:
    def __init__(self, boxes):
        self.boxes = MockBoxes(boxes)

class MockModel:
    def __call__(self, frame, imgsz, verbose):
        # Return 100 boxes, half are vehicles (2, 3, 5, 7)
        boxes = []
        for i in range(100):
            cls = 2 if i % 2 == 0 else 0
            boxes.append(MockBox(cls, [0, 0, 100, 100], 0.9))
        return [MockResults(boxes)]

    def to(self, device):
        return self

@patch('vision.detectors.YOLO')
def test_detect(mock_yolo):
    mock_yolo.return_value.to.return_value = MockModel()
    detector = YOLOVehicleDetector("dummy.pt")

    # Warmup
    detector.detect(dummy_frame)

    start = time.time()
    for _ in range(10000):
        detector.detect(dummy_frame)
    end = time.time()

    print(f"List Time taken: {end - start:.4f}s")

    detector.vehicle_classes = set(detector.vehicle_classes)

    # Warmup
    detector.detect(dummy_frame)

    start = time.time()
    for _ in range(10000):
        detector.detect(dummy_frame)
    end = time.time()

    print(f"Set Time taken: {end - start:.4f}s")

if __name__ == "__main__":
    test_detect()

@patch('vision.detectors.YOLO')
def test_detect_vectorized_set(mock_yolo):
    mock_yolo.return_value.to.return_value = MockModel()
    detector = YOLOVehicleDetector("dummy.pt")
    detector.vehicle_classes = set(detector.vehicle_classes)

    # Redefine detect
    def new_detect(frame: np.ndarray) -> list:
        results = detector.model(frame, imgsz=640, verbose=False)[0]
        if not results.boxes:
            return []

        boxes_data = results.boxes.data.cpu().numpy()
        detections = []
        for row in boxes_data:
            if int(row[5]) in detector.vehicle_classes:
                detections.append([row[0], row[1], row[2], row[3], float(row[4])])
        return detections

    detector.detect = new_detect

    # Warmup
    detector.detect(dummy_frame)

    start = time.time()
    for _ in range(10000):
        detector.detect(dummy_frame)
    end = time.time()

    print(f"Vectorized + Set Time taken: {end - start:.4f}s")

if __name__ == "__main__":
    test_detect_vectorized_set()

@patch('vision.detectors.YOLO')
def test_detect_vectorized_mask(mock_yolo):
    mock_yolo.return_value.to.return_value = MockModel()
    detector = YOLOVehicleDetector("dummy.pt")
    # vehicle_classes as list for np.isin

    # Redefine detect
    def new_detect(frame: np.ndarray) -> list:
        results = detector.model(frame, imgsz=640, verbose=False)[0]
        if not results.boxes:
            return []

        boxes_data = results.boxes.data.cpu().numpy()
        mask = np.isin(boxes_data[:, 5], detector.vehicle_classes)
        filtered = boxes_data[mask]

        return filtered[:, :5].tolist()

    detector.detect = new_detect

    # Warmup
    detector.detect(dummy_frame)

    start = time.time()
    for _ in range(10000):
        detector.detect(dummy_frame)
    end = time.time()

    print(f"Vectorized Mask Time taken: {end - start:.4f}s")

if __name__ == "__main__":
    test_detect_vectorized_mask()
