import numpy as np
import time
from unittest.mock import MagicMock, patch
from vision.detectors import YOLOVehicleDetector
import torch

dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

class MockModel:
    def __init__(self, data):
        self.data = torch.tensor(data)
        class _Boxes:
            def __init__(self, d):
                self.data = d
                self._is_empty = len(d) == 0
            def __bool__(self):
                return not self._is_empty
            def __len__(self):
                return len(self.data)
            def __iter__(self):
                # Mock the exact object structure for `box in results.boxes` loop
                for row in self.data:
                    class _Box:
                        def __init__(self, r):
                            self.xyxy = [r[:4]]
                            self.conf = [r[4]]
                            self.cls = [r[5]]
                    yield _Box(row)

        class _Result:
            def __init__(self, b):
                self.boxes = b

        self.result = [_Result(_Boxes(self.data))]

    def __call__(self, frame, imgsz, verbose):
        return self.result

    def to(self, device):
        return self

@patch('vision.detectors.YOLO')
def test_detect(mock_yolo):
    # 100 detections per frame
    data = []
    for i in range(100):
        cls = 2 if i % 2 == 0 else 0
        data.append([0, 0, 100, 100, 0.9, cls])

    mock_yolo.return_value.to.return_value = MockModel(data)

    detector = YOLOVehicleDetector("dummy.pt")

    # Warmup
    detector.detect(dummy_frame)

    start = time.time()
    for _ in range(1000):
        detector.detect(dummy_frame)
    end = time.time()

    print(f"Original Time taken: {end - start:.4f}s")

    # Vectorized List comprehension
    def new_detect_list(frame: np.ndarray) -> list:
        results = detector.model(frame, imgsz=640, verbose=False)[0]
        if not results.boxes:
            return []

        boxes_data = results.boxes.data.cpu().numpy()
        return [[row[0], row[1], row[2], row[3], float(row[4])]
                for row in boxes_data if int(row[5]) in detector.vehicle_classes]

    detector.detect = new_detect_list
    detector.vehicle_classes = set(detector.vehicle_classes)

    start = time.time()
    for _ in range(1000):
        detector.detect(dummy_frame)
    end = time.time()

    print(f"Vectorized + LC Time taken: {end - start:.4f}s")

if __name__ == "__main__":
    test_detect()
