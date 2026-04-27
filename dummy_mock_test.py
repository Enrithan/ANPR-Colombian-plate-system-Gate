import sys
from unittest.mock import MagicMock
sys.modules['cv2'] = MagicMock()
sys.modules['easyocr'] = MagicMock()

import numpy as np
import torch
sys.modules['cv2'].cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
sys.modules['cv2'].findContours.return_value = ([], None)

from vision.plate_reader import EasyOCRPlateReader

reader = EasyOCRPlateReader()
img = np.zeros((100, 100, 3), dtype=np.uint8)

res = reader.correct_perspective(img)
print("Success")
