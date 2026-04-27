import cv2
import numpy as np
from vision.plate_reader import EasyOCRPlateReader

reader = EasyOCRPlateReader()
img = np.zeros((100, 100, 3), dtype=np.uint8)
# create a contour of size > 500 area
cv2.rectangle(img, (10, 10), (90, 90), (255, 255, 255), -1)

try:
    res = reader.correct_perspective(img)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
