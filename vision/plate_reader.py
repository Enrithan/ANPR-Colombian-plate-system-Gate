import cv2
import string
import easyocr
import numpy as np
from .interfaces import IPlateReader
from util import license_complies_format, format_license

# Initialize the OCR reader once
_reader = easyocr.Reader(['en'], gpu=False)

class EasyOCRPlateReader(IPlateReader):
    def __init__(self):
        self.reader = _reader

    def license_complies_format(self, text):
        """
        Check if the license plate text complies with the required format.
        Colombian format Cars/Commercial: AAA123 (3 letters, 3 numbers)
        Colombian format Motorcycles: AAA12A (3 letters, 2 numbers, 1 letter)
        """
        return license_complies_format(text)

    def read_text(self, cropped_plate: np.ndarray) -> tuple[str, float]:
        # Preprocessing
        gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)

        # EasyOCR prediction
        detections = self.reader.readtext(thresh)

        for detection in detections:
            bbox, text, score = detection
            text = text.upper().replace(' ', '')
            
            # Simple format check
            if self.license_complies_format(text):
                return format_license(text), score

        return "", 0.0
