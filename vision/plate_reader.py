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

    def correct_perspective(self, img):
        """Finds the 4 corners of the plate and warps it to a clean rectangle."""
        # 1. Image preparation
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 50, 200)

        # 2. Find the largest rectangular-ish contour
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        screen_cnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # A plate is a quadrilateral (4 corners)
            if len(approx) == 4:
                screen_cnt = approx
                break
        
        if screen_cnt is None:
            return img # Return original if no 4-point polygon found

        # 3. Order the points: [top-left, top-right, bottom-right, bottom-left]
        pts = screen_cnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # 4. Perspective warp to a standard size (e.g. 320x160)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Standard Colombian plate aspect ratio
        dst = np.array([
            [0, 0],
            [320 - 1, 0],
            [320 - 1, 160 - 1],
            [0, 160 - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (320, 160))
        
        return warped

    def read_text(self, cropped_plate: np.ndarray) -> tuple[str, float]:
        # NEW: Phase 5 - Perspective Correction (Unwarping)
        processed_plate = self.correct_perspective(cropped_plate)
        
        # Preprocessing for OCR
        gray = cv2.cvtColor(processed_plate, cv2.COLOR_BGR2GRAY)
        # Apply a light adaptive threshold to improve OCR contrast
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # EasyOCR prediction
        detections = self.reader.readtext(thresh)

        for detection in detections:
            bbox, text, score = detection
            text = text.upper().replace(' ', '').replace('-', '').replace('.', '')
            
            # Use utility functions from remote branch refactor
            if license_complies_format(text):
                return format_license(text), score

        return "", 0.0
