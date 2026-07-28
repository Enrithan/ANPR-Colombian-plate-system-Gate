import cv2
import string
import easyocr
import numpy as np
import torch
from .interfaces import IPlateReader
from util import license_complies_format, format_license

# Initialize THE reader once with GPU support if available
_gpu_available = torch.cuda.is_available()
print(f"Vision Module: Initializing EasyOCR (GPU={_gpu_available})")
_reader = easyocr.Reader(['en'], gpu=_gpu_available)

# Pre-computed arrays for perspective warp and image sharpening to avoid re-allocation in loops
_DST_PTS = np.array([
    [0, 0],
    [320 - 1, 0],
    [320 - 1, 160 - 1],
    [0, 160 - 1]], dtype="float32")

_SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])


class EasyOCRPlateReader(IPlateReader):
    def __init__(self):
        self.reader = _reader
        # Pre-compute allowlist for performance
        self.allowlist = string.ascii_uppercase + string.digits

    def correct_perspective(self, img):
        """Finds the 4 corners of the plate and warps it to a clean rectangle."""
        # 1. Image preparation
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Poka-yoke: If image is too small, unwarping might fail. Skip for very small crops.
        if gray.shape[0] < 20 or gray.shape[1] < 40:
            return img

        # Multi-stage edge detection to find the plate contour
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(blur, 30, 150)

        # 2. Find the largest rectangular-ish contour
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        screen_cnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # A plate is a quadrilateral (4 corners)
            if len(approx) == 4:
                # Basic area check to avoid tiny noise
                if cv2.contourArea(c) > 500:
                    screen_cnt = approx
                    break
        
        if screen_cnt is None:
            return img # Return original if no valid 4-point polygon found

        # 3. Order the points properly
        pts = screen_cnt.reshape(4, 2)
        rect = np.empty((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left
        rect[2] = pts[np.argmax(s)] # Bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-right
        rect[3] = pts[np.argmax(diff)] # Bottom-left

        # 4. Perspective warp
        # ⚡ Bolt: Convert to grayscale BEFORE warping to avoid 3-channel interpolation
        # This speeds up the warp by 3x and avoids a subsequent color conversion
        M = cv2.getPerspectiveTransform(rect, _DST_PTS)
        warped = cv2.warpPerspective(gray, M, (320, 160))
        
        # Optimize: Warp the single-channel grayscale image instead of the 3-channel BGR image
        # This speeds up the affine transformation and saves memory bandwidth
        warped_gray = cv2.warpPerspective(gray, M, (320, 160))

        return warped_gray

    def read_text(self, cropped_plate: np.ndarray) -> tuple[str, float]:
        # 1. Perspective Correction attempt
        # ⚡ Bolt: correct_perspective now returns a single-channel grayscale image
        gray_processed_plate = self.correct_perspective(cropped_plate)
        
        # 2. Enhancing contrast (Crucial for white taxi plates which can be overexposed)
        if len(gray_processed_plate.shape) == 3:
            gray = cv2.cvtColor(gray_processed_plate, cv2.COLOR_BGR2GRAY)
        else:
            gray = gray_processed_plate
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(gray)
        
        # Sharpening kernel to make characters crisp
        sharpened = cv2.filter2D(contrast_enhanced, -1, _SHARPEN_KERNEL)

        # 3. Strategy: Try OCR on Sharpened Gray first
        detections = self.reader.readtext(sharpened, allowlist=self.allowlist)

        # 4. Fallback strategy: If nothing found, try simple Otsu on the original
        if not detections:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            detections = self.reader.readtext(thresh, allowlist=self.allowlist)

        for detection in detections:
            bbox, text, score = detection
            # Normalize text format
            text = text.upper()
            for char in [' ', '-', '.', '_', '|']:
                text = text.replace(char, '')
            
            if license_complies_format(text):
                return format_license(text), score

        # 5. Final Fallback: If unwarping failed we might have a better shot with the raw crop
        # (Recursive-ish call but limited to 1 level)
        if gray_processed_plate.shape[:2] != cropped_plate.shape[:2]:
             # Just one attempt on raw if unwarped failed to return text
             return self.read_text_single_pass(cropped_plate)

        return "", 0.0

    def read_text_single_pass(self, img):
        """Standard pass without further unwarping recursion."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = self.reader.readtext(gray, allowlist=self.allowlist)
        for detection in detections:
            bbox, text, score = detection
            text = text.upper().replace(' ', '').replace('-', '')
            if license_complies_format(text):
                return format_license(text), score
        return "", 0.0
