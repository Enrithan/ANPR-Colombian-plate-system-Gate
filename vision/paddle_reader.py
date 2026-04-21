import cv2
import numpy as np
import logging
import os
os.environ["FLAGS_enable_pir_api"] = "0"
from paddleocr import PaddleOCR
from .interfaces import IPlateReader
from util import license_complies_format, format_license

# Initialize PaddleOCR globally so we don't reload the models on every frame.
# We use the mobile English/number models which are extremely fast and small, perfect for edge.
logging.getLogger("ppocr").setLevel(logging.ERROR) # Suppress verbose paddle logs
_paddle_reader = PaddleOCR(use_angle_cls=False, lang='en', enable_mkldnn=False)

class PaddleOCRPlateReader(IPlateReader):
    def __init__(self):
        self.reader = _paddle_reader
        print("[AI] PaddleOCR Plate Reader (Edge Mobile v4) initialized.")

    def correct_perspective(self, img):
        """Finds the 4 corners of the plate and warps it to a clean rectangle."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Poka-yoke: Skip for very small crops
        if gray.shape[0] < 20 or gray.shape[1] < 40:
            return img

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(blur, 30, 150)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        screen_cnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                if cv2.contourArea(c) > 500:
                    screen_cnt = approx
                    break
        
        if screen_cnt is None:
            return img 

        pts = screen_cnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left
        rect[2] = pts[np.argmax(s)] # Bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-right
        rect[3] = pts[np.argmax(diff)] # Bottom-left

        dst = np.array([
            [0, 0],
            [320 - 1, 0],
            [320 - 1, 160 - 1],
            [0, 160 - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        # ⚡ Bolt: Convert to grayscale BEFORE warping to avoid 3-channel interpolation
        # This speeds up the warp by 3x and avoids a subsequent color conversion
        return cv2.warpPerspective(gray, M, (320, 160))

    def read_text(self, cropped_plate: np.ndarray) -> tuple[str, float]:
        processed_plate = self.correct_perspective(cropped_plate)
        
        if len(processed_plate.shape) == 3:
            gray = cv2.cvtColor(processed_plate, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed_plate

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(gray)
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)

        # PaddleOCR expects 3-channel inputs
        sharpened_3c = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        results = self.reader.ocr(sharpened_3c)

        # Fallback to simple grayscale if sharpening hurt the read
        if not results or not results[0]:
            gray_3c = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            results = self.reader.ocr(gray_3c)

        if results and results[0]:
            for line in results[0]:
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue
                res = line[1]
                if isinstance(res, (list, tuple)) and len(res) >= 2:
                    text, score = res[0], float(res[1])
                elif isinstance(res, str):
                    text, score = res, 1.0
                else:
                    try: text, score = str(res[0]), float(res[1])
                    except: text, score = str(res), 1.0

                text = text.upper().replace(' ', '').replace('-', '').replace('.', '').replace('_', '').replace('|', '')
                print(f"[OCR Raw] Extracted: '{text}' (Conf: {score:.2f})")
                if license_complies_format(text):
                    return format_license(text), score
        return "", 0.0

    def read_text_single_pass(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_3c = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        results = self.reader.ocr(gray_3c)
        if results and results[0]:
            for line in results[0]:
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue
                res = line[1]
                if isinstance(res, (list, tuple)) and len(res) >= 2:
                    text, score = res[0], float(res[1])
                elif isinstance(res, str):
                    text, score = res, 1.0
                else:
                    try: text, score = str(res[0]), float(res[1])
                    except: text, score = str(res), 1.0

                text = text.upper().replace(' ', '').replace('-', '')
                print(f"[OCR Raw] Extracted (fallback): '{text}' (Conf: {score:.2f})")
                if license_complies_format(text):
                    return format_license(text), score
        return "", 0.0
