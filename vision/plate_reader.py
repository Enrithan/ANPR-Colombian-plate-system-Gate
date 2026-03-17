import cv2
import string
import easyocr
import numpy as np
from .interfaces import IPlateReader

# Initialize the OCR reader once
_reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

class EasyOCRPlateReader(IPlateReader):
    def __init__(self):
        self.reader = _reader

    def license_complies_format(self, text):
        """
        Check if the license plate text complies with the required format.
        Colombian format Cars/Commercial: AAA123 (3 letters, 3 numbers)
        Colombian format Motorcycles: AAA12A (3 letters, 2 numbers, 1 letter)
        """
        if len(text) != 6:
            return False

        # First 3 are ALWAYS letters
        for i in range(3):
            if not (text[i] in string.ascii_uppercase or text[i] in dict_int_to_char.keys()):
                return False

        # Next 2 are ALWAYS numbers
        for i in range(3, 5):
            if not (text[i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[i] in dict_char_to_int.keys()):
                return False

        # Last 1 can be a number (Car) OR a letter (Motorcycle)
        last_char_valid_number = text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()
        last_char_valid_letter = text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()

        if last_char_valid_number or last_char_valid_letter:
            return True
        else:
            return False

    def format_license(self, text):
        """Format the literal output to match expected standard (Car or Motorcycle)."""
        license_plate_ = ''
        
        # Decide if motorcycle based on the last character OCR result
        is_motorcycle = False
        # If the last character is explicitly a letter (not a number in disguise based on our mapping and not a literal digit)
        # we lean towards it being a motorcycle plate format: AAA12A
        if text[5] in string.ascii_uppercase and text[5] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            is_motorcycle = True
        # OR if it's uniquely mapped to a letter going backwards
        elif text[5] in dict_int_to_char.keys() and text[5] not in dict_char_to_int.keys():
            # If it's a '0' trying to be an 'O', dict_int_to_char has '0':'O'
             is_motorcycle = True

        for j, c in enumerate(text):
            if j < 3: # Always letters
                if c in dict_int_to_char:
                    license_plate_ += dict_int_to_char[c]
                else:
                    license_plate_ += c
            elif j < 5: # Always numbers
                if c in dict_char_to_int:
                    license_plate_ += dict_char_to_int[c]
                else:
                    license_plate_ += c
            else: # Last char: letter if Moto, number if Car
                if is_motorcycle:
                    if c in dict_int_to_char:
                        license_plate_ += dict_int_to_char[c]
                    else:
                        license_plate_ += c
                else:
                    if c in dict_char_to_int:
                        license_plate_ += dict_char_to_int[c]
                    else:
                        license_plate_ += c
                        
        return license_plate_

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
                return self.format_license(text), score

        return "", 0.0
