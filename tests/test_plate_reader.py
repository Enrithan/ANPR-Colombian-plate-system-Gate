import pytest
from unittest.mock import patch, MagicMock
import sys

# Mock heavy dependencies before imports
mock_easyocr = MagicMock()
mock_ultralytics = MagicMock()
mock_cv2 = MagicMock()
mock_numpy = MagicMock()

with patch.dict(sys.modules, {
    'easyocr': mock_easyocr,
    'ultralytics': mock_ultralytics,
    'cv2': mock_cv2,
    'numpy': mock_numpy
}):
    from vision.plate_reader import EasyOCRPlateReader

class TestEasyOCRPlateReader:
    def setup_method(self):
        self.reader = EasyOCRPlateReader()

    def test_license_complies_format_valid_car(self):
        assert self.reader.license_complies_format("ABC123") is True
        assert self.reader.license_complies_format("XYZ789") is True

    def test_license_complies_format_valid_motorcycle(self):
        assert self.reader.license_complies_format("ABC12A") is True
        assert self.reader.license_complies_format("XYZ78Z") is True

    def test_license_complies_format_ocr_mappings_letters(self):
        # 0 maps to O, 1 maps to I, 3 maps to J, 4 maps to A, 5 maps to S, 6 maps to G
        assert self.reader.license_complies_format("0BC123") is True # 0 -> O
        assert self.reader.license_complies_format("A1C123") is True # 1 -> I
        assert self.reader.license_complies_format("AB3123") is True # 3 -> J
        assert self.reader.license_complies_format("4BC123") is True # 4 -> A
        assert self.reader.license_complies_format("5BC123") is True # 5 -> S
        assert self.reader.license_complies_format("6BC123") is True # 6 -> G
        assert self.reader.license_complies_format("013123") is True # 013 -> OIJ

    def test_license_complies_format_ocr_mappings_numbers(self):
        # O maps to 0, I maps to 1, J maps to 3, A maps to 4, S maps to 5, G maps to 6
        assert self.reader.license_complies_format("ABCO23") is True # O -> 0
        assert self.reader.license_complies_format("ABCI23") is True # I -> 1
        assert self.reader.license_complies_format("ABC1J3") is True # J -> 3
        assert self.reader.license_complies_format("ABC12A") is True # A is valid as motorcycle last char, or A->4
        assert self.reader.license_complies_format("ABC12S") is True # S is valid as motorcycle last char, or S->5
        assert self.reader.license_complies_format("ABC12G") is True # G is valid as motorcycle last char, or G->6
        assert self.reader.license_complies_format("ABCOIJ") is True # OIJ -> 013 (Valid Car)

    def test_license_complies_format_strict_positions(self):
        # Unmapped digit in letter position
        assert self.reader.license_complies_format("2BC123") is False
        assert self.reader.license_complies_format("A7C123") is False
        assert self.reader.license_complies_format("AB8123") is False
        assert self.reader.license_complies_format("999123") is False
        # Unmapped letter in middle number position
        assert self.reader.license_complies_format("ABCX23") is False
        assert self.reader.license_complies_format("ABC1Y3") is False
        assert self.reader.license_complies_format("ABCXY3") is False

    def test_license_complies_format_invalid_length(self):
        assert self.reader.license_complies_format("ABC12") is False
        assert self.reader.license_complies_format("ABC1234") is False
        assert self.reader.license_complies_format("") is False

    def test_license_complies_format_invalid_characters(self):
        assert self.reader.license_complies_format("abc123") is False  # Lowercase
        assert self.reader.license_complies_format("AB2234") is False  # Unmapped digit in letter position
        assert self.reader.license_complies_format("ABC1X3") is False  # Unmapped letter in middle digit position
        assert self.reader.license_complies_format("A!C123") is False  # Special character
        assert self.reader.license_complies_format("ABC 12") is False  # Space
