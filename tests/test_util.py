import pytest
from unittest.mock import patch, MagicMock
import sys

# Mock dependencies at the very top, before any other imports
mock_easyocr = MagicMock()
mock_ultralytics = MagicMock()
mock_cv2 = MagicMock()

with patch.dict(sys.modules, {
    'easyocr': mock_easyocr,
    'ultralytics': mock_ultralytics,
    'cv2': mock_cv2
}):
    from util import license_complies_format

def test_license_complies_format_valid_car():
    assert license_complies_format("ABC123") is True
    assert license_complies_format("XYZ789") is True
    assert license_complies_format("QWE000") is True

def test_license_complies_format_valid_motorcycle():
    assert license_complies_format("ABC12A") is True
    assert license_complies_format("XYZ78Z") is True

def test_license_complies_format_ocr_mapping_letters():
    # '0' maps to 'O', '1' maps to 'I', '3' maps to 'J', '4' maps to 'A', '6' maps to 'G', '5' maps to 'S'
    assert license_complies_format("013123") is True # OIJ123
    assert license_complies_format("465123") is True # AGS123
    assert license_complies_format("AB0123") is True # ABO123
    assert license_complies_format("1BC123") is True # IBC123

def test_license_complies_format_ocr_mapping_numbers():
    # 'O' maps to '0', 'I' maps to '1', 'J' maps to '3', 'A' maps to '4', 'G' maps to '6', 'S' maps to '5'
    assert license_complies_format("ABCOIJ") is True # ABC013
    assert license_complies_format("ABCAGS") is True # ABC465
    assert license_complies_format("ABC12O") is True # ABC120
    assert license_complies_format("ABCO23") is True # ABC023

def test_license_complies_format_ocr_mapping_motorcycle():
    # '0' mapped to 'O' at the end is a valid motorcycle letter
    assert license_complies_format("ABC120") is True # ABC12O
    assert license_complies_format("013OI0") is True # OIJ01O

def test_license_complies_format_invalid_length():
    assert license_complies_format("ABC12") is False
    assert license_complies_format("ABC1234") is False
    assert license_complies_format("") is False

def test_license_complies_format_invalid_characters():
    assert license_complies_format("abc123") is False  # Lowercase

    # '2' is not in string.ascii_uppercase and not in dict_int_to_char ('0','1','3','4','6','5')
    assert license_complies_format("2BC123") is False

    # 'X' is not in ['0'..'9'] and not in dict_char_to_int ('O','I','J','A','G','S')
    assert license_complies_format("ABC1X3") is False
    assert license_complies_format("ABCX23") is False

    # Both letters and numbers invalid
    assert license_complies_format("222XXX") is False

    # Special characters and spaces
    assert license_complies_format("A!C123") is False  # Special character
    assert license_complies_format("ABC 12") is False  # Space

def test_license_complies_format_non_string():
    # The function currently doesn't check for type, so it will raise TypeError on slicing if not string/sequence
    with pytest.raises(TypeError):
        license_complies_format(None)
    with pytest.raises(TypeError):
        license_complies_format(123456)
