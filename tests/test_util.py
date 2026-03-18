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

def test_license_complies_format_valid():
    assert license_complies_format("ABC123") is True
    assert license_complies_format("XYZ789") is True

def test_license_complies_format_invalid_length():
    assert license_complies_format("ABC12") is False
    assert license_complies_format("ABC1234") is False
    assert license_complies_format("") is False

def test_license_complies_format_invalid_characters():
    assert license_complies_format("abc123") is False  # Lowercase
    assert license_complies_format("AB1234") is False  # Digit in letter position
    assert license_complies_format("ABC12A") is False  # Letter in digit position
    assert license_complies_format("A!C123") is False  # Special character
    assert license_complies_format("ABC 12") is False  # Space

def test_license_complies_format_non_string():
    # The function currently doesn't check for type, so it will raise TypeError on slicing if not string/sequence
    with pytest.raises(TypeError):
        license_complies_format(None)
    with pytest.raises(TypeError):
        license_complies_format(123456)
