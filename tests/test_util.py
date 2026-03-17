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
    from util import license_complies_format, get_car

def test_license_complies_format_valid():
    assert license_complies_format("ABC123") is True
    assert license_complies_format("XYZ789") is True

def test_license_complies_format_invalid_length():
    assert license_complies_format("ABC12") is False
    assert license_complies_format("ABC1234") is False
    assert license_complies_format("") is False

def test_license_complies_format_invalid_characters():
    assert license_complies_format("abc123") is False  # Lowercase
    assert license_complies_format("AB8234") is False  # Digit in letter position (8 not in dict_int_to_char)
    assert license_complies_format("ABCH23") is False  # Letter in digit position (H not in dict_char_to_int)
    assert license_complies_format("A!C123") is False  # Special character
    assert license_complies_format("ABC 12") is False  # Space

def test_license_complies_format_non_string():
    # The function currently doesn't check for type, so it will raise TypeError on slicing if not string/sequence
    with pytest.raises(TypeError):
        license_complies_format(None)
    with pytest.raises(TypeError):
        license_complies_format(123456)


def test_get_car_found():
    license_plate = (10, 10, 20, 20, 0.9, 0)
    vehicle_track_ids = [
        [0, 0, 30, 30, 1], # Vehicle 1 contains the plate
        [40, 40, 50, 50, 2]
    ]
    result = get_car(license_plate, vehicle_track_ids)
    assert result == [0, 0, 30, 30, 1]

def test_get_car_not_found():
    license_plate = (100, 100, 120, 120, 0.9, 0)
    vehicle_track_ids = [
        [0, 0, 30, 30, 1],
        [40, 40, 50, 50, 2]
    ]
    result = get_car(license_plate, vehicle_track_ids)
    assert result == (-1, -1, -1, -1, -1)

def test_get_car_multiple_vehicles():
    license_plate = (45, 45, 48, 48, 0.9, 0)
    vehicle_track_ids = [
        [0, 0, 30, 30, 1],
        [40, 40, 50, 50, 2] # Vehicle 2 contains the plate
    ]
    result = get_car(license_plate, vehicle_track_ids)
    assert result == [40, 40, 50, 50, 2]

def test_get_car_no_vehicles():
    license_plate = (10, 10, 20, 20, 0.9, 0)
    vehicle_track_ids = []
    result = get_car(license_plate, vehicle_track_ids)
    assert result == (-1, -1, -1, -1, -1)

def test_get_car_exact_match():
    license_plate = (10, 10, 20, 20, 0.9, 0)
    vehicle_track_ids = [
        [10, 10, 20, 20, 1] # Exact match
    ]
    # Because get_car uses strictly > and < for bounds checking,
    # an exact boundary match will return not found
    result = get_car(license_plate, vehicle_track_ids)
    assert result == (-1, -1, -1, -1, -1)
