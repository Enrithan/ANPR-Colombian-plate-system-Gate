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

def test_license_complies_format_valid_car():
    assert license_complies_format("ABC123") is True
    assert license_complies_format("XYZ789") is True

def test_license_complies_format_valid_motorcycle():
    assert license_complies_format("ABC12A") is True
    assert license_complies_format("XYZ78Z") is True

def test_license_complies_format_ocr_mappings_letters():
    # 0 maps to O, 1 maps to I, 3 maps to J, 4 maps to A, 5 maps to S, 6 maps to G
    assert license_complies_format("0BC123") is True # 0 -> O
    assert license_complies_format("A1C123") is True # 1 -> I
    assert license_complies_format("AB3123") is True # 3 -> J
    assert license_complies_format("4BC123") is True # 4 -> A
    assert license_complies_format("5BC123") is True # 5 -> S
    assert license_complies_format("6BC123") is True # 6 -> G
    assert license_complies_format("013123") is True # 013 -> OIJ

def test_license_complies_format_ocr_mappings_numbers():
    # O maps to 0, I maps to 1, J maps to 3, A maps to 4, S maps to 5, G maps to 6
    assert license_complies_format("ABCO23") is True # O -> 0
    assert license_complies_format("ABCI23") is True # I -> 1
    assert license_complies_format("ABC1J3") is True # J -> 3
    assert license_complies_format("ABC12A") is True # A is valid as motorcycle last char, or A->4
    assert license_complies_format("ABC12S") is True # S is valid as motorcycle last char, or S->5
    assert license_complies_format("ABC12G") is True # G is valid as motorcycle last char, or G->6
    assert license_complies_format("ABCOIJ") is True # OIJ -> 013 (Valid Car)

def test_license_complies_format_strict_positions():
    # Unmapped digit in letter position
    assert license_complies_format("2BC123") is False
    assert license_complies_format("A7C123") is False
    assert license_complies_format("AB8123") is False
    assert license_complies_format("999123") is False
    # Unmapped letter in middle number position
    assert license_complies_format("ABCX23") is False
    assert license_complies_format("ABC1Y3") is False
    assert license_complies_format("ABCXY3") is False

def test_license_complies_format_invalid_length():
    assert license_complies_format("ABC12") is False
    assert license_complies_format("ABC1234") is False
    assert license_complies_format("") is False

def test_license_complies_format_invalid_characters():
    assert license_complies_format("abc123") is False  # Lowercase
    assert license_complies_format("AB2234") is False  # Unmapped digit in letter position
    assert license_complies_format("ABC1X3") is False  # Unmapped letter in middle digit position
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
