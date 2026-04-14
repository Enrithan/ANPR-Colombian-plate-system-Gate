import string
import csv
# REDUNDANT: Reader moved to vision/plate_reader.py to avoid multiple instances
# reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

# ⚡ Bolt: Pre-computed frozensets for O(1) character lookups in tight OCR loops
VALID_LETTERS = frozenset(string.ascii_uppercase) | frozenset(dict_int_to_char)
VALID_NUMBERS = frozenset("0123456789") | frozenset(dict_char_to_int)

def write_csv(results: dict, output_path: str):
    """
    Write the ANPR results to a CSV file with Colombian-plate compliance flag.

    Columns:
    - frame_nmr
    - car_id
    - car_x1, car_y1, car_x2, car_y2
    - plate_x1, plate_y1, plate_x2, plate_y2
    - plate_conf  (detection confidence)
    - plate_text
    - text_conf   (OCR confidence)
    - complies_colombia (True/False)
    """
    fieldnames = [
        'frame_nmr', 'car_id',
        'car_x1','car_y1','car_x2','car_y2',
        'plate_x1','plate_y1','plate_x2','plate_y2',
        'plate_conf','plate_text','text_conf',
        'complies_colombia'
    ]

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for frame_nmr, cars in results.items():
            for car_id, info in cars.items():
                # Skip incomplete entries
                if not all(k in info for k in ('car','license_plate')):
                    continue
                lp = info['license_plate']
                text = lp.get('text','').upper()
                row = {
                    'frame_nmr': frame_nmr,
                    'car_id':    car_id,
                    'car_x1':    info['car']['bbox'][0],
                    'car_y1':    info['car']['bbox'][1],
                    'car_x2':    info['car']['bbox'][2],
                    'car_y2':    info['car']['bbox'][3],
                    'plate_x1':  lp['bbox'][0],
                    'plate_y1':  lp['bbox'][1],
                    'plate_x2':  lp['bbox'][2],
                    'plate_y2':  lp['bbox'][3],
                    'plate_conf':lp['bbox_score'],
                    'plate_text':text,
                    'text_conf': lp['text_score'],
                    'complies_colombia': license_complies_format(text)
                }
                writer.writerow(row)


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.
    Colombian format Cars/Commercial: AAA123 (3 letters, 3 numbers)
    Colombian format Motorcycles: AAA12A (3 letters, 2 numbers, 1 letter)

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 6:
        return False

    # ⚡ Bolt: Optimized string validation by avoiding repetitive list allocation
    # First 3 are ALWAYS letters
    if not (text[0] in VALID_LETTERS and text[1] in VALID_LETTERS and text[2] in VALID_LETTERS):
        return False

    # Next 2 are ALWAYS numbers
    if not (text[3] in VALID_NUMBERS and text[4] in VALID_NUMBERS):
        return False

    # Last 1 can be a number (Car) OR a letter (Motorcycle)
    return text[5] in VALID_NUMBERS or text[5] in VALID_LETTERS


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.
    Formats the literal output to match expected standard (Car or Motorcycle).

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    # ⚡ Bolt: Use .get() instead of membership + indexing, and "".join() instead of +=
    last_char = text[5]

    # Decide if motorcycle based on the last character OCR result
    # If the last character is explicitly a letter (not a literal digit)
    # OR if it's uniquely mapped to a letter going backwards ('0' -> 'O')
    is_motorcycle = (last_char in VALID_LETTERS and not last_char.isdigit()) or \
                    (last_char in dict_int_to_char and last_char not in dict_char_to_int)

    return "".join([
        dict_int_to_char.get(text[0], text[0]),
        dict_int_to_char.get(text[1], text[1]),
        dict_int_to_char.get(text[2], text[2]),
        dict_char_to_int.get(text[3], text[3]),
        dict_char_to_int.get(text[4], text[4]),
        dict_int_to_char.get(last_char, last_char) if is_motorcycle else dict_char_to_int.get(last_char, last_char)
    ])


# UNUSED: read_license_plate removed in favor of vision/plate_reader.py


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    # ⚡ Bolt: Iterate directly over items and return early to avoid index lookups
    for vehicle in vehicle_track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return vehicle

    return -1, -1, -1, -1, -1
