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

# Pre-computed sets for O(1) lookups in tight OCR post-processing loops
VALID_LETTERS = frozenset(string.ascii_uppercase)
VALID_NUMBERS = frozenset('0123456789')


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

    # First 3 are ALWAYS letters
    for i in range(3):
        if text[i] not in VALID_LETTERS and text[i] not in dict_int_to_char:
            return False

    # Next 2 are ALWAYS numbers
    for i in range(3, 5):
        if text[i] not in VALID_NUMBERS and text[i] not in dict_char_to_int:
            return False

    # Last 1 can be a number (Car) OR a letter (Motorcycle)
    last_char = text[5]
    return (last_char in VALID_NUMBERS or last_char in dict_char_to_int or
            last_char in VALID_LETTERS or last_char in dict_int_to_char)


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.
    Formats the literal output to match expected standard (Car or Motorcycle).

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    if not text or len(text) < 6:
        return text

    # Decide if motorcycle based on the last character OCR result
    is_motorcycle = False
    last_char = text[5]
    # If the last character is explicitly a letter
    if last_char in VALID_LETTERS:
        is_motorcycle = True
    # OR if it's uniquely mapped to a letter going backwards
    elif last_char in dict_int_to_char and last_char not in dict_char_to_int:
        is_motorcycle = True

    # Use a list to construct the string, avoiding iterative concatenation
    formatted = []
    for j, c in enumerate(text):
        if j < 3: # Always letters
            formatted.append(dict_int_to_char.get(c, c))
        elif j < 5: # Always numbers
            formatted.append(dict_char_to_int.get(c, c))
        else: # Last char: letter if Moto, number if Car
            formatted.append(dict_int_to_char.get(c, c) if is_motorcycle else dict_char_to_int.get(c, c))

    return "".join(formatted)


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
    x1, y1, x2, y2, _, _ = license_plate

    # Return early when match is found, avoiding index lookups
    for vehicle in vehicle_track_ids:
        xcar1, ycar1, xcar2, ycar2, _ = vehicle
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return vehicle

    return -1, -1, -1, -1, -1
