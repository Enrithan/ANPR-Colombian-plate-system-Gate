import string
import easyocr
import csv
# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

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


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.
    Formats the literal output to match expected standard (Car or Motorcycle).

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
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


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, None


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

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
