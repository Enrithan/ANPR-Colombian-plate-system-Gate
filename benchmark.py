import timeit
import string

dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

def license_complies_format_old(text):
    if len(text) != 6:
        return False
    for i in range(3):
        if not (text[i] in string.ascii_uppercase or text[i] in dict_int_to_char.keys()):
            return False
    for i in range(3, 5):
        if not (text[i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[i] in dict_char_to_int.keys()):
            return False
    last_char_valid_number = text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()
    last_char_valid_letter = text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()
    if last_char_valid_number or last_char_valid_letter:
        return True
    else:
        return False

def format_license_old(text):
    if len(text) != 6:
        return text
    license_plate_ = ''
    is_motorcycle = False
    if text[5] in string.ascii_uppercase and text[5] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        is_motorcycle = True
    elif text[5] in dict_int_to_char.keys() and text[5] not in dict_char_to_int.keys():
         is_motorcycle = True

    for j, c in enumerate(text):
        if j < 3:
            if c in dict_int_to_char:
                license_plate_ += dict_int_to_char[c]
            else:
                license_plate_ += c
        elif j < 5:
            if c in dict_char_to_int:
                license_plate_ += dict_char_to_int[c]
            else:
                license_plate_ += c
        else:
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

VALID_LETTERS = frozenset(string.ascii_uppercase)
VALID_NUMBERS = frozenset('0123456789')
INT_TO_CHAR_KEYS = frozenset(dict_int_to_char.keys())
CHAR_TO_INT_KEYS = frozenset(dict_char_to_int.keys())

def license_complies_format_new(text):
    if len(text) != 6:
        return False
    for i in range(3):
        if not (text[i] in VALID_LETTERS or text[i] in INT_TO_CHAR_KEYS):
            return False
    for i in range(3, 5):
        if not (text[i] in VALID_NUMBERS or text[i] in CHAR_TO_INT_KEYS):
            return False
    last_char_valid_number = text[5] in VALID_NUMBERS or text[5] in CHAR_TO_INT_KEYS
    last_char_valid_letter = text[5] in VALID_LETTERS or text[5] in INT_TO_CHAR_KEYS
    if last_char_valid_number or last_char_valid_letter:
        return True
    return False

def format_license_new(text):
    if len(text) != 6:
        return text
    is_motorcycle = False
    last_char = text[5]
    if last_char in VALID_LETTERS and last_char not in VALID_NUMBERS:
        is_motorcycle = True
    elif last_char in INT_TO_CHAR_KEYS and last_char not in CHAR_TO_INT_KEYS:
         is_motorcycle = True

    chars = []
    for j, c in enumerate(text):
        if j < 3:
            chars.append(dict_int_to_char.get(c, c))
        elif j < 5:
            chars.append(dict_char_to_int.get(c, c))
        else:
            if is_motorcycle:
                chars.append(dict_int_to_char.get(c, c))
            else:
                chars.append(dict_char_to_int.get(c, c))
    return "".join(chars)

test_strings = ['AAA123', 'B0B456', 'XYZ12A', '123ABC']

print("Old complies:", timeit.timeit('for s in test_strings: license_complies_format_old(s)', globals=globals(), number=100000))
print("New complies:", timeit.timeit('for s in test_strings: license_complies_format_new(s)', globals=globals(), number=100000))

print("Old format:", timeit.timeit('for s in test_strings: format_license_old(s)', globals=globals(), number=100000))
print("New format:", timeit.timeit('for s in test_strings: format_license_new(s)', globals=globals(), number=100000))
