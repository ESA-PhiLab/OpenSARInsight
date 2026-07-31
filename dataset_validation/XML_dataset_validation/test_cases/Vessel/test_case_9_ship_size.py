DESCRIPTION = """
Test Case: Ship Size Validation for Dark Vessel Detection

Purpose:
  Validate the presence and correctness of required fields in the 'ProcessingData/List_of_ships/Ship/Size' section
  of the given XML data. Ensures fields follow specified formats and value ranges.

Inputs:
  - XML file for Dark Vessel Detection.
  - The test will extract and validate fields within the 'ProcessingData/List_of_ships/Ship/Size' child element.
  - Ranges and coditions that need to be checked for each fields within the 'ProcessingData/List_of_ships/Ship/Size':
    * Size:         [1 .. 1000]

Outputs:
    Pass criteria:
      - 'status' is 'PASS' if all required fields exist and are valid.
      - No missing fields.
      - No invalid format or out-of-range values.
      - No error messages in the 'details' dictionary.
    Fail criteria:
      - 'status' is 'FAIL' if one or more required fields are missing.
      - Or any field contains invalid format or value outside defined ranges.
      - The 'details' dictionary contains key 'missing' listing:
        * Missing fields.
        * Fields with invalid format.
        * Fields with out-of-range or unexpected values.
"""

REQUIRED = [
    'ProcessingData',
    'ProcessingData/StatisticsReport/Number_of_ships',
]

RANGES = {
    'ProcessingData/List_of_ships/Ship/Size': (1.0, 1000.0),  # float range [1..1000]
}


def validate_size_field(ship_element, ship_index, missing_or_invalid, min_val, max_val):
    """
    Validates the 'Size' element of a ship.
    - Must exist
    - Must be float within [min_val, max_val]
    - NaN is ignored
    Updates 'missing_or_invalid' dict with 'missing' or 'warnings'.
    """
    size_text = ship_element.findtext('Size')
    if size_text is None:
        missing_or_invalid['missing'].append(f"Ship {ship_index}: Missing Size")
        return

    size_text = size_text.strip()
    if size_text.lower() == 'nan':
        # Ignore NaN (not missing, not warning)
        return

    try:
        size_val = float(size_text)
    except ValueError:
        missing_or_invalid['missing'].append(f"Ship {ship_index}: Size value '{size_text}' is not a valid float")
        return

    if not (min_val <= size_val <= max_val):
        missing_or_invalid['warnings'].append(f"Ship {ship_index}: Size value {size_val} out of range [{min_val}..{max_val}]")


def run_test(xml_root):
    """
    Main test executor for validating ship sizes under ProcessingData.
    """
    missing_or_invalid = {
        'missing': [],
        'warnings': []
    }

    # Validate existence of ProcessingData
    processing_data = xml_root.find('ProcessingData')
    if processing_data is None:
        missing_or_invalid['missing'].append("Missing required element: ProcessingData")
        return {'status': 'FAIL', 'details': missing_or_invalid}

    # Validate existence and value of Number_of_ships
    num_ships_text = processing_data.findtext('StatisticsReport/Number_of_ships')
    if num_ships_text is None:
        missing_or_invalid['missing'].append("Missing required element: ProcessingData/StatisticsReport/Number_of_ships")
        return {'status': 'FAIL', 'details': missing_or_invalid}

    try:
        num_ships = int(num_ships_text.strip())
    except ValueError:
        missing_or_invalid['missing'].append(f"Invalid Number_of_ships: '{num_ships_text}' is not an integer")
        return {'status': 'FAIL', 'details': missing_or_invalid}

    if num_ships == 0:
        return {'status': 'PASS', 'details': missing_or_invalid}

    # Number_of_ships > 0 => Validate List_of_ships and Ship/Size
    list_of_ships = processing_data.find('List_of_ships')
    if list_of_ships is None:
        missing_or_invalid['missing'].append("Missing required element: ProcessingData/List_of_ships")
        return {'status': 'FAIL', 'details': missing_or_invalid}

    ships = list_of_ships.findall('Ship')
    for i, ship in enumerate(ships, start=1):
        validate_size_field(ship, i, missing_or_invalid, *RANGES['ProcessingData/List_of_ships/Ship/Size'])

    # Determine status
    if missing_or_invalid['missing']:
        return {'status': 'FAIL', 'details': missing_or_invalid}
    elif missing_or_invalid['warnings']:
        return {'status': 'WARNING', 'details': missing_or_invalid}
    else:
        return {'status': 'PASS', 'details': missing_or_invalid}
