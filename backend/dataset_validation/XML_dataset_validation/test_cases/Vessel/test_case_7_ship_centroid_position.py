DESCRIPTION = """
Test Case: Ship Centroid Position Validation for Dark Vessel Detection

Purpose:
  Validate the presence and correctness of required fields in the 'ProcessingData/List_of_ships/Ship/Centroid_Position' section
  of the given XML data. Ensures fields follow specified formats and value ranges.

Inputs:
  - XML file for Dark Vessel Detection.
  - The test will extract and validate fields within the 'ProcessingData/List_of_ships/Ship/Centroid_Position' child element.
  - Ranges and coditions that need to be checked for each fields within the 'ProcessingData/List_of_ships/Ship/Centroid_Position':
    * Latitude:         [-90.00 .. 90.00]
    * Longitude:        [-180.00 .. 180.00]
    * SARData_Sample:   [0 .. 50000]
    * SARData_Line:     [0 .. 20000]
    * Scene_Sample:     [0 .. 512]
    * Scene_Line:       [0 .. 512] 

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
    'ProcessingData/List_of_ships',
]

RANGES = {
    'Latitude': '[-90.00 .. 90.00] (float)',
    'Longitude': '[-180.00 .. 180.00] (float)',
    'SARData_Sample': '[0 .. 50000] (float)',
    'SARData_Line': '[0 .. 20000] (float)',
    'Scene_Sample': '[0 .. 512] (float)',
    'Scene_Line': '[0 .. 512] (float)',
}

def check_float_field(centroid, field_name, min_val, max_val, ship_index, issues):
    """
    Check if the given field exists, is a float, and lies within the specified range.

    Args:
        centroid: XML element for Centroid_Position.
        field_name: The field tag to check.
        min_val: Minimum acceptable value.
        max_val: Maximum acceptable value.
        ship_index: Index of the current ship (for error reporting).
        issues: List to append error messages to.
    """
    val_str = centroid.findtext(field_name)
    if val_str is None:
        issues.append(f'Ship_{ship_index}: missing {field_name}')
        return
    try:
        val = float(val_str)
        if not (min_val <= val <= max_val):
            issues.append(f'Ship_{ship_index}: {field_name} value {val} out of range [{min_val}..{max_val}]')
    except ValueError:
        issues.append(f'Ship_{ship_index}: {field_name} value "{val_str}" is not a float')

def check_centroid_position(ship, ship_index):
    """
    Validate the Centroid_Position fields for a single ship.

    Args:
        ship: XML element for the ship.
        ship_index: Index of the ship (for error messages).

    Returns:
        A list of issue strings found during validation.
    """
    issues = []

    centroid = ship.find('Centroid_Position')
    if centroid is None:
        issues.append(f'Ship_{ship_index}: missing Centroid_Position')
        return issues

    check_float_field(centroid, 'Latitude', -90.0, 90.0, ship_index, issues)
    check_float_field(centroid, 'Longitude', -180.0, 180.0, ship_index, issues)
    check_float_field(centroid, 'SARData_Sample', 0, 50000, ship_index, issues)
    check_float_field(centroid, 'SARData_Line', 0, 20000, ship_index, issues)
    check_float_field(centroid, 'Scene_Sample', 0, 512, ship_index, issues)
    check_float_field(centroid, 'Scene_Line', 0, 512, ship_index, issues)

    return issues

def run_test(xml_root):
    """
    Main test function to validate centroid position info for all ships.

    Steps:
      1. Check for ProcessingData element.
      2. Validate Number_of_ships element.
      3. If Number_of_ships > 0, validate each ship's Centroid_Position.
      4. Return all accumulated issues.

    Args:
        xml_root: Root XML element.

    Returns:
        Dictionary with 'status' and 'details' about missing or invalid data.
    """
    issues = []

    # Check for ProcessingData element
    processing_data = xml_root.find('ProcessingData')
    if processing_data is None:
        issues.append('ProcessingData')
        return {'status': 'FAIL', 'details': {'missing': issues}}

    # Get Number_of_ships and validate
    num_ships_str = processing_data.findtext('StatisticsReport/Number_of_ships')
    if num_ships_str is None:
        issues.append('ProcessingData/StatisticsReport/Number_of_ships')
        return {'status': 'FAIL', 'details': {'missing': issues}}

    try:
        num_ships = int(num_ships_str)
    except ValueError:
        issues.append('Number_of_ships is not an integer')
        return {'status': 'FAIL', 'details': {'missing': issues}}

    # If no ships, no further checks needed
    if num_ships == 0:
        return {'status': 'PASS', 'details': {}}

    # Get list of ships
    ships = processing_data.findall('List_of_ships/Ship')
    if not ships:
        issues.append('ProcessingData/List_of_ships/Ship')
        return {'status': 'FAIL', 'details': {'missing': issues}}

    # Check number of Ship elements matches Number_of_ships
    if len(ships) != num_ships:
        issues.append(f"Number_of_ships ({num_ships}) does not match number of Ship entries ({len(ships)})")

    # Validate centroid position for each ship
    for i, ship in enumerate(ships, start=1):
        ship_issues = check_centroid_position(ship, i)
        issues.extend(ship_issues)

    status = 'PASS' if not issues else 'FAIL'
    return {'status': status, 'details': {'missing': issues}}
