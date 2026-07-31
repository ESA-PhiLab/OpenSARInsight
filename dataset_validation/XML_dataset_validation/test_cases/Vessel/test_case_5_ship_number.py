DESCRIPTION = """
Test Case: Number_of_ships Validation for Dark Vessel Detection

Purpose:
  Validate the presence and correctness of required fields in the 'ProcessingData/StatisticsReport/Number_of_ships' section
  of the given XML data. Ensures fields follow specified formats and value ranges.

Inputs:
  - XML file for Dark Vessel Detection.
  - The test will extract and validate fields within the 'ProcessingData/StatisticsReport/Number_of_ships' child element.
  - Ranges and coditions that need to be checked for each fields within the 'ProcessingData/StatisticsReport/Number_of_ships':
   * Number_of_ships      : [0 .. infinite], Matches the number of <Ship> elements in ProcessingData/List_of_ships/Ship

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
    'ProcessingData/StatisticsReport',
    'ProcessingData/StatisticsReport/Number_of_ships',
]

RANGES = {
    'ProcessingData/StatisticsReport/Number_of_ships': (0, None),  # range: [0 .. infinite], Matches the number of <Ship> elements in ProcessingData/List_of_ships/Ship
}

def validate_number_of_ships(xml_root, text, errors, min_val, max_val):
    """
    Validate that Number_of_ships:
      - Exists
      - Is a valid integer ≥ min_val (and ≤ max_val if specified)
      - Matches the number of <Ship> elements in ProcessingData/List_of_ships/Ship
    """
    if text is None:
        errors.append("Missing Number_of_ships element text")
        return
    try:
        value = int(text.strip())
        if value < min_val:
            errors.append(f"Number_of_ships value {value} is less than minimum allowed {min_val}")
        elif max_val is not None and value > max_val:
            errors.append(f"Number_of_ships value {value} exceeds maximum allowed {max_val}")
        else:
            # Check against number of <Ship> nodes
            ship_elements = xml_root.findall('ProcessingData/List_of_ships/Ship')
            actual_count = len(ship_elements)
            if value != actual_count:
                errors.append(
                    f"Number_of_ships is {value}, but {actual_count} <Ship> elements found in ProcessingData/List_of_ships"
                )
    except ValueError:
        errors.append("Number_of_ships: Invalid integer format")

def run_test(xml_root):
    """
    Run validation:
      - Ensure required elements exist
      - Validate 'ProcessingData/StatisticsReport/Number_of_ships' value and consistency with <Ship> list
    """
    missing_or_invalid = []

    # Step 1: Check for required elements
    for path in REQUIRED:
        if xml_root.find(path) is None:
            missing_or_invalid.append(f"Missing required element: {path}")

    # Step 2: Exit early if any elements are missing
    if missing_or_invalid:
        return {'status': 'FAIL', 'details': {'missing': missing_or_invalid}}

    # Step 3: Validate content
    num_ships = xml_root.findtext('ProcessingData/StatisticsReport/Number_of_ships')
    validate_number_of_ships(xml_root, num_ships, missing_or_invalid, *RANGES['ProcessingData/StatisticsReport/Number_of_ships'])

    status = 'PASS' if not missing_or_invalid else 'FAIL'
    return {
        'status': status,
        'details': {'missing': missing_or_invalid}
    }
