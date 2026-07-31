DESCRIPTION = """
Test Case: Ship Bounding Box Validation for Dark Vessel Detection

Purpose:
  Validate the presence and correctness of required fields in the 'ProcessingData/List_of_ships/Ship/Centroid_Position' section
  of the given XML data. Ensures fields follow specified formats and value ranges.

Inputs:
  - XML file for Dark Vessel Detection.
  - The test will extract and validate fields within the 'ProcessingData/List_of_ships/Ship/Centroid_Position' child element.
  - Ranges and coditions that need to be checked for each fields within the 'ProcessingData/List_of_ships/Ship/Centroid_Position':
    * Top:         [1 .. 512],Top ≤ Bottom
    * Left:        [1 .. 512],Left ≤ Right
    * Right:       [1 .. 512],Right ≥ Left
    * Bottom:      [1 .. 512],Bottom ≥ Top

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
    'ProcessingData/List_of_ships/Ship/BoundingBox/Top',
    'ProcessingData/List_of_ships/Ship/BoundingBox/Left',
    'ProcessingData/List_of_ships/Ship/BoundingBox/Right',
    'ProcessingData/List_of_ships/Ship/BoundingBox/Bottom',
]

RANGES = {
    'ProcessingData/List_of_ships/Ship/BoundingBox/Top': '[1 .. 512] (int, Top ≤ Bottom)',
    'ProcessingData/List_of_ships/Ship/BoundingBox/Left': '[1 .. 512] (int, Left ≤ Right)',
    'ProcessingData/List_of_ships/Ship/BoundingBox/Right': '[1 .. 512] (int, Right ≥ Left)',
    'ProcessingData/List_of_ships/Ship/BoundingBox/Bottom': '[1 .. 512] (int, Bottom ≥ Top)',
}

def check_int_field(bbox, tag, min_val, max_val):
    """
    Check if the value for the given tag exists, is an integer,
    and falls within the specified range [min_val, max_val].

    Args:
        bbox: XML element containing BoundingBox data.
        tag: Tag name to find inside bbox.
        min_val: Minimum valid integer value.
        max_val: Maximum valid integer value.

    Returns:
        Integer value if valid,
        Otherwise, a string describing the error.
    """
    val_str = bbox.findtext(tag)
    if val_str is None:
        return f"{tag} is missing"
    if val_str.strip().lower() == 'nan':
        return None  # skip NaN values
    try:
        val = int(val_str)
        if not (min_val <= val <= max_val):
            return f"{tag} value '{val}' out of range [{min_val}..{max_val}]"
        return val
    except ValueError:
        return f"{tag} value '{val_str}' is not an integer"

def check_bounding_box(bbox, ship_index):
    """
    Validate all BoundingBox fields for a single ship.

    Checks that Top, Left, Right, Bottom are present, integers within the
    valid range, and satisfy the logical constraints:
      - Top ≤ Bottom
      - Left ≤ Right

    Args:
        bbox: XML element containing the BoundingBox.
        ship_index: Index of the ship (used for error messages).

    Returns:
        A list of issue strings found during validation.
    """
    issues = []

    # Validate each bounding box field
    top = check_int_field(bbox, 'Top', 1, 512)
    left = check_int_field(bbox, 'Left', 1, 512)
    right = check_int_field(bbox, 'Right', 1, 512)
    bottom = check_int_field(bbox, 'Bottom', 1, 512)

    # Check logical constraint: Top ≤ Bottom
    if isinstance(top, int) and isinstance(bottom, int):
        if top > bottom:
            issues.append(f"Ship {ship_index}: Top ({top}) > Bottom ({bottom})")
    else:
        # Append any errors found during Top/Bottom validation
        if isinstance(top, str):
            issues.append(f"Ship {ship_index}: {top}")
        if isinstance(bottom, str):
            issues.append(f"Ship {ship_index}: {bottom}")

    # Check logical constraint: Left ≤ Right
    if isinstance(left, int) and isinstance(right, int):
        if left > right:
            issues.append(f"Ship {ship_index}: Left ({left}) > Right ({right})")
    else:
        # Append any errors found during Left/Right validation
        if isinstance(left, str):
            issues.append(f"Ship {ship_index}: {left}")
        if isinstance(right, str):
            issues.append(f"Ship {ship_index}: {right}")

    return issues

def run_test(xml_root):
    """
    Main test function to validate the BoundingBox information for all ships.

    Steps:
      1. Check for presence of ProcessingData.
      2. Read and validate Number_of_ships.
      3. If Number_of_ships > 0, check each ship's BoundingBox.
      4. Accumulate and return all issues found.

    Args:
        xml_root: Root element of the parsed XML document.

    Returns:
        Dictionary with 'status' and 'details' about missing or invalid data.
    """
    issues = []

    # Find ProcessingData node
    processing_data = xml_root.find('ProcessingData')
    if processing_data is None:
        issues.append("Missing: ProcessingData")
        return {'status': 'FAIL', 'details': {'missing': issues}}

    # Get and validate Number_of_ships
    num_ships_text = processing_data.findtext('StatisticsReport/Number_of_ships')
    try:
        num_ships = int(num_ships_text)
    except (ValueError, TypeError):
        issues.append(f"Invalid or missing Number_of_ships: '{num_ships_text}'")
        return {'status': 'FAIL', 'details': {'missing': issues}}

    # If no ships, no bounding box validation needed
    if num_ships == 0:
        return {'status': 'PASS', 'details': {}}

    # Get the list of ships
    ships = processing_data.findall('List_of_ships/Ship')
    if not ships:
        issues.append("Missing: List_of_ships/Ship")
        return {'status': 'FAIL', 'details': {'missing': issues}}

    # Validate each ship's BoundingBox
    for i, ship in enumerate(ships, start=1):
        bbox = ship.find('BoundingBox')
        if bbox is None:
            issues.append(f"Ship {i}: Missing BoundingBox")
            continue

        # Collect bounding box validation issues for the current ship
        bbox_issues = check_bounding_box(bbox, i)
        issues.extend(bbox_issues)

    status = 'PASS' if not issues else 'FAIL'
    return {'status': status, 'details': {'missing': issues}}
