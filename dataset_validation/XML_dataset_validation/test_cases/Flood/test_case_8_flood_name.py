DESCRIPTION = """
Test Case: Flood Name Validation for Flood

Purpose:
  Validate the presence and correctness of required fields in the 'ProcessingData/List_of_ships/Ship/Name' section
  of the given XML data. Ensures fields follow specified formats and value ranges.

Inputs:
  - XML file for Flood.
  - The test will extract and validate fields within the 'ProcessingData/List_of_FloodEvents/Event' child element.
  - Ranges and coditions that need to be checked for each fields within the 'List_of_FloodEvents/Event/Name':
   * Name      : [FloodEvent_1 .. FloodEvent_infinite], The name do not repeat

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

import re

REQUIRED = [
    'ProcessingData',
    'ProcessingData/StatisticsReport',
    'ProcessingData/StatisticsReport/Number_of_FloodEvents',
]

def get_text(xml_root, path):
    element = xml_root.find(path)
    return element.text.strip() if element is not None and element.text else None

def validate_number_of_flood_events(text, errors):
    """
    Validate that Number_of_FloodEvents exists and is a valid non-negative integer.
    """
    if text is None:
        errors.append("Missing Number_of_FloodEvents value")
        return None
    try:
        val = int(text)
        if val < 0:
            errors.append("Number_of_FloodEvents must be >= 0")
        return val
    except:
        errors.append("Number_of_FloodEvents must be an integer")
        return None

def validate_event_names(xml_root, errors):
    """
    Validate that 'List_of_FloodEvents/Event/Name' exists if Number_of_FloodEvents > 0,
    and all names are:
      - Unique
      - Match pattern 'FloodEvent_<number>'
    """
    event_elements = xml_root.findall('ProcessingData/List_of_FloodEvents/Event')
    if not event_elements:
        errors.append("Missing ProcessingData/List_of_FloodEvents/Event elements")
        return

    seen_names = set()
    pattern = re.compile(r'^FloodEvent_\d+$')
    for idx, event in enumerate(event_elements):
        name_elem = event.find('Name')
        if name_elem is None or not name_elem.text.strip():
            errors.append(f"Missing Name in Event #{idx + 1}")
            continue
        name = name_elem.text.strip()

        if name in seen_names:
            errors.append(f"Duplicate Event Name found: {name}")
        else:
            seen_names.add(name)

        if not pattern.match(name):
            errors.append(f"Invalid Event Name format: '{name}' (expected 'FloodEvent_<number>')")

def run_test(xml_root):
    """
    Run the overall validation:
      - Check required fields
      - Validate Number_of_FloodEvents
      - If >0, check for List_of_FloodEvents/Event/Name existence and correctness
    Returns a dictionary with 'status' and 'details'.
    """
    missing_or_invalid = []

    # Check all required paths exist
    for path in REQUIRED:
        if xml_root.find(path) is None:
            missing_or_invalid.append(f"Missing required element: {path}")

    if missing_or_invalid:
        return {'status': 'FAIL', 'details': {'missing_or_invalid': missing_or_invalid}}

    # Validate Number_of_FloodEvents value
    number_text = get_text(xml_root, 'ProcessingData/StatisticsReport/Number_of_FloodEvents')
    count = validate_number_of_flood_events(number_text, missing_or_invalid)

    # If count > 0, validate event names
    if count is not None and count > 0:
        validate_event_names(xml_root, missing_or_invalid)

    status = 'PASS' if not missing_or_invalid else 'FAIL'
    return {
        'status': status,
        'details': {'missing_or_invalid': missing_or_invalid}
    }
