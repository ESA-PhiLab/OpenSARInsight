DESCRIPTION = """
Test Case: Ship Name Validation for Dark Vessel Detection

Purpose:
  Validate the presence and correctness of required fields in the 'ProcessingData/List_of_ships/Ship/Name' section
  of the given XML data. Ensures fields follow specified formats and value ranges.

Inputs:
  - XML file for Dark Vessel Detection.
  - The test will extract and validate fields within the 'ProcessingData/List_of_ships/Ship/Name' child element.
  - Ranges and coditions that need to be checked for each fields within the 'ProcessingData/List_of_ships/Ship/Name':
   * Name      : [Ship_1 .. Ship_100], The name do not repeat

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

def validate_list_of_ships(processing_data, number_of_ships, errors):
    """
    Validate the 'List_of_ships' element inside 'ProcessingData'.
    
    Checks:
      - Presence of 'List_of_ships' element.
      - The number of <Ship> elements matches 'Number_of_ships'.
      - Each <Ship> has a <Name> element.
      - Each <Name> matches the expected format 'Ship_i' where i is the index.
      - All ship names are unique.
    
    Args:
      processing_data: XML element representing 'ProcessingData'.
      number_of_ships: Integer representing the expected number of ships.
      errors: List to collect error messages.
    """
    # Find the List_of_ships element
    list_of_ships = processing_data.find('List_of_ships')
    if list_of_ships is None:
        errors.append('ProcessingData/List_of_ships missing')
        return

    # Find all Ship elements under List_of_ships
    ships = list_of_ships.findall('Ship')

    # Check if the number of Ship elements matches the expected number_of_ships
    if len(ships) != number_of_ships:
        errors.append(f"Number of <Ship> elements ({len(ships)}) does not match Number_of_ships ({number_of_ships})")
        return

    seen_names = set()
    for i, ship in enumerate(ships, start=1):
        # Extract the Name text of each Ship
        name = ship.findtext('Name')
        expected_name = f"Ship_{i}"

        # Check if Name exists
        if name is None:
            errors.append(f"ProcessingData/List_of_ships/Ship[{i}]/Name missing")
        else:
            # Check if the Name matches the expected pattern
            if name != expected_name:
                errors.append(f"ProcessingData/List_of_ships/Ship[{i}]/Name value '{name}' invalid, expected '{expected_name}'")

            # Check for duplicate ship names
            if name in seen_names:
                errors.append(f"ProcessingData/List_of_ships/Ship[{i}]/Name value '{name}' is duplicated")
            else:
                seen_names.add(name)


def run_test(xml_root):
    """
    Main test function to validate 'ProcessingData/List_of_ships' section against the 'Number_of_ships'.

    Args:
      xml_root: Root element of the parsed XML document.

    Returns:
      A dict with:
        - 'status': 'PASS' if validation passes, 'FAIL' otherwise.
        - 'details': dict containing 'missing' key with list of errors found.
    """
    missing_or_invalid = []

    # Check all required elements exist in the XML before further validation
    for path in REQUIRED:
        if xml_root.find(path) is None:
            missing_or_invalid.append(f"Missing required element: {path}")

    # Early return if required elements missing
    if missing_or_invalid:
        return {'status': 'FAIL', 'details': {'missing': missing_or_invalid}}

    # Extract the Number_of_ships and validate it
    number_of_ships_text = xml_root.findtext('ProcessingData/StatisticsReport/Number_of_ships')
    try:
        number_of_ships = int(number_of_ships_text)
        if number_of_ships < 0:
            missing_or_invalid.append(f"ProcessingData/StatisticsReport/Number_of_ships value '{number_of_ships}' < 0")
            return {'status': 'FAIL', 'details': {'missing': missing_or_invalid}}
    except ValueError:
        missing_or_invalid.append(f"ProcessingData/StatisticsReport/Number_of_ships value '{number_of_ships_text}' is not an integer")
        return {'status': 'FAIL', 'details': {'missing': missing_or_invalid}}

    # If number_of_ships > 0, perform detailed ships validation
    if number_of_ships > 0:
        processing_data = xml_root.find('ProcessingData')
        validate_list_of_ships(processing_data, number_of_ships, missing_or_invalid)

    # Final status is PASS if no errors found, otherwise FAIL
    status = 'PASS' if not missing_or_invalid else 'FAIL'

    return {
        'status': status,
        'details': {'missing': missing_or_invalid}
    }
