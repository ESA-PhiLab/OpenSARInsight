DESCRIPTION = """
Test Case: Number_of_FloodEvents Validation for Flood

Purpose:
  Validate the presence and correctness of required fields in the 'ProcessingData' section
  of the given XML data. Ensures fields follow specified formats and value ranges.

Inputs:
  - XML file for Flood.
  - The test will extract and validate fields within the 'ProcessingData' child element.
  - Ranges and coditions that need to be checked for each fields within the 'ProcessingData/StatisticsReport':
   * Number_of_FloodEvents      : [0 .. infinite], Matches the number of <Ship> elements in ProcessingData/List_of_FloodEvents/Event

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
    'ProcessingData/StatisticsReport/Number_of_FloodEvents'
]

RANGES = {
    'ProcessingData/StatisticsReport/Number_of_FloodEvents': (0, float('inf'))  # [0 .. ∞]
}

def validate_number_of_flood_events(xml_root, errors):
    """Validate the Number_of_FloodEvents field and check it matches actual Events count."""
    processing_data = xml_root.find('ProcessingData')
    if processing_data is None:
        errors.append('Missing required field: ProcessingData')
        return

    stats_report = processing_data.find('StatisticsReport')
    if stats_report is None:
        errors.append('Missing required field: ProcessingData/StatisticsReport')
        return

    number_elem = xml_root.find('ProcessingData/StatisticsReport/Number_of_FloodEvents')
    if number_elem is None:
        errors.append('Missing required field: ProcessingData/StatisticsReport/Number_of_FloodEvents')
        return

    try:
        number_value = int(number_elem.text)
        if number_value < 0:
            errors.append(f"Number_of_FloodEvents value '{number_value}' is less than 0 (must be >= 0)")
            return
    except (ValueError, TypeError):
        errors.append("Number_of_FloodEvents value is not a valid integer")
        return

    if number_value > 0:
        flood_list = processing_data.find('List_of_FloodEvents')
        if flood_list is None:
            errors.append("List_of_FloodEvents is missing while Number_of_FloodEvents > 0")
            return

        event_count = len(flood_list.findall('Event'))
        if event_count != number_value:
            errors.append(
                f"Number_of_FloodEvents is '{number_value}', but found {event_count} <Event> elements in List_of_FloodEvents"
            )

def run_test(xml_root):
    """Main test runner for flood event statistics validation."""
    missing_or_invalid = []
    validate_number_of_flood_events(xml_root, missing_or_invalid)

    status = 'PASS' if not missing_or_invalid else 'FAIL'
    return {
        'status': status,
        'details': {'missing_or_invalid': missing_or_invalid}
    }
