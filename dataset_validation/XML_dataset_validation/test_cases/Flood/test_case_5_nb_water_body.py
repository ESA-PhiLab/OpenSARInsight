DESCRIPTION = """
Test Case: Number_of_WaterBodys Validation for Flood

Purpose:
  Validate the presence and correctness of required fields in the 'ProcessingData' section
  of the given XML data. Ensures fields follow specified formats and value ranges.

Inputs:
  - XML file for Flood.
  - The test will extract and validate fields within the 'ProcessingData' child element.
  - Ranges and coditions that need to be checked for each fields within the 'ProcessingData/StatisticsReport':
   * Number_of_WaterBodys      : [0 .. infinite], Matches the number of <Ship> elements in ProcessingData/List_of_WaterBodys/Body

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
    'ProcessingData/StatisticsReport/Number_of_WaterBodys'
]

RANGES = {
    'ProcessingData/StatisticsReport/Number_of_WaterBodys': (0, float('inf'))  # [0 .. ∞]
}

def validate_number_of_water_bodys(xml_root, errors):
    """Validate the Number_of_WaterBodys field and check it matches actual Body count."""
    processing_data = xml_root.find('ProcessingData')
    if processing_data is None:
        errors.append('Missing required field: ProcessingData')
        return

    stats_report = processing_data.find('StatisticsReport')
    if stats_report is None:
        errors.append('Missing required field: ProcessingData/StatisticsReport')
        return

    number_elem = stats_report.find('Number_of_WaterBodys')
    if number_elem is None:
        errors.append('Missing required field: ProcessingData/StatisticsReport/Number_of_WaterBodys')
        return

    try:
        number_value = int(number_elem.text)
        if number_value < 0:
            errors.append(f"Number_of_WaterBodys value '{number_value}' is less than 0 (must be >= 0)")
            return
    except (ValueError, TypeError):
        errors.append("Number_of_WaterBodys value is not a valid integer")
        return

    if number_value > 0:
        body_list = processing_data.find('List_of_WaterBodys')
        if body_list is None:
            errors.append("List_of_WaterBodys is missing while Number_of_WaterBodys > 0")
            return

        body_count = len(body_list.findall('Body'))
        if body_count != number_value:
            errors.append(
                f"Number_of_WaterBodys is '{number_value}', but found {body_count} <Body> elements in List_of_WaterBodys"
            )

def run_test(xml_root):
    """Main test runner for water body statistics validation."""
    missing_or_invalid = []
    validate_number_of_water_bodys(xml_root, missing_or_invalid)

    status = 'PASS' if not missing_or_invalid else 'FAIL'
    return {
        'status': status,
        'details': {'missing_or_invalid': missing_or_invalid}
    }
