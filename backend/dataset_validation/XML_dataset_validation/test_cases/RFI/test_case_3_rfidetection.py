DESCRIPTION = """
Test Case: Scene_Info Validation for RFI

Purpose:
  Validate the presence and correctness of required fields in the 'ProcessingData' section
  of the given XML data file. Ensures fields follow specified formats and value ranges.

Inputs:
  - XML file for RFI.
  - The test will extract and validate fields within the 'ProcessingData' child element.
  - Ranges and coditions that need to be checked for each fields within the 'StatisticsReport':
   * RFIDetection   : [0 .. 1]

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
    'ProcessingData/StatisticsReport/RFIDetection'
]

RANGES = {
    'ProcessingData/StatisticsReport/RFIDetection': (0, 1)      #range:[0 .. 1]
}

def validate_rfidetection(value, missing_or_invalid, min_val, max_val):
    """
    Validate RFIDetection is an integer within [min_val..max_val].
    """
    if value is None:
        missing_or_invalid.append('ProcessingData/StatisticsReport/RFIDetection')
        return

    try:
        int_val = int(value)
        if not (min_val <= int_val <= max_val):
            missing_or_invalid.append(
                f"ProcessingData/StatisticsReport/RFIDetection value '{int_val}' not in range [{min_val}..{max_val}]"
            )
    except ValueError:
        missing_or_invalid.append(
            f"ProcessingData/StatisticsReport/RFIDetection value '{value}' is not a valid integer"
        )

def run_test(xml_root):
    """
    Main test function to validate RFIDetection presence and value.
    """
    missing_or_invalid = []

    # Check required elements
    for path in REQUIRED:
        if xml_root.find(path) is None:
            missing_or_invalid.append(path)

    # Validate ProcessingData/StatisticsReport/RFIDetection
    rfi_detection = xml_root.findtext('ProcessingData/StatisticsReport/RFIDetection')
    validate_rfidetection(rfi_detection, missing_or_invalid, *RANGES['ProcessingData/StatisticsReport/RFIDetection'])

    status = 'PASS' if not missing_or_invalid else 'FAIL'

    return {
        'status': status,
        'details': {'missing': missing_or_invalid}
    }
