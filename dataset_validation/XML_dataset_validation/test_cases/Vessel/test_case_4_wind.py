DESCRIPTION = """
Test Case: Wind Data Validation for Dark Vessel Detection

Purpose:
  Validate the presence and correctness of required fields in the 'ProcessingData/WindData' section
  of the given XML data. Ensures fields follow specified formats and value ranges.

Inputs:
  - XML file for Dark Vessel Detection.
  - The test will extract and validate fields within the 'ProcessingData/WindData' child element.
  - Ranges and coditions that need to be checked for each fields within the 'ProcessingData/WindData':
   * Direction      : [0 .. 360]
   * Speed          : [0 .. infinite]

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
    'ProcessingData/WindData/Direction',
    'ProcessingData/WindData/Speed',
]

RANGES = {
    'ProcessingData/WindData/Direction': '[0 .. 360] (float)',
    'ProcessingData/WindData/Speed': '[0 .. ∞] (float)',
}

def check_float_field_ignore_nan(parent, field_name, min_val, max_val, issues):
    """
    Check if a float field exists, is valid, and within range.
    Ignores fields with 'NaN' (case-insensitive) values.

    Args:
        parent: XML element containing the field.
        field_name: Name of the child field.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value (can be None for no upper bound).
        issues: Dictionary with keys 'missing' and 'warnings' to append issues.

    Returns:
        True if field exists and is valid or NaN,
        False if missing or invalid (missing issues are appended).
    """
    val_str = parent.findtext(field_name)
    if val_str is None:
        issues['missing'].append(f"Missing: {field_name}")
        return False

    val_str = val_str.strip()
    if val_str.lower() == 'nan':
        # NaN values are ignored (neither fail nor warning)
        return True

    try:
        val = float(val_str)
    except ValueError:
        issues['missing'].append(f"Invalid float: {field_name}='{val_str}'")
        return False

    if val < min_val or (max_val is not None and val > max_val):
        issues['warnings'].append(f"{field_name} value {val} out of range [{min_val}..{max_val if max_val is not None else '∞'}]")
    return True

def run_test(xml_root):
    """
    Main test function to validate WindData Direction and Speed fields.

    Args:
        xml_root: Root XML element.

    Returns:
        Dictionary with 'status' and 'details' about missing/warning issues.
    """
    issues = {
        'missing': [],
        'warnings': []
    }

    processing_data = xml_root.find('ProcessingData')
    if processing_data is None:
        issues['missing'].append("Missing: ProcessingData")
        return {'status': 'FAIL', 'details': issues}

    wind_data = processing_data.find('WindData')
    if wind_data is None:
        issues['missing'].append("Missing: WindData")
        return {'status': 'FAIL', 'details': issues}

    # Validate Direction
    if not check_float_field_ignore_nan(wind_data, 'Direction', 0.0, 360.0, issues):
        return {'status': 'FAIL', 'details': issues}

    # Validate Speed
    # Note: Speed has no upper bound; pass max_val=None
    if not check_float_field_ignore_nan(wind_data, 'Speed', 0.0, None, issues):
        return {'status': 'FAIL', 'details': issues}

    # Determine overall status
    status = 'PASS'
    if issues['missing']:
        status = 'FAIL'
    elif issues['warnings']:
        status = 'WARNING'

    return {'status': status, 'details': issues}
