DESCRIPTION = """
Test Case: Corner Coordinate Validation for RFI

Purpose:
  Validate the presence and correctness of required fields in the 'ProcessingData/Corner_Coord' section
  of the given XML data. Ensures fields follow specified formats and value ranges.

Inputs:
  - XML file for RFI.
  - The test will extract and validate fields within the 'ProcessingData/Corner_Coord' child element.
  - Ranges and coditions that need to be checked for each fields within the 'ProcessingData/Corner_Coord':
   * Latitude           : [-90.0000 .. 90.0000], series of 4 values
   * Longitude          : [-180.0000 .. 180.0000], series of 4 values
   * SARData_Sample     : [0 .. 50000], series of 4 values
   * SARData_Line       : [0 .. 20000], series of 4 values
   * Scene_Sample       : [0 .. 2000], series of 4 values, The first two values must be equal, The last two values must be equal
   * Scene_Line         : [0 .. 2000], series of 4 values, The first and last values must be equal, The middle two values must be equal

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
    'ProcessingData/Corner_Coord',
    'ProcessingData/Corner_Coord/Latitude',
    'ProcessingData/Corner_Coord/Longitude',
    'ProcessingData/Corner_Coord/SARData_Sample',
    'ProcessingData/Corner_Coord/SARData_Line',
    'ProcessingData/Corner_Coord/Scene_Sample',
    'ProcessingData/Corner_Coord/Scene_Line'
]

# RANGES dictionary defines expected count and value ranges for each element
RANGES = {
    'ProcessingData/Corner_Coord/Latitude': (4, -90.0000, 90.0000),             #range:[-90.0000 .. 90.0000], series of 4 values
    'ProcessingData/Corner_Coord/Longitude': (4, -180.0000, 180.0000),          #range:[-180.0000 .. 180.0000], series of 4 values
    'ProcessingData/Corner_Coord/SARData_Sample': (4, 0, 50000),                #range:[0 .. 50000], series of 4 values
    'ProcessingData/Corner_Coord/SARData_Line': (4, 0, 20000),                  #range:[0 .. 20000], series of 4 values
    'ProcessingData/Corner_Coord/Scene_Sample': (4, 0, 2000),                   #range:[0 .. 2000], series of 4 values, The first two values must be equal, The last two values must be equal
    'ProcessingData/Corner_Coord/Scene_Line': (4, 0, 2000),                     #range:[0 .. 2000], series of 4 values, The first and last values must be equal, The middle two values must be equal
}

def validate_latitude(text, errors, count, min_val, max_val):
    """
    Validate Latitude element text:
      - Must have 'count' float values
      - Each value must be in [min_val, max_val]
    Append error messages to errors list if invalid.
    """
    if text is None:
        errors.append("Missing Latitude element text")
        return
    values = text.strip().split()
    if len(values) != count:
        errors.append(f"Latitude: Expected {count} values, got {len(values)}")
        return
    try:
        floats = [float(v) for v in values]
    except:
        errors.append("Latitude: Invalid float format")
        return
    if any(not (min_val <= v <= max_val) for v in floats):
        errors.append(f"Latitude: Values out of range [{min_val}..{max_val}]")

def validate_longitude(text, errors, count, min_val, max_val):
    """
    Validate Longitude element text:
      - Must have 'count' float values
      - Each value must be in [min_val, max_val]
    Append error messages to errors list if invalid.
    """
    if text is None:
        errors.append("Missing Longitude element text")
        return
    values = text.strip().split()
    if len(values) != count:
        errors.append(f"Longitude: Expected {count} values, got {len(values)}")
        return
    try:
        floats = [float(v) for v in values]
    except:
        errors.append("Longitude: Invalid float format")
        return
    if any(not (min_val <= v <= max_val) for v in floats):
        errors.append(f"Longitude: Values out of range [{min_val}..{max_val}]")

def validate_sardata_sample(text, errors, count, min_val, max_val):
    """
    Validate SARData_Sample element text:
      - Must have 'count' integer values
      - Each value must be in [min_val, max_val]
    Append error messages to errors list if invalid.
    """
    if text is None:
        errors.append("Missing SARData_Sample element text")
        return
    values = text.strip().split()
    if len(values) != count:
        errors.append(f"SARData_Sample: Expected {count} values, got {len(values)}")
        return
    try:
        ints = [int(v) for v in values]
    except:
        errors.append("SARData_Sample: Invalid integer format")
        return
    if any(not (min_val <= v <= max_val) for v in ints):
        errors.append(f"SARData_Sample: Values out of range [{min_val}..{max_val}]")

def validate_sardata_line(text, errors, count, min_val, max_val):
    """
    Validate SARData_Line element text:
      - Must have 'count' integer values
      - Each value must be in [min_val, max_val]
    Append error messages to errors list if invalid.
    """
    if text is None:
        errors.append("Missing SARData_Line element text")
        return
    values = text.strip().split()
    if len(values) != count:
        errors.append(f"SARData_Line: Expected {count} values, got {len(values)}")
        return
    try:
        ints = [int(v) for v in values]
    except:
        errors.append("SARData_Line: Invalid integer format")
        return
    if any(not (min_val <= v <= max_val) for v in ints):
        errors.append(f"SARData_Line: Values out of range [{min_val}..{max_val}]")

def validate_scene_sample(text, errors, count, min_val, max_val):
    """
    Validate Scene_Sample element text:
      - Must have 'count' integer values
      - Each value must be in [min_val, max_val]
      - The first two values must be equal
      - The last two values must be equal
    Append error messages to errors list if invalid.
    """
    if text is None:
        errors.append("Missing Scene_Sample element text")
        return
    values = text.strip().split()
    if len(values) != count:
        errors.append(f"Scene_Sample: Expected {count} values, got {len(values)}")
        return
    try:
        ints = [int(v) for v in values]
    except:
        errors.append("Scene_Sample: Invalid integer format")
        return
    if any(not (min_val <= v <= max_val) for v in ints):
        errors.append(f"Scene_Sample: Values out of range [{min_val}..{max_val}]")
    if ints[0] != ints[1] or ints[2] != ints[3]:
        errors.append("Scene_Sample: first two numbers must be equal, last two must be equal")

def validate_scene_line(text, errors, count, min_val, max_val):
    """
    Validate Scene_Line element text:
      - Must have 'count' integer values
      - Each value must be in [min_val, max_val]
      - The first and last values must be equal
      - The middle two values must be equal
    Append error messages to errors list if invalid.
    """
    if text is None:
        errors.append("Missing Scene_Line element text")
        return
    values = text.strip().split()
    if len(values) != count:
        errors.append(f"Scene_Line: Expected {count} values, got {len(values)}")
        return
    try:
        ints = [int(v) for v in values]
    except:
        errors.append("Scene_Line: Invalid integer format")
        return
    if any(not (min_val <= v <= max_val) for v in ints):
        errors.append(f"Scene_Line: Values out of range [{min_val}..{max_val}]")
    if ints[0] != ints[3] or ints[1] != ints[2]:
        errors.append("Scene_Line: first and last must be equal, middle two must be equal")

def run_test(xml_root):
    """
    Run the overall validation test:
      - Check required elements existence
      - Call individual validation functions with proper parameters
      - Collect error messages and decide PASS/FAIL status
    Returns a dictionary with 'status' and 'details'.
    """
    missing_or_invalid = []

    # Check if all required elements exist in the XML
    for path in REQUIRED:
        if xml_root.find(path) is None:
            missing_or_invalid.append(f"Missing required element: {path}")

    # If any required elements missing, fail early
    if missing_or_invalid:
        return {'status': 'FAIL', 'details': {'missing': missing_or_invalid}}

    # Validate ProcessingData/Corner_Coord/Latitude
    latitude = xml_root.findtext('ProcessingData/Corner_Coord/Latitude')
    validate_latitude(latitude, missing_or_invalid, *RANGES['ProcessingData/Corner_Coord/Latitude'])

    # Validate ProcessingData/Corner_Coord/Longitude
    longitude = xml_root.findtext('ProcessingData/Corner_Coord/Longitude')
    validate_longitude(longitude, missing_or_invalid, *RANGES['ProcessingData/Corner_Coord/Longitude'])

    # Validate ProcessingData/Corner_Coord/SARData_Sample
    sar_sample = xml_root.findtext('ProcessingData/Corner_Coord/SARData_Sample')
    validate_sardata_sample(sar_sample, missing_or_invalid, *RANGES['ProcessingData/Corner_Coord/SARData_Sample'])

    # Validate ProcessingData/Corner_Coord/SARData_Line
    sar_line = xml_root.findtext('ProcessingData/Corner_Coord/SARData_Line')
    validate_sardata_line(sar_line, missing_or_invalid, *RANGES['ProcessingData/Corner_Coord/SARData_Line'])

    # Validate ProcessingData/Corner_Coord/Scene_Sample
    scene_sample = xml_root.findtext('ProcessingData/Corner_Coord/Scene_Sample')
    validate_scene_sample(scene_sample, missing_or_invalid, *RANGES['ProcessingData/Corner_Coord/Scene_Sample'])

    # Validate ProcessingData/Corner_Coord/Scene_Line
    scene_line = xml_root.findtext('ProcessingData/Corner_Coord/Scene_Line')
    validate_scene_line(scene_line, missing_or_invalid, *RANGES['ProcessingData/Corner_Coord/Scene_Line'])

    status = 'PASS' if not missing_or_invalid else 'FAIL'
    return {
        'status': status,
        'details': {'missing': missing_or_invalid}
    }
