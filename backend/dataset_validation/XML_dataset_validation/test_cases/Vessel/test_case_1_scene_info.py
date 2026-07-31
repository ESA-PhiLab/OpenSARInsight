import re

DESCRIPTION = """
Test Case: Scene_Info Validation for Dark Vessel Detection

Purpose:
  Validate the presence and correctness of required fields in the 'Scene_Info' section
  of the given XML data file. Ensures fields follow specified formats and value ranges.

Inputs:
  - XML file for Dark Vessel Detection.
  - The test will extract and validate fields within the 'Scene_Info' child element.
  - Ranges and coditions that need to be checked for each fields within the 'Scene_Info':
   * Scene_ID   : [DB_OPENSAR_DVD_0001 .. DB_OPENSAR_DVD_9999]
   * Date       : [20000101 .. 20251231]
   * Version    : [0 .. infinite]
   * CaseStudy  : ['DarkVesselDetection']

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
    'Scene_Info', 
    'Scene_Info/Scene_ID', 
    'Scene_Info/Date', 
    'Scene_Info/Version', 
    'Scene_Info/CaseStudy'
]

RANGES = {
    'Scene_Info/Scene_ID': (r'^DB_OPENSAR_DVD_(\d{1,4})$', 1, 9999),    #range:[DB_OPENSAR_DVD_0001 .. DB_OPENSAR_DVD_9999]
    'Scene_Info/Date': (20000101, 20251231),                            #range:[20000101 .. 20251231]
    'Scene_Info/Version': (0, None),                                    #range:[0 .. infinite]
    'Scene_Info/CaseStudy': 'DarkVesselDetection',                      #range:[DarkVesselDetection]
}

def validate_scene_id(scene_id, missing_or_invalid, regex, num_min, num_max):
    """
    Validate Scene_ID format and numeric range based on given parameters.
    """
    if scene_id is None:
        missing_or_invalid.append('Scene_Info/Scene_ID')
        return None

    match = re.match(regex, scene_id)
    if not match:
        missing_or_invalid.append(f"Scene_Info/Scene_ID value '{scene_id}' invalid format")
        return None

    number = int(match.group(1))
    if not (num_min <= number <= num_max):
        missing_or_invalid.append(f"Scene_Info/Scene_ID value '{scene_id}' out of valid range [{num_min}..{num_max}]")

    return scene_id

def validate_date(date_str, missing_or_invalid, date_min, date_max):
    """
    Validate 'Date' is an integer within specified range.
    """
    if date_str is None:
        missing_or_invalid.append('Scene_Info/Date')
        return

    try:
        date_val = int(date_str)
        if not (date_min <= date_val <= date_max):
            missing_or_invalid.append(f"Scene_Info/Date value '{date_str}' out of range [{date_min}..{date_max}]")
    except ValueError:
        missing_or_invalid.append(f"Scene_Info/Date value '{date_str}' is not a valid integer date")

def validate_version(version_str, missing_or_invalid, min_version):
    """
    Validate 'Version' is an integer >= min_version.
    """
    if version_str is None:
        missing_or_invalid.append('Scene_Info/Version')
        return

    try:
        version = int(version_str)
        if version < min_version:
            missing_or_invalid.append(f"Scene_Info/Version value '{version_str}' < {min_version}")
    except ValueError:
        missing_or_invalid.append(f"Scene_Info/Version value '{version_str}' is not an integer")

def validate_case_study(case_study, missing_or_invalid, expected_value):
    """
    Validate 'CaseStudy' matches expected string exactly.
    """
    if case_study is None:
        missing_or_invalid.append('Scene_Info/CaseStudy')
    elif case_study != expected_value:
        missing_or_invalid.append(f"Scene_Info/CaseStudy value '{case_study}' is not '{expected_value}'")

def run_test(xml_root):
    """
    Main function to run all validation checks on 'Scene_Info'.
    Returns a dict with overall status and details of missing/invalid fields.
    """
    missing_or_invalid = []

    scene_info = xml_root.find('Scene_Info')
    if scene_info is None:
        return {
            'status': 'FAIL',
            'details': {'missing': ['Scene_Info']}
        }

    # Validate Scene_Info/Scene_ID
    scene_id = scene_info.findtext('Scene_ID')
    validate_scene_id(scene_id, missing_or_invalid, *RANGES['Scene_Info/Scene_ID'])

    # Validate Scene_Info/Date
    date = scene_info.findtext('Date')
    validate_date(date, missing_or_invalid, *RANGES['Scene_Info/Date'])

    # Validate Scene_Info/Version
    version = scene_info.findtext('Version')
    validate_version(version, missing_or_invalid, RANGES['Scene_Info/Version'][0])

    # Validate Scene_Info/CaseStudy
    case_study = scene_info.findtext('CaseStudy')
    validate_case_study(case_study, missing_or_invalid, RANGES['Scene_Info/CaseStudy'])

    status = 'PASS' if not missing_or_invalid else 'FAIL'

    return {
        'status': status,
        'details': {'missing': missing_or_invalid}
    }
