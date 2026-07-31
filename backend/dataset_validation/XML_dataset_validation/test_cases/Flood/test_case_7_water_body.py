DESCRIPTION = """
Test Case: Water Body Validation for Flood

Purpose:
  Validate the presence and correctness of required fields in the 'ProcessingData' section
  of the given XML data. Ensures fields follow specified formats and value ranges.

Inputs:
  - XML file for Flood.
  - The test will extract and validate fields within the 'ProcessingData/List_of_WaterBodys/Body' child element.
  - Ranges and coditions that need to be checked for each fields within the 'ProcessingData/List_of_WaterBodys/Body/Polygon':
    * Latitude:         [-90.00 .. 90.00], Should contain an equal number of coordinates compared to Longitude
    * Longitude:        [-180.00 .. 180.00], Should contain an equal number of coordinates compared to Latitude
    * SARData_Sample:   [0 .. 50000], Should contain an equal number of coordinates compared to SARData_Line
    * SARData_Line:     [0 .. 20000], Should contain an equal number of coordinates compared to SARData_Sample
    * Scene_Sample:     [0 .. 512], Should contain an equal number of coordinates compared to Scene_Line
    * Scene_Line:       [0 .. 512], Should contain an equal number of coordinates compared to Scene_Sample 

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

POLYGON_FIELDS = {
    'Latitude': (-90.0, 90.0),
    'Longitude': (-180.0, 180.0),
    'SARData_Sample': (0.0, 50000.0),
    'SARData_Line': (0.0, 20000.0),
    'Scene_Sample': (0.0, 512.0),
    'Scene_Line': (0.0, 512.0)
}

def parse_float_list_with_nan(text, tag_name, min_val, max_val, errors):
    raw_values = text.strip().split()
    parsed = []

    for v in raw_values:
        if v.strip().lower() == 'nan':
            parsed.append(float('nan'))
            continue
        try:
            fval = float(v)
        except:
            errors.append(f"{tag_name} contains invalid float format: '{v}'")
            continue
        if not (min_val <= fval <= max_val):
            errors.append(f"{tag_name} value {fval} out of range [{min_val}..{max_val}].")
        parsed.append(fval)
    return parsed

def validate_polygon(polygon_elem, errors, body_index):
    data = {}
    lengths = {}
    for tag, (min_v, max_v) in POLYGON_FIELDS.items():
        el = polygon_elem.find(tag)
        if el is None or el.text is None:
            errors.append(f"Missing {tag} in Polygon for Body #{body_index}.")
            continue
        values = parse_float_list_with_nan(el.text, tag, min_v, max_v, errors)
        data[tag] = values
        lengths[tag] = len(values)

    coord_pairs = [
        ('Latitude', 'Longitude'),
        ('SARData_Sample', 'SARData_Line'),
        ('Scene_Sample', 'Scene_Line')
    ]
    for a, b in coord_pairs:
        if a in lengths and b in lengths and lengths[a] != lengths[b]:
            errors.append(f"{a} and {b} have different lengths for Body #{body_index}: {lengths[a]} vs {lengths[b]}.")

def run_test(xml_root):
    missing_or_invalid = []

    for path in REQUIRED:
        if xml_root.find(path) is None:
            missing_or_invalid.append(f"Missing required element: {path}")

    if missing_or_invalid:
        return {'status': 'FAIL', 'details': {'missing': missing_or_invalid}}

    number_text = xml_root.findtext('ProcessingData/StatisticsReport/Number_of_WaterBodys')
    try:
        num_bodies = int(number_text.strip())
    except:
        return {'status': 'FAIL', 'details': {'missing': ['Number_of_WaterBodys is not a valid integer.']}}

    if num_bodies == 0:
        return {'status': 'PASS', 'details': {'note': 'No water bodies to validate.'}}

    bodies = xml_root.findall('ProcessingData/List_of_WaterBodys/Body')
    if not bodies:
        return {'status': 'FAIL', 'details': {'missing': ['Expected Body elements under ProcessingData/List_of_WaterBodys but none found.']}}

    for idx, body in enumerate(bodies, 1):
        polygon = body.find('Polygon')
        if polygon is None:
            missing_or_invalid.append(f"Missing Polygon in Body #{idx}.")
            continue
        validate_polygon(polygon, missing_or_invalid, idx)

    status = 'PASS' if not missing_or_invalid else 'FAIL'
    return {
        'status': status,
        'details': {'missing': missing_or_invalid}
    }
