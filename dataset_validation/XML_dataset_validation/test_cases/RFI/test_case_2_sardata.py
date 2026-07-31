import re
from datetime import datetime

DESCRIPTION = """
Test Case: SARData Validation for RFI

Purpose:
  Validate the presence and correctness of required fields in the 'SARData' section
  of the given XML data file. Ensures fields follow specified formats and value ranges.

Inputs:
  - XML file for RFI.
  - The test will extract and validate fields within the 'SARData' child element.
  - Ranges and coditions that need to be checked for each fields within the 'SARData':
   * SAR_Mission    : [S1A, S1B, S1C]
   * SARProduct     : 'Format: S1X_IW_SLC__1SDV_<start>T<start_time>_<stop>T<stop_time>_<6 digits>_<6 alphanum>_<4 alphanum>'
   * SLCSwath       : [1 .. 3]
   * Start          : [20000101_000000..20250501_000000], Start <= Stop
   * Stop           : [20000101_000000..20250501_000000], Stop >= Start

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
    'SARData',
    'SARData/SAR_Mission',
    'SARData/SARProduct',
    'SARData/SLCSwath'
]

RANGES = {
    'SARData/SAR_Mission': ('S1A', 'S1B', 'S1C'),               #range:[S1A, S1B, S1C]
    'SARData/SLCSwath': (1, 3),                                 #range:[1 .. 3]
    'SARData/Time_Interval/Start': 'range: [20000101_000000..20250501_000000], Start <= Stop',
    'SARData/Time_Interval/Stop': 'range [20000101_000000..20250501_000000], Stop >= Start',
    'SARData/SARProduct': 'Format: S1X_IW_SLC__1SDV_<start>T<start_time>_<stop>T<stop_time>_<6 digits>_<6 alphanum>_<4 alphanum>',
}

def valid_datetime(dt_str):
    """Validate and parse datetime string in format YYYYMMDD_HHMMSS."""
    try:
        return datetime.strptime(dt_str, '%Y%m%d_%H%M%S')
    except Exception:
        return None

def validate_sar_mission(value, missing_or_invalid):
    """Validate SAR_Mission field against allowed values."""
    if value is None:
        missing_or_invalid.append('SARData/SAR_Mission')
    elif value not in RANGES['SARData/SAR_Mission']:
        missing_or_invalid.append(
            f"SARData/SAR_Mission value '{value}' is invalid (must be 'S1A', 'S1B' or 'S1C')"
        )

def validate_time_interval(time_interval, missing_or_invalid):
    """Validate Time_Interval Start and Stop datetime fields."""
    TIME_INTERVAL_MIN = datetime(2000, 1, 1, 0, 0, 0)   #(YYYY,MM,DD,HH,MM,SS)
    TIME_INTERVAL_MAX = datetime(2025, 5, 1, 0, 0, 0)   #(YYYY,MM,DD,HH,MM,SS)

    if time_interval is None:
        missing_or_invalid.append('SARData/Time_Interval')
        return None, None

    start = time_interval.findtext('Start')
    stop = time_interval.findtext('Stop')

    if not start:
        missing_or_invalid.append('SARData/Time_Interval/Start')
        start_dt = None
    else:
        start_dt = valid_datetime(start)
        if (not start_dt or
            start_dt < TIME_INTERVAL_MIN or
            start_dt > TIME_INTERVAL_MAX):
            missing_or_invalid.append(f"Start '{start}' out of allowed range or invalid format")

    if not stop:
        missing_or_invalid.append('SARData/Time_Interval/Stop')
        stop_dt = None
    else:
        stop_dt = valid_datetime(stop)
        if (not stop_dt or
            stop_dt < TIME_INTERVAL_MIN or
            stop_dt > TIME_INTERVAL_MAX):
            missing_or_invalid.append(f"Stop '{stop}' out of allowed range or invalid format")

    if start_dt and stop_dt and start_dt > stop_dt:
        missing_or_invalid.append(f"Start time '{start}' is after Stop time '{stop}'")

    return start, stop

def validate_sar_product(value, missing_or_invalid, sar_mission, start, stop):
    """Validate SARProduct field format and prefix."""
    if value is None:
        missing_or_invalid.append('SARData/SARProduct')
        return

    if not (sar_mission and start and stop):
        missing_or_invalid.append('SARProduct could not be validated due to missing Start, Stop or SAR_Mission')
        return

    start_fmt = start.replace('_', 'T')
    stop_fmt = stop.replace('_', 'T')
    expected_prefix = f"{sar_mission}_IW_SLC__1SDV_{start_fmt}_{stop_fmt}_"
    if not value.startswith(expected_prefix):
        missing_or_invalid.append(
            f"SARData/SARProduct value '{value}' does not start with expected prefix '{expected_prefix}'"
        )
        return

    suffix = value[len(expected_prefix):]
    pattern = re.compile(r'^\d{6}_[a-zA-Z0-9]{6}_[a-zA-Z0-9]{4}(\.SAFE)?$')
    if not pattern.match(suffix):
        missing_or_invalid.append(f"SARData/SARProduct suffix '{suffix}' does not match expected pattern")

def validate_slc_swath(value, missing_or_invalid):
    """Validate SLCSwath integer in range [1..3]."""
    if value is None:
        missing_or_invalid.append('SARData/SLCSwath')
        return

    try:
        swath = int(value)
        min_val, max_val = RANGES['SARData/SLCSwath']
        if swath < min_val or swath > max_val:
            missing_or_invalid.append(f"SARData/SLCSwath value '{swath}' out of range [{min_val}..{max_val}]")
    except ValueError:
        missing_or_invalid.append(f"SARData/SLCSwath value '{value}' is not an integer")

def run_test(xml_root):
    missing_or_invalid = []

    sar_data = xml_root.find('SARData')
    if sar_data is None:
        missing_or_invalid.append('SARData')
        return {
            'status': 'FAIL',
            'details': {'missing': missing_or_invalid}
        }

    # Validate SAR_Mission
    sar_mission = sar_data.findtext('SAR_Mission')
    validate_sar_mission(sar_mission, missing_or_invalid)

    # Validate Time_Interval
    time_interval = sar_data.find('Time_Interval')
    start, stop = validate_time_interval(time_interval, missing_or_invalid)

    # Validate SARProduct
    sar_product = sar_data.findtext('SARProduct')
    validate_sar_product(sar_product, missing_or_invalid, sar_mission, start, stop)

    # Validate SLCSwath
    slc_swath = sar_data.findtext('SLCSwath')
    validate_slc_swath(slc_swath, missing_or_invalid)

    status = 'PASS' if not missing_or_invalid else 'FAIL'

    return {
        'status': status,
        'details': {'missing': missing_or_invalid}
    }
