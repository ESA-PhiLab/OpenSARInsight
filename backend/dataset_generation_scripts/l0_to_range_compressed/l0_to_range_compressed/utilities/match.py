# type: ignore
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from l0_to_range_compressed.utilities.paths import (
    vessel_data_headers_dir,
    rfi_data_headers_dir,
    flood_data_headers_dir,
    splits
)

'''
Match L0 Patches to Headers and Calibration Files for Vessel Detection
'''

# Setup logging
log_dir = Path(__file__).parent
log_file = log_dir / f"matching_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def match_patch_to_safe(patch_name: str, safe_dirs: List[Path]) -> Optional[Path]:
    """
    Match a patch filename to its corresponding SAFE directory.
    
    Extracts timestamp, absolute orbit, and mission data take ID from the patch
    filename and finds the matching SAFE directory.
    
    Args:
        patch_name: Name of the L0 patch file (e.g., 's1a-iw-raw-s-vh-20200407t052956-20200407t053029-032017-03b2e2-IW1-VD_1778.dat')
        safe_dirs: List of SAFE directory paths to search through
        
    Returns:
        Path to the matching SAFE directory, or None if no match found
    """
    # Parse patch filename
    # Format: s1a-iw-raw-s-vh-20200407t052956-20200407t053029-032017-03b2e2-IW1-VD_1778.dat
    parts = patch_name.lower().split("-")
    
    if len(parts) < 9:
        print(f"Warning: Unexpected patch filename format: {patch_name}")
        return None
    
    satellite = parts[0].upper()  # s1a -> S1A
    start_time = parts[5]  # 20200407t052956
    end_time = parts[6]  # 20200407t053029
    abs_orbit = parts[7]  # 032017
    mission_data_take = parts[8].upper()  # 03b2e2 -> 03B2E2
    
    # Format times for SAFE directory pattern (uppercase T)
    start_time_safe = start_time.upper()  # 20200407T052956
    end_time_safe = end_time.upper()  # 20200407T053029
    
    # SAFE directory pattern: S1A_IW_RAW__0SDV_20200407T052956_20200407T053029_032017_03B2E2_*.SAFE
    pattern_prefix = f"{satellite}_IW_RAW__0SDV_{start_time_safe}_{end_time_safe}_{abs_orbit}_{mission_data_take}_"
    
    # Find matching SAFE directory
    for safe_dir in safe_dirs:
        if safe_dir.name.startswith(pattern_prefix) and safe_dir.name.endswith('.SAFE'):
            return safe_dir
    
    return None


def find_header_with_polarization(safe_dir: Path, swath: str, pol: str, burst: str = "02") -> Optional[Path]:
    """
    Find a header file with exact polarization match only.
    
    Args:
        safe_dir: SAFE directory to search in
        swath: Swath identifier (IW1, IW2, IW3)
        pol: Requested polarization (vh or vv)
        burst: Burst number (default: "02")
        
    Returns:
        Tuple of (header_path, actual_polarization) or (None, None) if not found
    """
    # Try requested polarization only - no fallback
    header_filename = f"SentinelHeader_{swath}-Burst_{burst}-{pol}.npy"
    header_path = safe_dir / header_filename
    
    if header_path.exists():
        return header_path, pol
    
    return None, None


def find_calibration_files(safe_dir: Path, swath: str, pol: str) -> Tuple[Optional[Path], Optional[Path], str]:
    """
    Find calibration data and header files with fallback logic.

    Priority:
    1. CAL1-IW1 (default)
    2. CAL2-IW1 (fallback)
    3. CAL3-IW1 (fallback)

    Polarization must match exactly - no fallback to other polarization.

    Args:
        safe_dir: SAFE directory to search in
        swath: Swath identifier (ignored in this implementation)
        pol: Polarization (vh or vv)

    Returns:
        Tuple of (cal_data_path, cal_header_path, cal_type) where cal_type describes which calibration was used
    """

    # Always use IW1 for calibration files
    fixed_swath = "IW1"

    # Try all calibration types (CAL1, CAL2, CAL3) with fixed swath and requested polarization
    for cal_type in ['CAL1', 'CAL2', 'CAL3']:
        cal_data_filename = f"SentinelData_{cal_type}-{fixed_swath}-Burst_01-{pol}.npy"
        cal_header_filename = f"SentinelHeader_{cal_type}-{fixed_swath}-Burst_01-{pol}.npy"
        cal_data_path = safe_dir / cal_data_filename
        cal_header_path = safe_dir / cal_header_filename

        if cal_data_path.exists() and cal_header_path.exists():
            match_type = "EXACT MATCH" if cal_type == "CAL1" else f"FALLBACK: {cal_type}"
            if cal_type != "CAL1":
                logger.info(f"Using {cal_type}-{fixed_swath}-{pol} (CAL1 not available) for {safe_dir.name}/{fixed_swath}-{pol}")
            return cal_data_path, cal_header_path, f"{cal_type}-{fixed_swath}-{pol} [{match_type}]"

    # Nothing found - log details
    logger.error(f"No calibration files found in {safe_dir.name} for {fixed_swath}-{pol}")
    logger.error(f"  Tried: CAL1/CAL2/CAL3-{fixed_swath}-{pol} (exact polarization only)")

    return None, None, "NONE"


def matches(split: str = 'train', dataset: str = 'vessel', data_dir: Optional[Path] = None) -> List[Dict[str, Path | str]]:
    """
    Matches L0 patches to header and calibration files for specified dataset.
    
    For each L0 patch in the specified split, finds the corresponding SAFE directory
    and extracts paths to:
    - Image burst header
    - Calibration pulse data (with fallback logic)
    - Calibration pulse header (with fallback logic)
    
    Args:
        split: Data split to process ('train', 'test', or 'val')
        dataset: Dataset type ('vessel', 'rfi', or 'flood')
        data_dir: Optional custom directory for L0 patches
        headers_dir: Optional custom directory for SAFE header folders
    Returns:
        List of dictionaries, where each dictionary contains:
            - 'patch_name': Patch identifier (e.g., 'VD_1778')
            - 'patch': Path to the L0 patch file
            - 'safe_dir': Path to the SAFE directory
            - 'header': Path to the image burst header file
            - 'cal_data': Path to the calibration pulse data
            - 'cal_header': Path to the calibration pulse header
            - 'polarization': Polarization (vh or vv)
            - 'swath': Swath identifier (IW1, IW2, or IW3)
            - 'split': Data split (train, test, or val)
            - 'cal_type': Description of which calibration was used
            - 'header_pol': Actual polarization of header used
    """
    
    if split not in splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {splits}")
    
    if dataset not in ['vessel', 'rfi', 'flood']:
        raise ValueError(f"Invalid dataset '{dataset}'. Must be one of ['vessel', 'rfi', 'flood']")
    
    logger.info(f"{'='*70}")
    logger.info(f"Starting matching process for split: {split}, dataset: {dataset}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"{'='*70}")
    
    matches: List[Dict[str, Path | str]] = []
    
    # Select paths based on dataset, allow override
    import inspect
    frame = inspect.currentframe()
    headers_dir = None
    patch_subdir = None
    headers_dir_arg = None
    if 'headers_dir' in frame.f_back.f_locals:
        headers_dir_arg = frame.f_back.f_locals['headers_dir']
    if headers_dir_arg is not None:
        headers_dir = Path(headers_dir_arg)
        if dataset == 'vessel':
            patch_subdir = 'new_viewsar3_full_split'
        elif dataset == 'rfi':
            patch_subdir = 'new_aresys_full_split'
        else:
            patch_subdir = 'new_kurosiwo_validation_sample_split'
    else:
        if dataset == 'vessel':
            headers_dir = vessel_data_headers_dir
            patch_subdir = 'new_viewsar3_full_split'
        elif dataset == 'rfi':
            headers_dir = rfi_data_headers_dir
            patch_subdir = 'new_aresys_full_split'
        else:
            headers_dir = flood_data_headers_dir
            patch_subdir = 'new_kurosiwo_validation_sample_split'

    # Build paths for this split, optionally override with data_dir
    if data_dir is not None:
        l0_patch_dir = data_dir
    else:
        l0_patch_dir = Path(str(headers_dir.parent) + f'/{patch_subdir}/{split}/raw')
    
    if not l0_patch_dir.exists():
        logger.error(f"L0 patch directory does not exist: {l0_patch_dir}")
        return matches
    
    # Get all SAFE directories
    if not headers_dir.exists():
        logger.error(f"Headers directory does not exist: {headers_dir}")
        return matches
    
    safe_dirs = [d for d in headers_dir.iterdir() if d.is_dir() and d.name.endswith('.SAFE')]
    
    if not safe_dirs:
        logger.error(f"No SAFE directories found in {vessel_data_headers_dir}")
        return matches
    
    # Get all L0 patches
    patches = list(l0_patch_dir.glob("*.dat"))
    
    if not patches:
        logger.error(f"No .dat files found in {l0_patch_dir}")
        return matches
    
    logger.info(f"Found {len(patches)} patches to process in split '{split}'")
    logger.info(f"Found {len(safe_dirs)} SAFE directories to match against")
    
    matched_count = 0
    skipped_count = 0
    missing_safe_count = 0
    missing_header_count = 0
    missing_cal_count = 0
    
    for patch in patches:
        # Parse patch filename to extract metadata
        # Format: s1a-iw-raw-s-vh-20200407t052956-20200407t053029-032017-03b2e2-IW1-VD_1778.dat
        parts = patch.stem.split("-")
        
        if len(parts) < 10:
            logger.warning(f"Unexpected patch filename format: {patch.name}, skipping...")
            skipped_count += 1
            continue
        
        pol = parts[4]  # vh or vv
        swath = parts[9]  # IW1, IW2, or IW3
        patch_id = parts[10]  # VD_1778
        
        # Find matching SAFE directory
        safe_dir = match_patch_to_safe(patch.name, safe_dirs)
        
        if safe_dir is None:
            logger.error(f"MISSING SAFE | Patch: {patch.name} | Scene: NONE FOUND")
            missing_safe_count += 1
            skipped_count += 1
            continue
        
        # Find header with polarization fallback
        header_path, header_pol = find_header_with_polarization(safe_dir, swath, pol, burst="02")
        
        if header_path is None:
            logger.error(f"MISSING HEADER | Patch: {patch.name} | Scene: {safe_dir.name} | Swath: {swath} | Pol: {pol}")
            missing_header_count += 1
            skipped_count += 1
            continue
        
        # Find calibration files with fallback logic
        cal_data_path, cal_header_path, cal_type = find_calibration_files(safe_dir, swath, pol)
        
        if cal_data_path is None or cal_header_path is None:
            logger.error(f"MISSING CALIBRATION | Patch: {patch.name} | Scene: {safe_dir.name} | Swath: {swath} | Pol: {pol} | Missing: CAL1/CAL2/CAL3-{swath}")
            missing_cal_count += 1
            skipped_count += 1
            continue
        
        # Create match dictionary
        match: Dict[str, Path | str] = {
            'patch_name': patch_id,
            'patch': patch,
            'safe_dir': safe_dir,
            'header': header_path,
            'cal_data': cal_data_path,
            'cal_header': cal_header_path,
            'polarization': pol,
            'swath': swath,
            'split': split,
            'cal_type': cal_type,
            'header_pol': header_pol
        }
        
        matches.append(match)
        matched_count += 1
        logger.info(f"MATCHED | Patch: {patch.name} | Scene: {safe_dir.name} | Swath: {swath} | Pol: {pol} | Header: {header_path.name} ({header_pol}) | Cal: {cal_type}")
    
    logger.info(f"{'='*70}")
    logger.info(f"Matching summary for split '{split}':")
    logger.info(f"  Successfully matched: {matched_count}/{len(patches)}")
    logger.info(f"  Total skipped: {skipped_count}")
    logger.info(f"    - Missing SAFE directory: {missing_safe_count}")
    logger.info(f"    - Missing header: {missing_header_count}")
    logger.info(f"    - Missing calibration: {missing_cal_count}")
    logger.info(f"{'='*70}")
    
    return matches


def matches_all_splits() -> Dict[str, List[Dict[str, Path | str]]]:
    """
    Match L0 patches to headers for all data splits.
    
    Returns:
        Dictionary with keys 'train', 'test', 'val', each containing a list of matches
    """
    all_matches = {}
    
    logger.info(f"\n{'#'*70}")
    logger.info(f"# Processing all splits")
    logger.info(f"{'#'*70}\n")
    
    for split in splits:
        all_matches[split] = matches_v2(split)
    
    logger.info(f"\n{'#'*70}")
    logger.info(f"# All splits completed")
    logger.info(f"# Total matches: {sum(len(m) for m in all_matches.values())}")
    logger.info(f"{'#'*70}\n")
    
    return all_matches


if __name__ == "__main__":
    # Example: Match patches for train split
    train_matches = matches('test')
    
    if train_matches:
        logger.info(f"\nExample match (first):")
        logger.info(f"  Patch: {train_matches[0]['patch'].name}")
        logger.info(f"  SAFE: {train_matches[0]['safe_dir'].name}")
        logger.info(f"  Polarization: {train_matches[0]['polarization']}")
        logger.info(f"  Header Polarization: {train_matches[0]['header_pol']}")
        logger.info(f"  Swath: {train_matches[0]['swath']}")
        logger.info(f"  Patch ID: {train_matches[0]['patch_name']}")
        logger.info(f"  Calibration Type: {train_matches[0]['cal_type']}")
    
    # Uncomment to process all splits
    # all_matches = matches_all_splits()
    # logger.info(f"\nTotal matches across all splits:")
    # for split, matches in all_matches.items():
    #     logger.info(f"  {split}: {len(matches)} patches")
