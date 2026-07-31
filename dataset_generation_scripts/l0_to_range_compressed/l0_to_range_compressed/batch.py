"""
Batch processing for l0_to_range_compressed
"""
import numpy as np
from pathlib import Path

from .scripts.read_l0_patch import read_l0_patch
from .scripts.range_compressed import range_compression
from .scripts.range_rescaler import rescale_range_compressed_data
from .utilities.match import matches

def compress_multiple(matches, rescaled_output_dir, scaled_size=512, window_size=8, window_type="hamming"):
    """
    Compress multiple L0 patches and save rescaled outputs.

    Args:
        matches (list of dict): Each dict must have keys: 'patch', 'header', 'cal_data', 'cal_header'.
        rescaled_output_dir (str or Path): Output directory for rescaled .npy files.
        scaled_size (int, optional): Output range size (default 512).
        window_size (int, optional): Sinc window half-width (default 8).
        window_type (str, optional): Window type for interpolation (default 'hamming').

    Returns:
        list of tuple: Each tuple: (patch_name, rescaled_path, error). Error is None if successful.
    """
    rescaled_output_dir = Path(rescaled_output_dir)
    rescaled_output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for match in matches:
        try:
            patch_path = Path(match['patch'])
            header_path = Path(match['header'])
            cal_data_path = Path(match['cal_data'])
            cal_header_path = Path(match['cal_header'])
            original_filename = patch_path.stem
            patch_data = read_l0_patch(patch_path)
            header = np.load(header_path, allow_pickle=True)[1]
            cal_data = np.load(cal_data_path, allow_pickle=True)
            cal_header = np.load(cal_header_path, allow_pickle=True)[1]
            rc_data, ref_length = range_compression(header, patch_data, cal_data, cal_header)
            rc_cropped = rc_data[:, :-ref_length]
            rescaled_data = rescale_range_compressed_data(
                rc_cropped, scaled_size=scaled_size, window_size=window_size, window_type=window_type
            )
            rescaled_filename = f"{original_filename}_rescaled.npy"
            rescaled_path = rescaled_output_dir / rescaled_filename
            rescaled_data = np.array(rescaled_data)
            np.save(rescaled_path, rescaled_data)
            results.append((patch_path.name, str(rescaled_path), None))
        except Exception as e:
            results.append((match.get('patch', 'unknown'), None, str(e)))
    return results


def compress_split(split, dataset, rescaled_output_dir, scaled_size=512, window_size=8, window_type="hamming", data_dir=None, limit=None):
    """
    Compress all L0 patches for a split/dataset.

    Args:
        split (str): Data split ('train', 'test', 'val').
        dataset (str): Dataset type ('vessel', 'rfi', 'flood').
        rescaled_output_dir (str or Path): Output directory for rescaled .npy files.
        scaled_size (int, optional): Output range size (default 512).
        window_size (int, optional): Sinc window half-width (default 8).
        window_type (str, optional): Window type for interpolation (default 'hamming').
        data_dir (str or Path, optional): Custom directory for L0 patches.
        limit (int, optional): Limit the number of patches to process.

    Returns:
        list of tuple: compress_multiple results for all matches in split/dataset.
    """
    match_list = matches(split=split, dataset=dataset, data_dir=data_dir)
    if limit is not None:
        match_list = match_list[:limit]
    return compress_multiple(match_list, rescaled_output_dir, scaled_size, window_size, window_type)
