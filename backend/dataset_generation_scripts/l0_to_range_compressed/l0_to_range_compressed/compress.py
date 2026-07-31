# type: ignore
"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: compress.py

Description:
    Entry point to Perform range compression of Sentinel 1 decoded L0 data using a nominal image replica
    generated from L0 header parameters.

History:
    - 2026-04-29:
      V1

---------------------------------------------------------------------
Author: Abdulhameed Yunusa (ABHY)
E-mail: ayunusa@indracompany.com
Creation Date: 2026-04-28

Â© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""

import numpy as np
from pathlib import Path
from .scripts.read_l0_patch import read_l0_patch
from .scripts.range_compressed import range_compression
from .scripts.range_rescaler import rescale_range_compressed_data

def compress(
    patch_path,
    header_path,
    cal_data_path,
    cal_header_path,
    rescaled_output_dir,
    scaled_size=512,
    window_size=8,
    window_type="hamming"
):
    """
    Compress a single L0 patch and save the rescaled output.

    Args:
        patch_path (str or Path): Path to L0 patch file.
        header_path (str or Path): Path to header .npy file.
        cal_data_path (str or Path): Path to calibration data .npy file.
        cal_header_path (str or Path): Path to calibration header .npy file.
        rescaled_output_dir (str or Path): Output directory for rescaled .npy file.
        scaled_size (int, optional): Output range size (default 512).
        window_size (int, optional): Sinc window half-width (default 8).
        window_type (str, optional): Window type for interpolation (default 'hamming').

    Returns:
        str: Path to saved rescaled .npy file.
    """
    patch_path = Path(patch_path)
    header_path = Path(header_path)
    cal_data_path = Path(cal_data_path)
    cal_header_path = Path(cal_header_path)
    rescaled_output_dir = Path(rescaled_output_dir)
    rescaled_output_dir.mkdir(parents=True, exist_ok=True)

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
    return str(rescaled_path)