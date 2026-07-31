"""
Project: OpenSAR Insight
Customer: ESA

File: read_l0_patch.py

Description:
    Level-0 SAR patch reader for binary data files.

Functions:
    read_l0_patch: Read complex SAR data from binary L0 patch files
    
Author : University de Alcala
Email: Nil
Date: 

History: <summary of changes since the file was first baselined>
Version | Date       | Author       | Change History
1.0     |22/07/2025  | Abdulhameed Yunusa | Modified code to conform to PEP8 Standard

© Copyright INDRA DEIMOS, 2025. All rights reserved.
"""
import os
import struct

import numpy as np
import numpy.typing as npt 

def read_l0_patch(file_name: str) -> npt.NDArray[np.complex64]:
    """Read a Level-0 SAR patch from a file.

    Reads the raw data matrix of a patch from the specified file. The file format is:
    - Little Endian
    - Complex data stored as real part followed by imaginary part
    - Complex data in 32-bit float type
    - File header contains size (rows, cols) in uint32 type
    - Data layout: <rows> <cols> <real(pulse0,sample0)> <imag(pulse0,sample0)> ...
                   <real(rows,cols)> <imag(rows,cols)>

    Args:
        file_name (str): Path to the L0 SAR data file.

    Returns:
        numpy.ndarray: Complex-valued matrix of shape (rows, cols) containing the raw data echoes.
            Each row represents a pulse, and each column represents a sample.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        PermissionError: If the file cannot be read due to permissions.
        RuntimeError: If the file is invalid or cannot be processed (e.g., incomplete header or data).
    """
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"File '{file_name}' does not exist.")

    try:
        with open(file_name, "rb") as fid:
            # Read header: 8 bytes (2 x uint32 for rows, cols)
            header = fid.read(8)
            if len(header) < 8:
                raise ValueError("Invalid file: Header incomplete.")

            nrows, ncols = struct.unpack('<II', header)
            num_data = nrows * ncols * 2  # Real and imaginary parts

            # Read data: nrows x ncols x 2 (real, imag) in float32
            data = np.fromfile(fid, dtype='<f4', count=num_data)
            if data.size != num_data:
                raise ValueError("Invalid file: File has less data than expected.")

            # Create complex matrix
            real_part = data[0::2]  # I
            imag_part = data[1::2]  # Q
            l0_patch = real_part + 1j * imag_part
            return l0_patch.reshape((nrows, ncols))

    except PermissionError:
        raise PermissionError(f"File permissions do not allow reading '{file_name}'.")
    except (OSError, struct.error, ValueError) as e:
        raise RuntimeError(f"Error processing file: {e}")
