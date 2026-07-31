import numpy as np
import struct
import os

def read_l0_patch(file_name):
    """Read a L0 patch from a file

    Read the raw data matrix of a patch from the file with the name provided.
    File format is:
        - Data is stored in Little Endian format
        - Complex data are stored first real part, then imaginary part
        - Complex data are stored in 32-bits float type
        - File header contains the size (rows, cols) in uint32 type

            <Number of pulses (az_size)> in 32-bits unsigned integer
            <Number of samples (rg_size)> in 32-bits unsigned integer
            <real(pulse0,sample0)>
            <imag(pulse0,sample0)>
            ....
            <real(pulse0,rg_size)>
            <imag(pulse0,rg_size)>
            <real(pulse1,sample0)>
            <imag(pulse1,sample0)>
            ....
            <real(pulse1,rg_size)>
            <imag(pulse1,rg_size)>
            ...............
            <real(az_size,rg_size)>
            <imag(az_size,rg_size)>

    :parameter
        filename:   string. Name (path included) of the file to be read
    :return
        array of complex. Matrix with the raw data echoes of a single patch.
        Each row is a pulse and each column is a sample of the pulse
    """
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"File '{file_name}' does not exist.")
    
    try:
        with open(file_name, "rb") as fid:
            # Header: 8 bytes (2 x uint32)
            header = fid.read(8)
            if len(header) < 8:
                raise ValueError("Invalid file: Header incomplete.")
            
            nrows, ncols = struct.unpack('<II', header)
            num_data = nrows * ncols * 2  # I y Q for each data

            # data: nrows x ncols x 2 (I, Q values) in float32
            data = np.fromfile(fid, dtype='<f4', count=num_data)
            if data.size != num_data:
                raise ValueError("Invalid file: File has less data "
                                 "than expected.")

            # Complex matrix creation
            I = data[0::2]
            Q = data[1::2]
            l0_patch = I + 1j * Q
            return l0_patch.reshape((nrows, ncols))

    except PermissionError:
        raise PermissionError(f"File permissions do not allow for reading "
                              f"the file '{file_name}'.")
    except (OSError, struct.error, ValueError) as e:
        raise RuntimeError(f"Error processing file: {e}")
