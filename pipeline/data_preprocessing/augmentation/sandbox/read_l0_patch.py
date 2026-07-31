import numpy as np
import struct
import os
import matplotlib.pyplot as plt
from dataset_generation_scripts.utils import get_config
cfg = get_config("DATA_PREPROC_PATH")

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
    

def demo_read_l0_patch():
    filename = cfg["read_l0_patch"]["input_file"]
    patch = read_l0_patch(filename)
    print("Patch shape:", patch.shape)
    print("Sample value [0, 0]:", patch[0, 0])
    save_patch_visualizations(patch, output_dir=".")


def save_patch_visualizations(patch: np.ndarray, output_dir: str = "."):
    """Save amplitude and phase visualizations of a complex SAR patch.

    Args:
        patch (np.ndarray): 2D complex-valued SAR patch.
        output_dir (str): Directory where images will be saved. Defaults to current dir.
    """
    # Amplitude
    amplitude = np.abs(patch)
    plt.figure(figsize=(10, 6))
    plt.imshow(amplitude, cmap='gray', aspect='auto')
    plt.title("Amplitude (|I/Q|)")
    plt.xlabel("Range Samples")
    plt.ylabel("Azimuth Lines")
    plt.colorbar(label="Magnitude")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/l0_patch_amplitude.png", dpi=300)
    plt.close()

    # Phase
    phase = np.angle(patch)
    plt.figure(figsize=(10, 6))
    plt.imshow(phase, cmap='twilight', aspect='auto')
    plt.title("Phase (angle of I/Q)")
    plt.xlabel("Range Samples")
    plt.ylabel("Azimuth Lines")
    plt.colorbar(label="Radians")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/l0_patch_phase.png", dpi=300)
    plt.close()

    print("Saved visualizations to:")
    print(f"  → {output_dir}/l0_patch_amplitude.png")
    print(f"  → {output_dir}/l0_patch_phase.png")
