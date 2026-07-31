# type: ignore
"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: range_rescaler.py

Description:
   This script rescales range-compressed patches to match the range-size of the L1 SLC patches.
   This is 512 for the Vessel and Flood use-cases; RFI will be larger, in the range of 1400-1600.
   Note that the patches output by this script should be used ONLY for the detection heads; the
   focusing model should only retain procedures that are in line with proper focusing techniques.

History:
    - 2025-10-10:
        Initial version.
    - 2025-12-10:
        Cleaned code; fixed a bug.
        Updated saved file names to specify they are rescaled.
        Updated src and dst paths.
        Sped up processing time.

---------------------------------------------------------------------
Author: Anya Forestell (AMFF)
E-mail: amforestell@indracompany.com
Creation Date: 2025-10-10

© Copyright INDRA DEIMOS, 2025. All rights reserved.
---------------------------------------------------------------------
"""

import numpy as np
from pathlib import Path
from scipy.signal import windows

def find_scaling_factor(file_name):
    """
    Finds the scaling factor based on the IW swath number.
    
    Parameters:
        file_name (str): Name of L0 file.
        
    Returns:
        scaling_factor (float): Scaling factor to go from compressed L0 range size to L1 range size.

    Ex:
    L1_range_size = compressed_L0_range_size * scaling_factor
    """
    IW_mode = int(file_name.split("-")[-2][-1])
    if IW_mode == 1:
        scaling_factor = 1.0
    elif IW_mode == 2:
        scaling_factor = 1.178571429
    elif IW_mode == 3:
        scaling_factor = 1.371428571
    else:
        print("This is not a recognised IW swath number: "+str(IW_mode[-1]))
        return None
    return scaling_factor

def check_scaled_length(scaled_length, expected_length, epsilon=1):
    """
    Checks if the scaled length will be approximately equal to the expected length.
    Returns true if it is the expected length; returns False if it is invalid or not of the expected length.
    
    Parameters:
        scaled_length (float or int): Scaled length of the range coordinate.
        expected_length (float or int): Expected length of the scaled range coordinate.
        epsilon (float or int): Allowed error. Default is 1.
        
    Returns:
        Boolean: True is the scaled length is equal to the expected length (+/- epsilon). False otherwise.
    """
    scaled_length = int(scaled_length)
    expected_length = int(expected_length)
    epsilon = int(epsilon)

    if scaled_length >= expected_length-epsilon and scaled_length <= expected_length+epsilon:
        # print("The scaled length is approximately equal to 512.")
        return True
    elif scaled_length < expected_length-epsilon:
        # print("The scaled length is smaller than 512.")
        return False
    elif scaled_length > expected_length+epsilon:
        # print("The scaled length is larger than 512; if trying to use this code for the RFI use-case, please update the expected range size.")
        return False
    else:
        # print("The scaled length is invalid.")
        return False

def rescale_range_compressed_data(data, scaled_size=512, window_size=8, window_type="hamming"):
    """
    Rescales the range direction of a range-compressed L0 patch to be the range size of an L1 SLC patch.
    
    Parameters:
        data: Data to be scaled. 
        scaled_size: Desired scaled size of data. Default is 512.
        window_size: Half-width of sinc window in samples. Default is 8, for a full window size of 16.
        window_type: Type of window to use for interpolation
        
    Returns:
        rescaled_data: Data that has been rescaled in the range direction (typically to size 512).
    """
    rescaled_data = []
    upsampled_positions = np.linspace(0, len(data[0]) - 1, scaled_size)
    rescaled_data = [sinc_interp(line, upsampled_positions=upsampled_positions, window_size=window_size, window_type=window_type) for line in data]
    return rescaled_data

def sinc_interp(range_line, upsampled_positions, window_size=8, window_type="hamming"):
    """
    Perform sinc-based interpolation with a Hamming window. Window type can be changed.
    
    Parameters:
        range_line: 1D input array (e.g., range line from L0 data)
        upsampled_positions: array of new sample positions (floats)
        window_size: half-width of sinc window in samples
        window_type: type of window to use for interpolation
        
    Returns:
        scaled_line: interpolated array at upsampled_positions
    """
    range_line = np.asarray(range_line)
    scaled_line = np.zeros_like(upsampled_positions, dtype=np.complex64)

    for i, t in enumerate(upsampled_positions):
        n = np.arange(np.floor(t - window_size), np.ceil(t + window_size) + 1)
        n = n.astype(int)
        # Keep indices within bounds
        valid = (n >= 0) & (n < len(range_line))
        n = n[valid]

        sinc_arg = t - n
        if window_type=="hamming":
            window = windows.hamming(len(n))
        else:
            print("Window type not recognised; please adjust code to allow for your preferred window type.")
        kernel = np.sinc(sinc_arg) * window
        kernel /= np.sum(kernel)  # Normalise

        scaled_line[i] = np.sum(range_line[n] * kernel)
    return scaled_line

def process_directory(src_path, dst_path):
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.mkdir(parents=True, exist_ok=True)

    for file_path in src_path.glob("*.npy"):
        print("")
        print(f"Processing file: {file_path.name}")
        
        data = np.load(file_path, allow_pickle=True)
        scaling_factor = find_scaling_factor(str(file_path.name))
        if scaling_factor == None:
            continue

        compressed_range_length = len(data[0])
        scaled_range_length = compressed_range_length*scaling_factor

        if check_scaled_length(scaled_length=scaled_range_length, expected_length=512, epsilon=1):
            print("Scaled length is as expected; proceeding with rescaling.")
            if compressed_range_length == 512:
                filename = str(file_path.name).split(".npy")[0] + str("_scaled.npy")
                dst_path_temp = dst_path / filename
                np.save(dst_path_temp, data)
            else:
                scaled_data = rescale_range_compressed_data(data, scaled_size=512, window_size=8, window_type="hamming")
                filename = str(file_path.name).split(".npy")[0] + str("_scaled.npy")
                dst_path_temp = dst_path / filename
                np.save(dst_path_temp, scaled_data)
        else:
            print("Scaled length is incorrect; skipping.")
            print("")
            continue

if __name__ == "__main__":
    # Example usage
    file_name = "s1a-iw-raw-s-vh-20200214t174427-20200214t174459-031252-03982e-IW3-VD_5750.npy"
    data = np.load(file_name, allow_pickle=True)
    rescaled_data = rescale_range_compressed_data(data, scaled_size=512, window_size=8, window_type="hamming")
    rescaled_data = np.array(rescaled_data)
    print(f"Rescaled data shape: {rescaled_data.shape}")
    