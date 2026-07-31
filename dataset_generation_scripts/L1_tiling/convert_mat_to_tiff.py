"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: convert_mat_to_tiff.py

Description:
   This is a Python script to take complex .mat files representing L1 SLC patches and
   converts them to the expected .tiff format. It also contains helpful functions for
   reading and manipulating information from both .mat and .tiff files.

History:
    - 2025-08-05:
        Initial version.
    - 2025-08-06:
        Added documentation.
    - 2025-08-18:
        Added main script when executing this python script.
        Added functionality for opening .mat files with h5py.
    - 2025-10-24:
        Added disclaimer for patch size when saving .tiff files.
        Updated paths in main script to be placeholders.

---------------------------------------------------------------------
Author: Anya Forestell (AMFF)
E-mail: amforestell@indracompany.com
Creation Date: 2025-08-05

© Copyright INDRA DEIMOS, 2025. All rights reserved.
---------------------------------------------------------------------
"""

# Import relevant libraries
import h5py
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from scipy.io import loadmat

# Define functions for reading data and metadata from .tiff and .mat files
def read_tiff_data(path):
    """
    Reads the band 1 data from a .tiff file. This function assumes
    only one band is used in SAR data.
    
    Args:
        path (string): Full path to the .tiff file.

    Returns:
        data (np.array): Array of band 1 data stored in the .tiff file. 
    """
    with rasterio.open(path) as src:
        data = src.read(1)
    return data

def print_tiff_data(path):
    """
    Reads and prints the band 1 data from a .tiff file. This function assumes
    only one band is used in SAR data. It also prints metadata such as the shape and
    dtype of the data stored in the .tiff file.
    
    Args:
        path (string): Full path to the .tiff file.

    Returns:
        data (np.array): Array of band 1 data stored in the .tiff file. 
    """
    with rasterio.open(path) as src:
        data = src.read(1)
    print("Printing data for "+path[-18:-5]+":")
    print(data)
    print(data.shape)
    print(data.dtype)
    print("")
    return data

def print_mat_cal_data(path, polarisation):
    """
    Reads and prints calibration data from a .mat file.
    
    Args:
        path (string): Full path to the .mat file.
        polarisation (string): Name of the polarisation tag to access in the .mat file.
    """
    data = loadmat(path)
    print(data.keys())
    cal_factor = data[polarisation]
    print(cal_factor)
    print(cal_factor.shape)
    print(cal_factor.dtype)

def print_mat_data(path, polarisation, RFI=False):
    """
    Reads and prints data from a .mat file. It also prints metadata such as the full
    list of keys, shape, and dtype of the data stored in the .mat file.
    
    Args:
        path (string): Full path to the .mat file.
        polarisation (string): Name of the polarisation tag to access in the .mat file.
        RFI (boolean): Default is False. Sets whether the use-case is RFI or not. RFI requires
                       a different .mat loading function.

    Returns:
        data (np.array): Array of data stored in the .mat file. 
    """
    if RFI==False:
        data = loadmat(path)[polarisation]
        print(data.keys())        
    elif RFI==True:
        f = h5py.File(path)
        print(list(f.keys()))
        data = f[polarisation][:]
        real = data["real"]
        img = data["imag"]
    else:
        print("Must provide a true/false value for whether this is the RFI use-case.")
        return None
    print("Printing data for "+path[-18:-4]+":")
    print(data)
    print(data.shape)
    print(data.dtype)
    if data.dtype == "complex128": # Sentinel-1 SLC data is stored as complex64
        data = data.astype(np.complex64)
    print(data.dtype)
    print("")
    return data

def get_amplitude_from_complex_data(complex_data):
    """
    Produces the amplitude array from a complex array.
    
    Args:
        complex_data (np.array): Array of complex data.

    Returns:
        amplitude (np.array): Array of amplitude of the complex data. 
    """
    amplitude = np.abs(complex_data)
    amplitude = amplitude.astype(np.float64)
    return amplitude

def get_complex_from_split_data(data, description=None):
    """
    Given an array where complex data is saved as separate rows for I and Q values,
    this function recombines them into complex values.

    Args:
        data (np.array): Data to be combined into complex values. It assumes odd rows
                         provide the I data while even rows provide the Q data.
        description (string): Optional. Default is None. Will be printed for information.

    Returns:
        complex_matrix (np.array): Array of complex64 values. 
    """
    I_mat = data[0::2, :]
    Q_mat = data[1::2, :]
    complex_matrix = I_mat +1j * Q_mat
    complex_matrix = complex_matrix.astype(np.complex64)
    print("Printing complex data for "+description+":")
    print(complex_matrix)
    print("")
    return complex_matrix

def plot_complex_data(complex_data):
    """
    Plots the amplitude of a complex array using Matplotlib. Useful comparison to
    the ESA SNAP tool.
    
    Args:
        complex_data (np.array): Array of complex values to be plotted. 
    """
    plt.imshow(np.abs(complex_data), cmap='gray') 

def save_complex_tiff(complex_data, dst_name):
    """
    Saves a complex array into a .tiff file, keeping the complex structure.
    
    Args:
        complex_data (np.array): Array of complex values to be saved.
        dst_name (string): Name of .tiff file to be saved.
    """
    with rasterio.open(
        dst_name, 'w',
        driver='GTiff',
        height=512,             # assumes a 512x512 patch
        width=512,              # assumes a 512x512 patch
        count=1,                # single band
        dtype='complex64',      # complex data type
        crs=None,               # or your CRS string, e.g., 'EPSG:4326'
        transform=None          # or your affine transform
    ) as dst:
        dst.write(complex_data, 1)  # write to band 1

def save_amp_tiff(data, dst_name):
    """
    Saves a non-complex array (amplitude should be calculated from a complex array in
    advance) into a .tiff file.
    
    Args:
        data (np.array): Array of amplitude values to be saved.
        dst_name (string): Name of .tiff file to be saved.
    """
    with rasterio.open(
        dst_name, 'w',
        driver='GTiff',
        height=512,             # assumes a 512x512 patch
        width=512,              # assumes a 512x512 patch
        count=1,                # single band
        dtype='float64',        # data type
        crs=None,               # or your CRS string, e.g., 'EPSG:4326'
        transform=None          # or your affine transform
    ) as dst:
        dst.write(data, 1)  # write to band 1

def print_all_tiff_tags(path):
    """
    Prints all available tags of a .tiff file.
    
    Args:
        path (string): Path to the .tiff file to be read.
    """
    with rasterio.open(path) as src:
        print("=== General metadata (file-level tags) ===")
        print(src.tags())

        print("\n=== Band-specific metadata ===")
        for i in range(1, src.count + 1):
            print(f"Band {i} tags:", src.tags(i))

        print("\n=== Spatial Metadata ===")
        print("CRS:", src.crs)
        print("Transform (Affine):", src.transform)
        print("Width x Height:", src.width, "x", src.height)
        print("Number of Bands:", src.count)
        print("Data Types per Band:", src.dtypes)
        print("Driver:", src.driver)

def convert_complex_mat(path_complex_mat_VV, path_complex_mat_VH, save_amplitude=False):
    """
    Reads a complex .mat file and saves it as a complex .tiff file.
    Optionally saves the amplitude of the complex data as a .tiff file.
    
    Args:
        path_complex_mat_VV (string): Path to the VV .mat file to be converted.
        path_complex_mat_VH (string): Path to the VH .mat file to be converted.
        save_amplitude (boolean): Default is False. Boolean to define whether
                                  amplitude of the complex data should be saved.
    """
    # Read data
    mat_complex_VV_data = print_mat_data(path_complex_mat_VV, "complexpatchvv")
    mat_complex_VH_data = print_mat_data(path_complex_mat_VH, "complexpatchvh")

    # Save complex as tiff
    save_complex_tiff(mat_complex_VV_data, "complex_from_mat_complex_VV.tiff") # SNAP does not read this well
    save_complex_tiff(mat_complex_VH_data, "complex_from_mat_complex_VH.tiff") # SNAP does not read this well

    if save_amplitude:
        # Save amplitude as tiff
        amp_mat_complex_VV_data = get_amplitude_from_complex_data(mat_complex_VV_data) # SNAP does read this well
        amp_mat_complex_VH_data = get_amplitude_from_complex_data(mat_complex_VH_data) # SNAP does read this well

        save_amp_tiff(amp_mat_complex_VV_data, "amp_from_mat_complex_VV.tiff")
        save_amp_tiff(amp_mat_complex_VH_data, "amp_from_mat_complex_VH.tiff")

if __name__ == "__main__":
    path_VV = "/fill_in_path/DB_OPENSAR_RFI_1_SLC_VV.mat"
    path_VH = "/fill_in_path/DB_OPENSAR_RFI_1_SLC_VH.mat"

    # Read VV and VH
    VV_data = print_mat_data(path_VV, "complexpatchvv", RFI=True)
    VH_data = print_mat_data(path_VH, "complexpatchvh", RFI=True)
