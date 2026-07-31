import os
import inspect
import struct
import warnings
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

import CoreEnumeration_def as OBJ_CORE_ENUM
import CoreAssert_lib as OBJ_CORE_ASSERT
import CoreImageProcessing as OBJ_CORE_IP
import RadarConstants_def as OBJ_MIS_RADAR

def _read_tiff_dimensions(filepath):
    """Read width and height from a TIFF or BigTIFF file header without loading image data."""
    with open(filepath, 'rb') as f:
        byte_order = f.read(2)
        if byte_order == b'II':
            endian = '<'
        elif byte_order == b'MM':
            endian = '>'
        else:
            raise ValueError(f"Not a TIFF file: {filepath}")
        magic = struct.unpack(endian + 'H', f.read(2))[0]
        if magic == 42:  # Classic TIFF
            ifd_offset = struct.unpack(endian + 'I', f.read(4))[0]
            f.seek(ifd_offset)
            num_entries = struct.unpack(endian + 'H', f.read(2))[0]
            width = height = None
            for _ in range(num_entries):
                entry = f.read(12)
                tag = struct.unpack(endian + 'H', entry[:2])[0]
                type_id = struct.unpack(endian + 'H', entry[2:4])[0]
                if tag in (256, 257):
                    val = struct.unpack(endian + 'H', entry[8:10])[0] if type_id == 3 \
                          else struct.unpack(endian + 'I', entry[8:12])[0]
                    if tag == 256:
                        width = val
                    else:
                        height = val
                if width is not None and height is not None:
                    break
        elif magic == 43:  # BigTIFF
            f.read(4)  # skip bigtiff-specific fields
            ifd_offset = struct.unpack(endian + 'Q', f.read(8))[0]
            f.seek(ifd_offset)
            num_entries = struct.unpack(endian + 'Q', f.read(8))[0]
            width = height = None
            for _ in range(num_entries):
                entry = f.read(20)
                tag = struct.unpack(endian + 'H', entry[:2])[0]
                type_id = struct.unpack(endian + 'H', entry[2:4])[0]
                if tag in (256, 257):
                    val = struct.unpack(endian + 'H', entry[12:14])[0] if type_id == 3 \
                          else struct.unpack(endian + 'Q', entry[12:20])[0]
                    if tag == 256:
                        width = val
                    else:
                        height = val
                if width is not None and height is not None:
                    break
        else:
            raise ValueError(f"Unsupported TIFF magic: {magic}")
    if width is None or height is None:
        raise ValueError(f"Could not read dimensions from TIFF: {filepath}")
    return width, height


# Function to generate the constants dictionary dynamically from Constants_def.py
def fcn_GenDataDictionary(constants_file):
    """
    Generates a dictionary of constants from the specified constants file.

    :param constants_file: Path to the Python file containing the generated constants.
    :return: A dictionary containing the constants from the constants file.
    """
    constants = {}
    try:
        with open(constants_file, 'r') as f:
            exec(f.read(), constants)
    except Exception as e:
        print(f"Error loading constants from {constants_file}: {e}")
        return None

    return constants

# Function to assert image name format
def fcn_AssertImageName(imageName, constants_file, Case, verbosity: OBJ_CORE_ENUM.Verbosity):
    """
    Asserts if the image name is in the correct format based on previously generated constants.

    :param imageName: The image name to be verified.
    :param verbosity: The verbosity level to control the logging output.
    :param constants_file: Path to the Python file containing the generated constants.
    :return: True if the image name matches the expected format, False otherwise.
    """
    ImgType = "None"
    ImgFormat = "tiff"
    # Print name of function for debugging
    if verbosity.value > OBJ_CORE_ENUM.Verbosity.HIGH.value:
        print(f"\t\tDebugging in {OBJ_CORE_ENUM.Colors.YELLOW.value}{inspect.currentframe().f_code.co_name}(){OBJ_CORE_ENUM.Colors.RESET.value}:")

    # Retrieve the constants from constants_file using fcn_GenDataDictionary()
    dataDictionary = fcn_GenDataDictionary(constants_file)

    if dataDictionary is None:
        print("Error: Failed to load constants.")
        return False, ImgType, ImgFormatExtracted

        # Split the image name by underscores (_)
    image_name_parts = imageName.split('_')
    if len(image_name_parts) == 7:
        ImgType = image_name_parts[5]
        ImgFormatExtracted = image_name_parts[6]
        ImgFormatExtracted = ImgFormatExtracted[3:]
    if len(image_name_parts) == 6:
        ImgType = image_name_parts[4]
        ImgFormatExtracted = image_name_parts[5]
        ImgFormatExtracted = ImgFormatExtracted[3:]
    if len(image_name_parts) == 5:
        ImgType = image_name_parts[2]
        ImgFormatExtracted = image_name_parts[4]
        ImgFormatExtracted = ImgFormatExtracted[5:]
    
    # Use regular expressions to find integers in the image name parts
    integers = [part for part in image_name_parts if part.isdigit()]
    # Construct the expected image name based on constants
    expected_image_name = ""
    # Initialize the counter for range constants
    foundIntegers = 0
    # Iterate over the constants dictionary
    constant_keys = list(dataDictionary.keys())  # Get a list of all keys to iterate sequentially
    
    if Case[0] == "F":
        Case = "FD_"
    if Case[0] == "R":
        Case = "RFI_"
    if Case[0] == "V":
        Case = "VD_"

    for idx, constant_key in enumerate(constant_keys):
        if isinstance(dataDictionary[constant_key], str) and not constant_key.startswith(('MIN_', 'MAX_')):
            # Step 1: Concatenate string constants
            expected_image_name += dataDictionary.get(constant_key, '') + "_"
            if idx==2:
                expected_image_name += Case
        elif isinstance(dataDictionary[constant_key], int) and constant_key.startswith('MIN_'):
            # Step 2: Process range constants

            # Get the corresponding MAX_ constant
            min_thr = dataDictionary.get(constant_key)  # MIN_ value
            max_thr = dataDictionary.get(constant_keys[idx + 1])  # Next item should be the corresponding MAX_ value

            if min_thr is None or max_thr is None:
                print(f"Error: One or more constants for the ranges are not defined in Constants_def.py.")
                return False, ImgType, ImgFormatExtracted

            # Ensure max_thr is an integer
            max_thr = int(max_thr)

            # Verify the TileCounter and RAWimageReferece values against the ranges
            if min_thr <= int(integers[foundIntegers]) <= max_thr:  # Cast integers[foundIntegers] to int
                expected_image_name += str(integers[foundIntegers]) + "_"
            else:
                print(f"Item value {integers[foundIntegers]} is out of range ({min_thr}-{max_thr}).")
                return False, ImgType, ImgFormatExtracted

            foundIntegers += 1  # Move to the next integer value

    # Remove the trailing underscore
    expected_image_name = expected_image_name[:-1]
    # Add the file extension
    expected_image_name += "."
    expected_image_name += ImgFormatExtracted

    # Print the expected and actual image name for debugging
    if verbosity.value > OBJ_CORE_ENUM.Verbosity.MEDIUM.value:
        print(f"Expected image name: {expected_image_name}")
        print(f"Actual image name: {imageName}")

    # Compare the constructed image name with the given image name using fcn_AssertString
    is_correct_format = OBJ_CORE_ASSERT.fcn_AssertString(expected_image_name, imageName, 
                                                         verbosity, 0, 0,
                                                         len(expected_image_name)-5)

    if is_correct_format:
        if verbosity.value > OBJ_CORE_ENUM.Verbosity.MEDIUM.value:
            print(f"Image name format is CORRECT.")
        return 1, ImgType, ImgFormatExtracted
    else:
        if verbosity.value > OBJ_CORE_ENUM.Verbosity.MEDIUM.value:
            print(f"Image name format is INCORRECT.")
        return 0, ImgType, ImgFormatExtracted

_fcn_ProcessSAR_count = 0

def fcn_ProcessSAR(TypeOfImages, folderTestPath, filename, ImgFormatExtracted, verbosity):
    global _fcn_ProcessSAR_count
    _fcn_ProcessSAR_count += 1
    print(f"\r  Processing file {_fcn_ProcessSAR_count}: {filename[:60]:<60}", end='', flush=True)

    filepath = os.path.join(folderTestPath, filename)
    # SAR TIFF files contain complex/float data that PIL cannot identify;
    # skip PIL entirely and read only the TIFF header for dimensions.
    if filename.lower().endswith(('.tiff', '.tif')):
        w, h = _read_tiff_dimensions(filepath)
    else:
        # For other formats try PIL, fall back to TIFF header parser
        try:
            image = Image.open(filepath)
            w, h = image.width, image.height
        except Exception:
            w, h = _read_tiff_dimensions(filepath)

    if w != 512 or h != 512:
        print(f"\n  [SIZE] {filename}: {w}x{h}")

    return w, h

def extract_sar_product_from_file(XML_path):
    try:
        tree = ET.parse(XML_path)
        root = tree.getroot()
        sar_product = root.find('.//SARProduct')
        return sar_product.text if sar_product is not None else None
    except (ET.ParseError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return None

def fcn_AssertDat(imageName, extractedStrings, NumOfMatchRawXML, NumOfRaw, FormatValidity):
    # assert format
    parts1 = imageName.split('.')
    if parts1[1] == 'dat':
        FormatValidity = True
        NumOfRaw = NumOfRaw + 1
    else:        
        FormatValidity = False
    
    # assert name content
    parts0 = imageName.split('-')
    if len(parts0) >= 9:
        id1 = parts0[7]  # "030967"
        id2 = parts0[8]  # "038e50"
        for product in extractedStrings:
            parts = product.split('_')
            if len(parts) >= 8:
                if id1 == parts[7]:#  and id2 == parts[8]:  # "03B2E2"
                    NumOfMatchRawXML = NumOfMatchRawXML + 1
                    return FormatValidity, NumOfRaw, NumOfMatchRawXML
    return FormatValidity, NumOfRaw, NumOfMatchRawXML
    
    

    # Example usage:
    #file_path = "path/to/your/file.xml"
    #result = extract_sar_product_from_file(file_path)
    #print("SAR Product:", result)
    #print(constants_file)
    #print(Case)