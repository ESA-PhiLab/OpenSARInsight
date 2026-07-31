import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import inspect
import cv2

import CoreConstants_def as OBJ_CORE_CONSTANT
import CoreEnumeration_def as OBJ_CORE_ENUMERATION
import CoreAssert_lib as OBJ_CORE_ASSERT
import RadarConstants_def as OBJ_MIS_RADAR
# ========================================================================================
# =================================== LOCAL FUNCTIONS ====================================
# ========================================================================================

# read a SAR image (recommended tiff format) - main read image function
def fcn_ReadSARImage(TypeOfImages, parentfolder, filename, RADARBuffer, ImageFormat, verbosity: OBJ_CORE_ENUMERATION.Verbosity):
    # print name of fcn for debugging
    if verbosity.value>OBJ_CORE_ENUMERATION.Verbosity.HIGH.value:
        print(f"\t\tDebugging in {OBJ_CORE_ENUMERATION.Colors.YELLOW.value}{inspect.currentframe().f_code.co_name}(){OBJ_CORE_ENUMERATION.Colors.RESET.value}:")
    
    # verify RADAR buffer lenght (how many pixels is able to read at once)
    if(OBJ_CORE_ASSERT.fcn_AssertInteger(RADARBuffer, OBJ_MIS_RADAR.RADAR_BUFFER_SIZE, verbosity)):
        if verbosity.value>OBJ_CORE_ENUMERATION.Verbosity.LOW.value:
            print('\t\tCorrect RADAR buffer size.')
    
    # verify image format
    if TypeOfImages == 'PNG':
        if(OBJ_CORE_ASSERT.fcn_AssertString(ImageFormat, OBJ_CORE_ENUMERATION.ImageFormat.PNG.value, verbosity)):
            if verbosity.value>OBJ_CORE_ENUMERATION.Verbosity.LOW.value:
                print('\t\tCorrect image format.')
            else:
                if verbosity.value>OBJ_CORE_ENUMERATION.Verbosity.LOW.value:
                    print('\t\tIncorrect format.')
        else:
            print('\t\tFailed asserting format')
    else:     
        if(OBJ_CORE_ASSERT.fcn_AssertString(ImageFormat, OBJ_CORE_ENUMERATION.ImageFormat.TIFF.value, verbosity)):
            if verbosity.value>OBJ_CORE_ENUMERATION.Verbosity.LOW.value:
                print('\t\tCorrect image format.')
            else:
                if verbosity.value>OBJ_CORE_ENUMERATION.Verbosity.LOW.value:
                    print('\t\tIncorrect format.')
        else:
            print('\t\tFailed asserting format')
    
    # Read a TIFF file as binary data in chunks."""
    binary_data = bytearray()
    
    # read pixels in the same manner the RADAR is reading (a set of RADAR_BUFFER_SIZE at once until the full image is read) 
    with open(os.path.join(parentfolder, filename), "rb") as file:
        while BUFF_SIZE := file.read(RADARBuffer):  # Read in chunks of RADAR_BUFFER_SIZE [bytes]
            binary_data.extend(BUFF_SIZE)
    
    return binary_data

# Displays a grayscale image from a NumPy array for [displayInterval] seconds
def fcn_DisplaySARImage(image_array, verbosity: OBJ_CORE_ENUMERATION.Verbosity, displayInterval: float = OBJ_CORE_CONSTANT.DISPLAY_INTERVAL):
    # print name of fcn for debugging
    if verbosity.value>OBJ_CORE_ENUMERATION.Verbosity.HIGH.value:
        print(f"\t\tDebugging in {OBJ_CORE_ENUMERATION.Colors.YELLOW.value}{inspect.currentframe().f_code.co_name}(){OBJ_CORE_ENUMERATION.Colors.RESET.value}:")
        print(f"\t\t\tdisplay image with param: interval = {OBJ_CORE_ENUMERATION.Colors.YELLOW.value}{displayInterval}{OBJ_CORE_ENUMERATION.Colors.RESET.value} [s]")
    plt.imshow(image_array, cmap="gray")
    plt.axis("off")
    plt.title("FD_2_NNN")
    plt.show(block=False)  # Show the image without blocking execution
    plt.pause(displayInterval)         # Display for TBD seconds
    plt.close()            # Close the image window
  
# Read a tiff image using Pillow open-source library    
def fcn_ReadSARImagePillow(filename, verbosity: OBJ_CORE_ENUMERATION.Verbosity, displayInterval: float = OBJ_CORE_CONSTANT.DISPLAY_INTERVAL):
    # Open a TIFF image using Pillow
    image = Image.open(filename)
    
    # Print the image mode
    if verbosity.value>OBJ_CORE_ENUMERATION.Verbosity.HIGH.value:
        print(f"Image Mode: {image.mode}")
        print(f"Image Size: {image.size}")
    
    # Convert to numpy array (compatible with OpenCV)
    image_array = np.array(image)

    # Display the image with the filename as the window title
    cv2.imshow(filename, image_array)
    
    # Wait for a key press and close the window
    cv2.waitKey(int(1000*displayInterval))
    cv2.destroyAllWindows()

    print(f"Image Shape: {image_array.shape}")
    return image_array  # Returning the array for further use