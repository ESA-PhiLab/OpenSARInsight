import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import H1
from scripts.helper_functions import Utilities

print('____________________________________________________________________')
print('Execution of Test script starts here')

# Get the H1 dir from dataset
d = H1["data"]
print(f'H1 data file path: {d}')
# Load image files
images = Utilities.load_tiff_files(d)
# Count the number of files in H1
print(f'Length of H1 images: {len(images)}')

# Check if there are enough images before accessing images[81]
if len(images) > 81:
    # Unpack the image tuple
    image, file_name = images[81]
    # Check Image data type
    print(f'image type {type(image)}')
    # Plot an Image
    Utilities.plot_image(image, file_name)
else:
    print("Not enough images to access images[81]")

print('____________________________________________________________________')
print('Filtering based on Product and Polarization starts here')

# Product types
product_type = ('SLC', 'GRD')

# Polarizations
polarization = ('VV', 'VH')

for product in product_type:
    for pol in polarization:
        print(f'{product} in {pol}')
        filtered_images = Utilities.product_filter(images, product, pol)

        # Check if there are any images after filtering
        if filtered_images:
            # Unpack the image tuple
            image, file_name = filtered_images[0]

            # Check Image data type
            print(f'image type {type(image)}')

            # Plot an Image
            Utilities.plot_image(image, file_name)
        else:
            print("No images found after filtering.")