import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# User defined functions and classes 
from scripts import helper_functions
from constants import patches0, patches1, labels

print(f'Label Path : {labels}')
print(f'P1 Path : {patches1}')
print(f'P0 Path : {patches0}')


# Load Patches
P1_images = helper_functions.Utilities.load_tiff_files(patches1)

# Select P1 patches 
P1_SLC_vh = helper_functions.Utilities.product_filter(P1_images, 'SLC', 'VH')

# Load Labels 
labels = helper_functions.Utilities.load_xml_files(labels)

# Select a Label to confirm loaded labels
print(f'{labels[0]}')

# Match Labels to Images
P1_SLC_vh_matched = helper_functions.Utilities.match_images_to_labels(P1_SLC_vh, labels) # P1 VH

# Select all patches from a given product
p1_ea99 = helper_functions.Utilities.select_patches_from_scene(P1_SLC_vh_matched, target_scene="S1A_IW_SLC__1SDV_20200824T052155_20200824T052225_034044_03F3B6_EA99")


# Checks
print(len(p1_ea99))