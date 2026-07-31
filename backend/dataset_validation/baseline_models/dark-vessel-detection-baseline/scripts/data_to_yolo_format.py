import os
import sys

# Add the parent directory to the Python path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from yolo_preprocessing import YoloPreprocessing
from constants import patches0, patches1, yolo_data_path, labels
from scripts import helper_functions
import cv2

# Load Patches and Labels
P0 = helper_functions.Utilities.load_tiff_files(patches0)  # Load patch set 0
P1 = helper_functions.Utilities.load_tiff_files(patches1)  # Load patch set 1
labels = helper_functions.Utilities.load_xml_files(labels)  # Load label XMLs

# Match Patches to Labels
P0_matched = helper_functions.Utilities.match_images_to_labels(P0, labels)
P1_matched = helper_functions.Utilities.match_images_to_labels(P1, labels)

# Select slices [2650:2950] to test, adjust if needed
P0_matched_slice = P0_matched[2650:2950]
P1_matched_slice = P1_matched[2600:]

# Define File Paths
yolo_data_dir = yolo_data_path
patches0_dir = os.path.join(yolo_data_dir, 'patches0')

# Convert the Matched Patches to Yolo Format
YoloPreprocessing.convert_to_yolo_format(P1_matched_slice, yolo_data_dir)
YoloPreprocessing.convert_to_yolo_format(P0_matched_slice, patches0_dir)

# Split the dataset and create config
dataset_config = YoloPreprocessing.split_dataset(
    yolo_data_dir,
    negative_dir=patches0_dir
)