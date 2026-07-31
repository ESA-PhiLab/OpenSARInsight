import os
import sys
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import H1
from scripts.helper_functions import Utilities

print('____________________________________________________________________')
print('Execution of Test script starts here')

# Get the H1 dir from dataset
d = H1["labels"]
print(f'H1 labels file path: {d}')  # Debugging line

# Load image files
labels = Utilities.load_xml_files(d)

# Unpack Labels 
label, label_name = labels[0]
# Count the number of files in H1
print(f'Length of H1 Labels: {len(labels)}')

# Get root
root = label.getroot() 
attributes = root.attrib

print(f'Attributes : {attributes}')

for child in root:
   print(f'{child.tag}, {child.attrib}')

print('Tags in the XML file:')
for elem in root.iter():
    print(f' - {elem.tag}')

print(ET.tostring(root, encoding='utf8').decode('utf8')) # print xml as string

for box in root.iter('BoundingBox'):
    print(box.tag, box.attrib, box.text)