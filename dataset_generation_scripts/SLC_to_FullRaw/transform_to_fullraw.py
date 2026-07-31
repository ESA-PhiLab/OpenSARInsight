#!/usr/bin/env python3
"""
Transform vessel bounding boxes from SLC coordinates to Full Raw (L0) coordinates.

This script transforms BOTH azimuth (line) and range (sample) coordinates to raw space,
unlike the RC (Range Compressed) transformation which only transforms azimuth.

Usage:
    python transform_to_fullraw.py <xml_directory> <slc_product_folder> <output_directory>

The script:
1. Reads XML files containing vessel bounding boxes in SLC coordinates
2. Transforms both azimuth AND range coordinates using slc2raw functions
3. Adds BoundingBox_FullRaw_Global and BoundingBox_FullRaw_Local to each vessel
4. Saves updated XMLs to output directory

Note: This creates FULL raw coordinates (both directions transformed), 
different from RC which only transforms azimuth.
"""

import os
import sys
import glob
import logging
import argparse
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path

# Add slc2raw module to path
slc2raw_path = Path(__file__).parent.parent / "L0_preparation" / "main" / "slc2raw-main"
sys.path.insert(0, str(slc2raw_path))

from slc2raw.raw_data_params import raw_data_line, raw_data_pixel


def setup_logger(verbose: bool = True) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("fullraw_transform")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def transform_bbox_to_fullraw(
    xml_file: Path,
    slc_product_folder: Path,
    logger: logging.Logger
) -> bool:
    """
    Transform vessel bounding boxes from SLC to Full Raw coordinates.
    
    Args:
        xml_file: Path to XML file with SLC bounding boxes
        slc_product_folder: Path to SLC .SAFE product folder
        logger: Logger instance
        
    Returns:
        True if transformation succeeded, False otherwise
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get swath
        swath = int(root.find('.//SARData/SLCSwath').text)
        
        # Get patch corner coordinates
        samples = list(map(int, root.find('.//ProcessingData/Corner_Coord/SARData_Sample').text.split()))
        lines = list(map(int, root.find('.//ProcessingData/Corner_Coord/SARData_Line').text.split()))
        
        sample_ini = min(samples)
        line_ini = min(lines)
        
        # Find annotation file
        annotation_pattern = f'*00{swath}.xml'
        annotation_files = glob.glob(os.path.join(slc_product_folder, 'annotation', annotation_pattern))
        if not annotation_files:
            logger.error(f"Annotation file not found for swath {swath} in {slc_product_folder}")
            return False
        annotation_file = annotation_files[0]
        
        # Get annotation info
        xdoc = ET.parse(annotation_file)
        root_anno = xdoc.getroot()
        lines_per_burst = int(root_anno.find('.//swathTiming/linesPerBurst').text)
        
        # Get patch corners for reference
        min_slc_line = min(lines)
        max_slc_line = max(lines)
        
        _, ra_len, az_len, _, _, _, _, _ = raw_data_line(min_slc_line, annotation_file)
        
        # Get first valid sample
        burst_number = min_slc_line // lines_per_burst
        first_valid_sample_list = list(map(int, root_anno.findall('.//firstValidSample')[burst_number].text.strip().split()))
        first_valid_sample = next(v for v in first_valid_sample_list if v > -1)
        
        # Get patch min raw coordinates for local transformation
        min_slc_sample = min(samples) - first_valid_sample
        patch_raw_line_min_global, _, _, _, _, _, _, _ = raw_data_line(min_slc_line, annotation_file)
        patch_raw_line_min = int(np.floor(patch_raw_line_min_global - az_len / 2))
        patch_raw_sample_min = int(np.floor(raw_data_pixel(min_slc_sample, annotation_file)))
        
        # Transform each vessel's bounding box
        ships = root.findall('.//Ship')
        transformed_count = 0
        
        for ship in ships:
            bbox = ship.find('BoundingBox')
            if bbox is None:
                continue
            
            # Get SLC bounding box (local to patch) with validation
            top_elem = bbox.find('Top')
            left_elem = bbox.find('Left')
            bottom_elem = bbox.find('Bottom')
            right_elem = bbox.find('Right')
            
            if any(elem is None or elem.text is None for elem in [top_elem, left_elem, bottom_elem, right_elem]):
                logger.warning(f"Skipping ship with incomplete BoundingBox in {xml_file.name}")
                continue
            
            top = int(top_elem.text)
            left = int(left_elem.text)
            bottom = int(bottom_elem.text)
            right = int(right_elem.text)
            
            # Convert to global SLC coordinates
            slc_sample_min = sample_ini + left - 1
            slc_sample_max = sample_ini + right - 1
            slc_line_min = line_ini + top - 1
            slc_line_max = line_ini + bottom - 1
            
            # Transform to Full Raw coordinates (BOTH azimuth AND range)
            # Azimuth transformation
            raw_line_min_global, _, _, _, _, _, _, _ = raw_data_line(int(slc_line_min), annotation_file)
            raw_line_max_global, _, _, _, _, _, _, _ = raw_data_line(int(slc_line_max), annotation_file)
            raw_line_min = int(np.floor(raw_line_min_global - az_len / 2))
            raw_line_max = int(np.floor(raw_line_max_global + az_len / 2))
            
            # Range transformation (adjust for first valid sample)
            vessel_slc_sample_min = slc_sample_min - first_valid_sample
            vessel_slc_sample_max = slc_sample_max - first_valid_sample
            raw_sample_min = int(np.floor(raw_data_pixel(vessel_slc_sample_min, annotation_file)))
            raw_sample_max = int(np.floor(raw_data_pixel(vessel_slc_sample_max, annotation_file) + ra_len))
            
            # Check if FullRaw coordinates already exist (avoid duplicates)
            existing_fullraw_local = ship.find('BoundingBox_FullRaw_Local')
            existing_fullraw_global = ship.find('BoundingBox_FullRaw_Global')
            
            # Remove existing elements if present (safer than updating in case of malformed XML)
            if existing_fullraw_local is not None:
                ship.remove(existing_fullraw_local)
            if existing_fullraw_global is not None:
                ship.remove(existing_fullraw_global)
            
            # Create FullRaw bounding box elements (local coordinates)
            # Note: Local coordinates are 1-indexed (consistent with SLC Scene coordinates)
            bbox_fullraw_local = ET.SubElement(ship, 'BoundingBox_FullRaw_Local')
            ET.SubElement(bbox_fullraw_local, 'Top').text = str(raw_line_min - patch_raw_line_min + 1)
            ET.SubElement(bbox_fullraw_local, 'Left').text = str(raw_sample_min - patch_raw_sample_min + 1)
            ET.SubElement(bbox_fullraw_local, 'Bottom').text = str(raw_line_max - patch_raw_line_min + 1)
            ET.SubElement(bbox_fullraw_local, 'Right').text = str(raw_sample_max - patch_raw_sample_min + 1)
            
            # Create FullRaw bounding box elements (global coordinates)
            bbox_fullraw_global = ET.SubElement(ship, 'BoundingBox_FullRaw_Global')
            ET.SubElement(bbox_fullraw_global, 'Top').text = str(raw_line_min)
            ET.SubElement(bbox_fullraw_global, 'Left').text = str(raw_sample_min)
            ET.SubElement(bbox_fullraw_global, 'Bottom').text = str(raw_line_max)
            ET.SubElement(bbox_fullraw_global, 'Right').text = str(raw_sample_max)
            
            transformed_count += 1
        
        if transformed_count > 0:
            logger.info(f"Transformed {transformed_count} vessels in {xml_file.name}")
            return True
        else:
            logger.warning(f"No vessels found in {xml_file.name}")
            return False
            
    except Exception as e:
        logger.error(f"Error processing {xml_file.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Transform vessel bounding boxes from SLC to Full Raw coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform all XMLs in a directory
  python transform_to_fullraw.py \\
      output/scene_001/label/ \\
      inputs/slc/S1A_IW_SLC__1SDV_*.SAFE \\
      output/scene_001/label_fullraw/

  # Process single XML
  python transform_to_fullraw.py \\
      output/scene_001/label/DB_OPENSAR_DVD_001.xml \\
      inputs/slc/S1A_IW_SLC__1SDV_*.SAFE \\
      output/scene_001/label_fullraw/
        """
    )
    parser.add_argument("xml_path", help="XML file or directory containing XML files")
    parser.add_argument("slc_product", help="Path to SLC .SAFE product folder")
    parser.add_argument("output_dir", help="Output directory for transformed XMLs")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    logger = setup_logger(args.verbose)
    
    xml_path = Path(args.xml_path)
    slc_product = Path(args.slc_product)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not xml_path.exists():
        logger.error(f"XML path does not exist: {xml_path}")
        sys.exit(1)
    
    if not slc_product.exists():
        logger.error(f"SLC product not found: {slc_product}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of XML files
    if xml_path.is_file():
        xml_files = [xml_path]
    else:
        xml_files = list(xml_path.glob("*.xml"))
    
    if not xml_files:
        logger.error(f"No XML files found in {xml_path}")
        sys.exit(1)
    
    logger.info(f"Found {len(xml_files)} XML files to process")
    logger.info(f"SLC product: {slc_product}")
    logger.info(f"Output directory: {output_dir}")
    print()
    
    # Process each XML
    success_count = 0
    fail_count = 0
    
    for xml_file in xml_files:
        if transform_bbox_to_fullraw(xml_file, slc_product, logger):
            # Save transformed XML
            tree = ET.parse(xml_file)
            ET.indent(tree, space="   ", level=0)
            output_path = output_dir / xml_file.name
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            success_count += 1
        else:
            fail_count += 1
    
    print()
    logger.info("=" * 70)
    logger.info(f"Transformation complete: {success_count} succeeded, {fail_count} failed")
    logger.info("=" * 70)
    
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
