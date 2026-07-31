"""Trace coordinates from SLC to RAW"""
import os
import glob
import sys
import argparse
import logging
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from pathlib import Path 
from slc2raw.raw_data_params import raw_data_line, raw_data_pixel
from dataset_generation_scripts.utils import get_config

cfg = get_config("DGS_CONFIG_PATH")

PROG = __package__
VERSION = "1.0"


def slc_to_raw_traceability(label_folder:Path,
                            product_folder:Path,
                            output_file:Path,
                            logger:logging.Logger):
    """Trace coordinates from SLC to RAW

    Function for tracing the coordinates of the SLC product to coordinates
    of the RAW product.


    :param:
        label_folder:          Folder with SLC label files.
        product_folder:        Folder with the product under analysis.
        output_file:           File where output is generated.
        logger:                Python Logger.

    :return:
        rawH.npy:              Matrix with coordinates in the raw data.
        traceability_log.txt:  File with information about the process.

    """
    logger.info(f"Started slc_to_raw_traceability for {product_folder.stem}")

    # Identify the product
    if not product_folder.exists():
        logger.info("Product folder does not exist.\n")
        return
    slc_name = product_folder.stem

    # Check the number of label files
    label_files = glob.glob(os.path.join(label_folder, '*.xml'))
    raw_detection = []
    # Extract information from XML files
    # after testing would be better to vectorize this
    for label_file in label_files:
        tree = ET.parse(label_file)
        root = tree.getroot()

        # Read the product
        sar_data = root.find('.//SARData')
        sar_product = sar_data.find('.//SARProduct')
        sar_product_ini = sar_product.text
        sar_product = sar_product_ini.split('.')[0] if '.' in sar_product_ini else sar_product_ini

        # Check if file belongs to the same product
        if sar_product not in slc_name:
            #logger.warning(f"Warning: This label file {label_file} belongs to a different product than {slc_name}. Skipped label file.\n")
            continue

        # Get swath
        swath = int(root.find('.//SARData/SLCSwath').text)

        # Get patch number
        scene_id = root.find('.//Scene_Info/Scene_ID').text
        patch_id = scene_id.split('OPENSAR_')[-1]

        # Get coordinates
        samples = list(map(int, root.find('.//ProcessingData/Corner_Coord/SARData_Sample').text.split()))
        lines   = list(map(int, root.find('.//ProcessingData/Corner_Coord/SARData_Line').text.split()))

        sample_ini, sample_fin = min(samples), max(samples)
        line_ini, line_fin = min(lines), max(lines)
        pixel_cords = np.array([[sample_ini, line_ini],
                                [sample_ini, line_fin],
                                [sample_fin, line_fin],
                                [sample_fin, line_ini]])

        # Read image product annotation file
        annotation_pattern = f'*00{swath}.xml'
        annotation_files = glob.glob(os.path.join(product_folder, 'annotation', annotation_pattern))
        if not annotation_files:
            logger.error(f"Annotation File not found. Skipping {label_file}")
            continue
        annotation_file = annotation_files[0]

        xdoc = ET.parse(annotation_file)
        root_anno = xdoc.getroot()
        # Get burst size
        lines_per_burst = int(root_anno.find('.//swathTiming/linesPerBurst').text)

        # Check if SLC coordinates span over more than one burst
        # Zero-indexed coordinates
        burst_number_start = min(pixel_cords[:, 1]) // lines_per_burst
        burst_number_end = max(pixel_cords[:, 1]) // lines_per_burst

        if burst_number_start != burst_number_end:
            logger.info(f"Warning: label {patch_id} lies between two bursts. Detection removed.\n")
            continue

        # Get validity vectors
        first_valid_sample_list = list(map(int, root_anno.findall('.//firstValidSample')[burst_number_start].text.strip().split()))

        # Tracing SLC lines to L0 lines and get line number within burst
        min_slc_line = int(min(pixel_cords[:, 1]))
        max_slc_line = int(max(pixel_cords[:, 1]))

        _, ra_len, az_len, min_raw_line_burst, burst_time, min_az_mid_burst_time, min_sensing_mid_burst_time, l1_burst_num = raw_data_line(min_slc_line, annotation_file)
        _, _, _, max_raw_line_burst, _,_,_, _ = raw_data_line(max_slc_line, annotation_file)

        # Adding reference function both azimuth sides
        min_raw_line = int(np.floor(min_raw_line_burst - az_len / 2))
        max_raw_line = int(np.floor(max_raw_line_burst + az_len / 2))
        # Get valid range samples
        first_valid_sample = next(v for v in first_valid_sample_list if v > -1)
        # Get min & max range samples
        min_slc_sample = min(pixel_cords[:, 0]) - first_valid_sample
        max_slc_sample = max(pixel_cords[:, 0]) - first_valid_sample

        # Trace SLC range samples to L0 samples
        min_raw_sample = int(np.floor(raw_data_pixel(min_slc_sample, annotation_file)))
        max_raw_sample = int(np.floor(raw_data_pixel(max_slc_sample, annotation_file) + ra_len))

        # Transform vessel bounding boxes if present
        list_of_ships = root.find('.//ProcessingData/List_of_ships')
        if list_of_ships is not None:
            ships = list_of_ships.findall('Ship')
            
            # Only process if there are actually ships to transform
            if ships:
                updated_count = 0
                for ship in ships:
                    bbox = ship.find('BoundingBox')
                    if bbox is None:
                        continue
                    
                    # Get SLC bounding box coordinates (local to patch) with validation
                    top_elem = bbox.find('Top')
                    left_elem = bbox.find('Left')
                    bottom_elem = bbox.find('Bottom')
                    right_elem = bbox.find('Right')
                    
                    if any(elem is None or elem.text is None for elem in [top_elem, left_elem, bottom_elem, right_elem]):
                        logger.warning(f"Skipping ship with incomplete BoundingBox in {label_file}")
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
                    
                    # Transform vessel bbox to RC (Range Compressed) coordinates
                    # RC: Only azimuth is transformed; range samples remain unchanged
                    vessel_raw_line_min_global, _, _, _, _, _, _, _ = raw_data_line(int(slc_line_min), annotation_file)
                    vessel_raw_line_max_global, _, _, _, _, _, _, _ = raw_data_line(int(slc_line_max), annotation_file)
                    vessel_raw_line_min = int(np.floor(vessel_raw_line_min_global - az_len / 2))
                    vessel_raw_line_max = int(np.floor(vessel_raw_line_max_global + az_len / 2))
                    
                    # Range samples remain unchanged for RC (range-compressed) data
                    vessel_rc_sample_min = slc_sample_min
                    vessel_rc_sample_max = slc_sample_max
                    
                    # Check if RC coordinates already exist (avoid duplicates)
                    existing_rc_local = ship.find('BoundingBox_RC_Local')
                    existing_rc_global = ship.find('BoundingBox_RC_Global')
                    
                    # Remove existing elements if present (safer than updating in case of malformed XML)
                    if existing_rc_local is not None:
                        ship.remove(existing_rc_local)
                    if existing_rc_global is not None:
                        ship.remove(existing_rc_global)
                    
                    # Create RC bounding box elements (local coordinates)
                    # Note: Range (Left/Right) copied directly from SLC (no transformation)
                    #       Azimuth (Top/Bottom) transformed and calculated as 1-indexed local coordinates
                    bbox_rc_local = ET.SubElement(ship, 'BoundingBox_RC_Local')
                    ET.SubElement(bbox_rc_local, 'Top').text = str(vessel_raw_line_min - min_raw_line + 1)
                    ET.SubElement(bbox_rc_local, 'Left').text = str(left)
                    ET.SubElement(bbox_rc_local, 'Bottom').text = str(vessel_raw_line_max - min_raw_line + 1)
                    ET.SubElement(bbox_rc_local, 'Right').text = str(right)
                    
                    # Create RC bounding box elements (global coordinates)
                    bbox_rc_global = ET.SubElement(ship, 'BoundingBox_RC_Global')
                    ET.SubElement(bbox_rc_global, 'Top').text = str(vessel_raw_line_min)
                    ET.SubElement(bbox_rc_global, 'Left').text = str(vessel_rc_sample_min)
                    ET.SubElement(bbox_rc_global, 'Bottom').text = str(vessel_raw_line_max)
                    ET.SubElement(bbox_rc_global, 'Right').text = str(vessel_rc_sample_max)
                    
                    updated_count += 1
                
                # Only save XML if we actually updated ships
                if updated_count > 0:
                    ET.indent(tree, space="   ", level=0)
                    tree.write(label_file, encoding='utf-8', xml_declaration=True)
                    logger.info(f"Updated {updated_count} vessel bounding boxes in {label_file}")
                else:
                    logger.debug(f"No valid ships to update in {label_file}")
            else:
                logger.debug(f"No ships found in {label_file}")

        # Build up output matrix
        raw_detection.append([
            swath,
            patch_id,
            pixel_cords.tolist(),
            [min_raw_line, max_raw_line, min_raw_sample, max_raw_sample],
            burst_time,
            min_az_mid_burst_time, 
            min_sensing_mid_burst_time,
            l1_burst_num,
        ])
    # Save output as .npy
    raw_detection_array =  np.array(raw_detection, dtype=object)
    if raw_detection_array.size != 0:
        out_fp = Path(output_file.parent, f"{product_folder.stem}.npy")
        np.save(out_fp, raw_detection_array)
        logger.info(f"Completed slc_to_raw_traceability, saved data to {out_fp}")
    else:
        logger.warning(f"Could not find matches for {product_folder.stem}!")

def get_parser(subparsers=None) -> argparse.ArgumentParser:
    """Instantiate the command line argument (sub-)parser."""
    name = PROG
    synopsis = __doc__.splitlines()[0]
    doc = __doc__

    if subparsers is None:
        parser = argparse.ArgumentParser(prog=name, description=doc)
        parser.add_argument(
            "--version",
            action="version",
            version="%(prog)s v" + VERSION
        )
    else:
        parser = subparsers.add_parser(name, description=doc, help=synopsis)


    # Command line options
    parser.add_argument(
       "-o",
       "--output_file",
       help="output file with the L0 coordinates after traceability "
       "(if not present, output file will be 'rawH.npy' in current folder)",
    )
    parser.add_argument(
       "-l",
       "--log_file",
       help="log file name (if not present, the name will be "
            "'traceability_log.txt' and located at working directory)",
    )

    # Positional arguments
    parser.add_argument("label_folder",
                        help="Folder where the patch labels (one file "
                             "per patch) are located")
    parser.add_argument("product_folder",
                        help="Folder where the L1B SLC product is located")

    return parser


def parse_args(args=None, namespace=None, parser=None):
    """Parse command line arguments."""
    if parser is None:
        parser = get_parser()

    args = parser.parse_args(args, namespace)
    return args



def main (*argv):
    """Main CLI interface."""

    use_args = False 
    logger = logging.getLogger("log")
    logger.level = logging.DEBUG
    label_folder, product_folder, products_folder, output_file, log_file = None, None, None, None, None
    file_handler = None
    if use_args:
        args = parse_args(argv if argv else None)
        label_folder = Path(args.label_folder)
        product_folder = Path(args.product_folder)
        if args.output_file is None:
            output_file = Path("rawH.npy")
        else:
            output_file = args.output_file
        if args.log_file is None:
            log_file = Path("traceability.log")
        else:
            log_file = args.log_file
    else: # use yaml
        cfg = None
        cfg_path = cfg["slc2raw"]["config_path"]
        with open(cfg_path) as cfg_file:
            cfg = yaml.safe_load(cfg_file)
        label_folder = Path(cfg["label_folder"])
        # here we can specify products
        products_folder = Path(cfg["product_folder"])
        output_file = Path(cfg["output_file"])
        log_file = Path(cfg["log_file"])
        file_handler = logging.FileHandler(Path(log_file))
        console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setLevel(logger.level)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    if products_folder is None:
        slc_to_raw_traceability(label_folder, product_folder, output_file, logger)
    else:
        product_paths = [Path(products_folder, fn) for fn in sorted(os.listdir(products_folder)) if '.zip' not in fn and Path(products_folder,fn).is_dir()]
        for product_path in product_paths:
            slc_to_raw_traceability(label_folder, product_path, output_file, logger)
if __name__ == "__main__":
    sys.exit(main())



