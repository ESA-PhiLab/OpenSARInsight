"""
---------------------------------------------------------------------
Project: OpenSAR Insight
---------------------------------------------------------------------
generate_yolo_labels.py

Tool: Generate YOLO format labels from XML annotations for RC patches

Author: Abdulhameed Yunusa (ABHY)
E-mail: ayunusa@indracompany.com
Creation Date: 2026-02-26

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""
# type: ignore
from pathlib import Path
from typing import Optional, Literal
import shutil
from utilities.read_xml import parse_vessel_xml
from utilities.read_yaml import read_yaml
from utilities.filter_ships import filter_ships
from utilities.create_data_yaml import create_data_yaml
from utilities.convert_bbox_to_pose import bbox_to_pose_line

# Load Config File
config = read_yaml(Path('config.yaml'))

# Load all file paths
dataset_root = Path(config['datapaths']['dataset_root'])
png_rc_patches_path = Path(config['datapaths']['png_rc_patches_path'])
yolo_dataset_path = Path(config['datapaths']['yolo_dataset_filtered_rc'])



def generate_yolo_labels(
    dataset_root: Path,
    png_rc_patches_path: Path,
    yolo_dataset_path: Path,
    img_width: int = 512,
    img_height: int = 512,
    class_id: int = 0,
    distance_threshold_km: Optional[float] = None,
    distance_metric: Literal['xview', 'global', 'both'] = 'xview',
    is_vessel_filter: Optional[bool] = None,
    bbox_tag: str = 'BoundingBox_RC_Local'
) -> dict:
    """
    Generate YOLO format labels from XML annotations.

    XML annotations were produced against images of size img_width x img_height (default 512x512).
    The output PNGs may have a variable height (i.e. they are rescaled versions of the same scene),
    but the YOLO normalisation is always performed against the original annotation dimensions.
    This is correct because YOLO labels are geometry-preserving: a normalised coordinate of 0.781
    represents the same relative position regardless of the pixel dimensions of the image at
    training time.

    Args:
        dataset_root:           Root path of the unified dataset (with train/val/test subdirs
                                each containing a labels/ folder with XML files).
        png_rc_patches_path:    Path to directory containing PNG images (with train/test/val subdirs).
        yolo_dataset_path:      Output path for YOLO dataset structure.
        img_width:              Width in pixels of the images used to produce the XML annotations (default: 512).
        img_height:             Height in pixels of the images used to produce the XML annotations (default: 512).
                                Do NOT replace this with the actual PNG height — the XMLs were annotated
                                at this resolution and normalisation must be consistent with that.
        class_id:               YOLO class ID for vessels (default: 0).
        distance_threshold_km:  Maximum distance from shore in km (None = no filter).
        distance_metric:        Which distance metric to use ('xview', 'global', or 'both').
        is_vessel_filter:       Filter by is_vessel status (None = no filter, True = vessels only,
                                False = non-vessels only).

    Returns:
        dict: Statistics about generated labels.
    """
    stats = {
        'total_images': 0,
        'images_with_detections': 0,
        'total_ships_processed': 0,
        'ships_after_filtering': 0,
        'splits': {}
    }

    # Open log file for ships skipped due to NaN bounding box values
    log_file_path = Path('skipped_ships_log.txt')
    log_file = open(log_file_path, 'w')
    log_file.write("Skipped Ships Log (NaN Bounding Box Values)\n")
    log_file.write("=" * 60 + "\n\n")

    # Process each split
    for split in ['train', 'test', 'val']:
        split_png_path = png_rc_patches_path / split
        split_xml_path = dataset_root / split / 'labels'
        split_stats = {'images': 0, 'detections': 0}

        if not split_png_path.exists():
            print(f"Warning: Split directory not found: {split_png_path}")
            continue

        # Create output directories for this split
        output_images_path = yolo_dataset_path / 'images' / split
        output_labels_path = yolo_dataset_path / 'labels' / split
        output_images_path.mkdir(parents=True, exist_ok=True)
        output_labels_path.mkdir(parents=True, exist_ok=True)

        # Process each PNG image in the split
        for png_file in split_png_path.glob('*.png'):

            # Map PNG filename to its XML counterpart
            # PNG files are dual-polarisation composites (VH_VD or VV_VD);
            # XML files are named after the VD (vertical dual) stem only.
            xml_stem = png_file.stem.replace('VH_VD', 'VD').replace('VV_VD', 'VD')
            xml_file = split_xml_path / (xml_stem + '.xml')

            if not xml_file.exists():
                continue

            # Parse the XML annotation file
            try:
                xml_data = parse_vessel_xml(xml_file, bbox_tag=bbox_tag)
            except Exception as e:
                print(f"Error parsing {xml_file}: {e}")
                continue

            # Log any ships that were skipped due to NaN bounding box values
            for skipped in xml_data.get('skipped_ships', []):
                log_file.write(f"Split: {split}, Patch: {png_file.stem}, Ship: {skipped['name']}\n")

            # Skip images with no ship annotations
            if xml_data['number_of_ships'] == 0:
                continue

            stats['total_ships_processed'] += len(xml_data['ships'])

            # Apply optional filters (distance from shore, vessel classification)
            filtered_ships = filter_ships(
                xml_data['ships'],
                distance_threshold_km=distance_threshold_km,
                distance_metric=distance_metric,
                is_vessel_filter=is_vessel_filter
            )

            # Skip images where no ships pass the filter
            if not filtered_ships:
                continue

            stats['ships_after_filtering'] += len(filtered_ships)
            stats['total_images'] += 1
            split_stats['images'] += 1

            # Copy image to the YOLO dataset directory (skip if already present)
            output_image_file = output_images_path / png_file.name
            if not output_image_file.exists():
                shutil.copy2(png_file, output_image_file)

            # Write YOLO-pose format label file
            label_file = output_labels_path / (png_file.stem + '.txt')
            pose_lines = [
                bbox_to_pose_line(ship['bbox'], class_id, img_width, img_height)
                for ship in filtered_ships
            ]
            label_file.write_text('\n'.join(pose_lines))
            stats['images_with_detections'] += 1
            split_stats['detections'] += 1

        stats['splits'][split] = split_stats
        print(f"Processed {split}: {split_stats['images']} images with {split_stats['detections']} detections")

    log_file.close()
    print(f"Skipped ships log saved to: {log_file_path}")

    # Write the YOLO data.yaml file
    create_data_yaml(yolo_dataset_path)

    return stats


if __name__ == "__main__":
    label_settings = config.get('label_settings', {})
    IMG_WIDTH              = label_settings.get('img_width', 512)
    IMG_HEIGHT             = label_settings.get('img_height', 512)
    DISTANCE_THRESHOLD_KM  = label_settings.get('distance_threshold_km', None)
    DISTANCE_METRIC        = label_settings.get('distance_metric', 'xview')
    IS_VESSEL_FILTER       = label_settings.get('is_vessel_filter', None)

    print("=" * 60)
    print("YOLO Label Generator")
    print("=" * 60)
    print(f"Dataset Root:         {dataset_root}")
    print(f"PNG Patches Path:     {png_rc_patches_path}")
    print(f"YOLO Dataset Path:    {yolo_dataset_path}")
    print(f"Annotation Size:      {IMG_WIDTH} x {IMG_HEIGHT} px")
    print(f"Distance Filter:      {f'{DISTANCE_THRESHOLD_KM} km' if DISTANCE_THRESHOLD_KM else 'None (all)'}")
    print(f"Distance Metric:      {DISTANCE_METRIC}")
    print(f"Vessel Filter:        {IS_VESSEL_FILTER if IS_VESSEL_FILTER is not None else 'None (all)'}")
    print("=" * 60)

    stats = generate_yolo_labels(
        dataset_root=dataset_root,
        png_rc_patches_path=png_rc_patches_path,
        yolo_dataset_path=yolo_dataset_path,
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
        distance_threshold_km=DISTANCE_THRESHOLD_KM,
        distance_metric=DISTANCE_METRIC,
        is_vessel_filter=IS_VESSEL_FILTER
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total images included:  {stats['total_images']}")
    print(f"Images with detections: {stats['images_with_detections']}")
    print(f"Total ships in XML:     {stats['total_ships_processed']}")
    print(f"Ships after filtering:  {stats['ships_after_filtering']}")
    print("=" * 60)