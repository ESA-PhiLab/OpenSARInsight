import os
import argparse
import xml.etree.ElementTree as ET
from PIL import Image

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate black masks for scenes with no RFI detected.")
    parser.add_argument('--input_dir', required=True, help='Directory containing XML files')
    parser.add_argument('--output_dir', required=True, help='Directory to save black mask PNGs')
    return parser.parse_args()

def create_black_image(output_path: str, size=(512, 512)) -> None:
    """Create and save a black PNG image at the given path."""
    image = Image.new('L', size, 0)  # Grayscale mode, 0 = black
    image.save(output_path)

def process_xml_file(xml_path: str, output_dir: str):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract Scene_ID
        scene_id_elem = root.find('.//Scene_ID')
        rfi_detection_elem = root.find('.//StatisticsReport/RFIDetection')

        if scene_id_elem is None or rfi_detection_elem is None:
            print(f"Skipping {xml_path}: Missing Scene_ID or RFIDetection.")
            return

        scene_id = scene_id_elem.text.strip()
        rfi_flag = rfi_detection_elem.text.strip()

        if rfi_flag == '0':
            output_filename = f"{scene_id}_MASK.png"
            output_path = os.path.join(output_dir, output_filename)
            create_black_image(output_path)
            print(f"Generated black mask: {output_filename}")
        else:
            print(f"RFI detected in {scene_id}, skipping the mask.")

    except Exception as e:
        print(f"Error processing {xml_path}: {e}")

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    for file_name in os.listdir(args.input_dir):
        if file_name.lower().endswith('.xml'):
            full_path = os.path.join(args.input_dir, file_name)
            process_xml_file(full_path, args.output_dir)

if __name__ == '__main__':
    main()


"""
Example usage:

python generate_black_masks.py --input_dir path/to/xml_files --output_dir path/to/output_black_masks

"""
