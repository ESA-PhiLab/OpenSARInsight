import os
from PIL import Image

def generate_blank_masks_(input_dir, output_dir, mask_format='png'):
    """
    Generates blank (black) mask images with the same dimensions as each TIFF file in the input directory.
    For each TIFF image found in `input_dir`, this function creates a blank (all-black) mask of the same width and height,
    and saves it to `output_dir` in the specified format.
    Args:
        input_dir (str): Path to the directory containing input TIFF images.
        output_dir (str): Path to the directory where blank mask images will be saved.
        mask_format (str, optional): File format for the output mask images (default: 'png').
    Each mask is named after the corresponding TIFF file, with '_MASK' appended to the base name.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    tiff_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tiff', '.tif'))]
    
    for tiff_file in tiff_files:
        try:
            input_path = os.path.join(input_dir, tiff_file)
            
            # Get dimensions
            with Image.open(input_path) as img:
                width, height = img.size
            
            # Generate filename
            base_name = os.path.splitext(tiff_file)[0]
            if base_name.endswith('_SLC_VH'):
                base_name = base_name[:-7]
            
            mask_filename = f"{base_name}_MASK.{mask_format}"
            mask_path = os.path.join(output_dir, mask_filename)
            
            # Create and save blank mask directly
            blank_mask = Image.new('L', (width, height), 0)
            blank_mask.save(mask_path)
            
            print(f"{tiff_file} ({width}x{height}) -> {mask_filename}")
            
        except Exception as e:
            print(f"Error processing {tiff_file}: {str(e)}")

# Invoke function

input_dir = "/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/rfi/aresys_baseline/rfi/patches/patches0_SLC_VH" # Update path as required
output_dir = "/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/rfi/aresys_baseline/rfi/masks/binary_masks_0" # Update path as required
generate_blank_masks_(input_dir, output_dir, mask_format='png')