import os
import re
from pathlib import Path
from dataset_generation_scripts.utils import get_config
cfg = get_config("DATA_PREPROC_PATH")

# ==== EDIT THESE TWO PATHS ====
NPY_FOLDER = Path(cfg["rfi_mask_copying"]["npy_folder"])   # folder with *.npy files
PNG_FOLDER = Path(cfg["rfi_mask_copying"]["png_folder"])   # folder with DB_OPENSAR_RFI_11_MASK.png
# ==============================

def extract_rfi_id(filename: str):
    """
    Extract the ID that comes after 'RFI_' in a filename.
    Example:
        's1a-iw-raw...RFI_11_scaled.npy'     -> '11'
        'DB_OPENSAR_RFI_11_MASK.png'        -> '11'
    """
    m = re.search(r"RFI_(\d+)", filename)
    return m.group(1) if m else None

def main():
    # 1. Collect all RFI IDs that exist as PNGs
    png_ids = set()
    for png_path in PNG_FOLDER.glob("*.png"):
        rfi_id = extract_rfi_id(png_path.name)
        if rfi_id is not None:
            png_ids.add(rfi_id)
        else:
            print(f"[WARN] Could not extract RFI id from PNG file: {png_path.name}")

    print(f"Found {len(png_ids)} unique RFI IDs in PNG folder")

    # 2. For every NPY file, check if there is a matching PNG ID; if not, delete the NPY
    deleted = 0
    kept = 0

    for npy_path in NPY_FOLDER.glob("*.npy"):
        rfi_id = extract_rfi_id(npy_path.name)
        if rfi_id is None:
            print(f"[WARN] Could not extract RFI id from NPY file (skipping): {npy_path.name}")
            continue

        if rfi_id in png_ids:
            kept += 1
        else:
            print(f"[DELETE] No PNG for RFI_{rfi_id}, removing NPY: {npy_path.name}")
            os.remove(npy_path)
            deleted += 1

    print(f"Kept {kept} NPY files")
    print(f"Deleted {deleted} NPY files without matching PNG")

if __name__ == "__main__":
    main()
