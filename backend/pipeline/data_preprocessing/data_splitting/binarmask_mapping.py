from pathlib import Path
import re
import shutil
from dataset_generation_scripts.utils import get_config
cfg = get_config("DATA_PREPROC_PATH")

FOLDER1 = Path(cfg["data_preprocessing"]["folder1"])  
FOLDER2 = Path(cfg["data_preprocessing"]["folder2"])   
OUTPUT_FOLDER = Path(cfg["data_preprocessing"]["output_folder"]) 


def extract_id_from_folder1(name: str):
    """
    Expect patterns like: DB_OPENSAR_FD_2_97_mask.png
    We want the '97' before '_mask'.
    """
    m = re.search(r"_(\d+)_mask\.png$", name)
    return m.group(1) if m else None


def extract_id_from_folder2(name: str):
    """
    Expect patterns like: ..._FD_97_scaled_mask.png
    We want the '97' before '_scaled'.
    """
    m = re.search(r"_(\d+)_scaled_mask\.png$", name)
    return m.group(1) if m else None


def make_vh_vv_names(template_name: str):
    """
    Given one template name containing '-vh-' or '-vv-',
    return a (vh_name, vv_name) pair.
    """
    if "-vh-" in template_name:
        vh_name = template_name
        vv_name = template_name.replace("-vh-", "-vv-", 1)
    elif "-vv-" in template_name:
        vv_name = template_name
        vh_name = template_name.replace("-vv-", "-vh-", 1)
    else:
        # No polarization token: just return the same name twice
        # (or you could raise an error if you prefer)
        vh_name = template_name
        vv_name = template_name
    return vh_name, vv_name


def main():
    # Map: ID -> one template filename from folder2
    id_to_template = {}

    for png2 in FOLDER2.glob("*.png"):
        ID = extract_id_from_folder2(png2.name)
        if ID is None:
            continue
        # Keep the first one we see – enough to build vh & vv names
        if ID not in id_to_template:
            id_to_template[ID] = png2.name

    if not id_to_template:
        print(f"No matching '_scaled_mask' PNGs found in {FOLDER2}")
        return

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    matched = 0
    for png1 in FOLDER1.glob("*.png"):
        ID = extract_id_from_folder1(png1.name)
        if ID is None:
            continue

        if ID not in id_to_template:
            continue

        template = id_to_template[ID]
        vh_name, vv_name = make_vh_vv_names(template)

        # Full destination paths
        dest_vh = OUTPUT_FOLDER / vh_name
        dest_vv = OUTPUT_FOLDER / vv_name

        # Copy the same binary mask content for both polarizations
        shutil.copy2(png1, dest_vh)
        shutil.copy2(png1, dest_vv)

        matched += 1
        print(f"ID={ID}: {png1.name} -> {dest_vh.name}, {dest_vv.name}")

    print(f"Done. Total IDs matched (with vh & vv created): {matched}")


if __name__ == "__main__":
    main()
