"""Download paths for scene categories."""
from dataset_generation_scripts.utils import get_config
cfg = get_config("DGS_CONFIG_PATH")

# Base Path
BASE_PATH = cfg["sentinel_scene_downloader"]["base_path"]

# Use Cases
DVD_DIR = "vessels/"
RFI_DIR = "rfi/"
FLOOD_DIR = "flood/"

# Download directories
dvd_path = BASE_PATH + DVD_DIR + "full_raw_scenes"
rfi_path = BASE_PATH + RFI_DIR + "aresys_full/" + "full_raw_scenes"
fld_path = BASE_PATH + FLOOD_DIR + "full_raw_scenes"