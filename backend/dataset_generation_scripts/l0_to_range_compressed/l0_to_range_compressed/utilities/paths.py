from pathlib import Path
from dataset_generation_scripts.utils import get_config

cfg = get_config("DGS_CONFIG_PATH")

splits = cfg["l0_to_range_compressed"]["splits"]
# Vessel Paths
vessel_data_headers_dir = Path(cfg["l0_to_range_compressed"]["vessel_data_headers_dir"])
rfi_data_headers_dir = Path(cfg["l0_to_range_compressed"]["rfi_data_headers_dir"])
flood_data_headers_dir = Path(cfg["l0_to_range_compressed"]["flood_data_headers_dir"])

# test compression scene: S1A_IW_RAW__0SDV_20200928T171756_20200928T171831_034562_0405EA_2A56.SAFE
test_compression_dict = cfg["l0_to_range_compressed"]["test_compression_dict"]

#'patch', 'header', 'cal_data', 'cal_header
test_matching_list = [
    cfg["l0_to_range_compressed"]["test_matching_list"]["test_1"],
    cfg["l0_to_range_compressed"]["test_matching_list"]["test_2"],
]