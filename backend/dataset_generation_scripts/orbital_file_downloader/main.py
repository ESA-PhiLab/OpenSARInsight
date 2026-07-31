import pprint

import s1_orbits  # type: ignore
from .scenes.rfi import rfi_l0, rfi_l1  # List of RFI Scenes for L0 and L1
from dataset_generation_scripts.utils import get_config
cfg = get_config("DGS_CONFIG_PATH")

RFI_L1_ORBIT_DIR = (
    cfg["orbital_file_downloader"]["rfi_l1_orbit_dir"]
)
RFI_L0_ORBIT_DIR = (
    cfg["orbital_file_downloader"]["rfi_l0_orbit_dir"]
)

# Fetch orbit files for RFI scenes in L1
orbit_files_rfi_l1: list[str] = [
    s1_orbits.fetch_for_scene(rf, RFI_L1_ORBIT_DIR) for rf in rfi_l1  # type: ignore
]
pprint.pp(orbit_files_rfi_l1)

# Fetch orbit files for RFI scenes in L0
orbit_files_rfi_l0: list[str] = [
    s1_orbits.fetch_for_scene(rf, RFI_L0_ORBIT_DIR) for rf in rfi_l0  # type: ignore
]
pprint.pp(orbit_files_rfi_l0)