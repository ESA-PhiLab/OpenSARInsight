import rasterio
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.geocoding_block.geocode import Geocoder
from random import seed, choice
from os import listdir
import cProfile 
import pstats
import io 

def geocode_patch(logger, cfg_fp, patch_filename):
    logger.debug(f"Geocoding {patch_filename.split('.')[0]}")
    gocdr=Geocoder(logger, cfg_fp, patch_filename)
    geocoded_img_fp = gocdr.run()
    # if running for an already geocoded patch with overwrite = false then the patch has already been geocoded!
    if geocoded_img_fp is not None:
        assert geocoded_img_fp.exists()   

def test_geocoder_run_for_single_patch_39(logger):
    cfg_fp = Path(__file__).parent.parent.joinpath("configuration.yaml")
    geocode_patch(logger, cfg_fp, "DB_OPENSAR_RFI_39.xml")
    
def test_geocoder_run_for_single_patch_profiling(logger):
    cfg_fp = Path(__file__).parent.parent.joinpath("configuration.yaml")
    gocdr=Geocoder(logger, cfg_fp, "DB_OPENSAR_FD_4_7529.xml")
    p = cProfile.Profile()
    p.enable()
    p.runcall(gocdr.run)
    p.disable()
    s = io.StringIO()
    ps = pstats.Stats(p, stream=s).sort_stats('tottime')
    ps.print_stats()

    profiling_stats = Path(__file__).parent.parent.parent.joinpath("data","tmp","test_geocoder_run_for_single_patch_profiling.txt")
    if not profiling_stats.exists():
        with open(profiling_stats,'w+') as file:
            file.write(f"===PROFILING for test_geocoder_run_for_single_patch_profiling FUNCTION===\n")
            file.write(s.getvalue())
    else:
        with open(profiling_stats,'w+') as file:
            file.truncate(0)
            file.write(f"===PROFILING for test_geocoder_run_for_single_patch_profiling FUNCTION===\n")
            file.write(s.getvalue())


def test_geocoder_run_for_random_patch(cfg, logger):
    cfg_fp = Path(__file__).parent.parent.joinpath("configuration.yaml")
    labels_fp = cfg["labels_directory_path"]
    random_patch = choice([label for label in listdir(labels_fp)])
    geocode_patch(logger, cfg_fp, random_patch) 

def test_specific_patch(logger):
    cfg_fp = Path(__file__).parent.parent.joinpath("configuration.yaml")
    geocode_patch(logger, cfg_fp, "DB_OPENSAR_FD_7_36746.xml")

def test_specific_vessel_patch(logger):
    cfg_fp = Path(__file__).parent.parent.joinpath("configuration.yaml")
    geocode_patch(logger, cfg_fp, "DB_OPENSAR_VD_256.xml")

def test_specific_vessel_patch_2(logger):
    cfg_fp = Path(__file__).parent.parent.joinpath("configuration.yaml")
    # this patch is out of bounds in cop-glo-30 dem
    geocode_patch(logger, cfg_fp, "DB_OPENSAR_VD_3024.xml")

def test_geocoder_run_for_10_random_patches(cfg, logger):
    cfg_fp = Path(__file__).parent.parent.joinpath("configuration.yaml")
    seed_numbers = [i for i in range(1,10,1)]
    labels_fp = cfg["labels_directory_path"]
    for sn in seed_numbers:
        seed(sn)
        random_patch = choice([label for label in listdir(labels_fp)])
        geocode_patch(logger, cfg_fp, random_patch) 