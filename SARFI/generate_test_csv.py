from logging import Logger
import os
import pandas as pd
from pathlib import Path 
from random import choice 
from xmltodict import parse as xml_parse
import yaml
from typing import Union
from utils import SARFILogger

def generate_test_csv(products_dir_path: str, output_csv_path: str, num_prods: int = 10, logger: Logger = None):
    prod_fps = set([Path(products_dir_path, dn) for dn in os.listdir(products_dir_path) if Path(products_dir_path, dn).is_dir()])
    cand_prod_fps = set()
    # generate candidate SLC products
    for _ in range(num_prods):
        prod_pick = choice(list(prod_fps - cand_prod_fps))
        cand_prod_fps.add(prod_pick)
    # next we collate all the annotation files
    annotation_fps = list() 
    for prod in cand_prod_fps:
        annotation_fps.extend(prod.glob(f"**/annotation/*.xml"))
    # generate candidate SLC annotation files
    cand_ant_fps = set()
    ant_fps = set(annotation_fps)
    for _ in range(num_prods):
        ant_pick = choice(list(ant_fps - cand_ant_fps))
        cand_ant_fps.add(ant_pick)
    # now we get the timestamps and random latitude and longitude values for the candidate annotation files
    metadata_list = list()
    for ant_fp in cand_ant_fps:
        with ant_fp.open() as f:
            metadata = xml_parse(f.read()) 
        mdst = metadata['product']['adsHeader']['startTime']
        mdet = metadata['product']['adsHeader']['stopTime']
        geolocs = metadata['product']['geolocationGrid']['geolocationGridPointList']['geolocationGridPoint']
        random_geoloc = choice(geolocs)
        lat = random_geoloc['latitude']
        lon = random_geoloc['longitude']
        metadata_list.append(dict({
            # the filepaths are not actually required for the slc matching but it's useful to include it to verify that the matches
            # found from the script correspond to the correct annotation files 
            "filepath": str(ant_fp),
            "start_time": mdst,
            "stop_time": mdet,
            "latitude": lat,
            "longitude": lon
        }))
    # finally we write the metadata to a csv file
    pd.DataFrame(metadata_list).to_csv(output_csv_path, index=False)
    # check that the csv file was created successfully
    df = pd.read_csv(output_csv_path)
    assert len(df) == len(metadata_list), "The number of rows in the CSV file does not match the number of metadata entries."
    if logger:
        logger.info(f"CSV file created successfully at {output_csv_path}")
    else:
        print(f"CSV file created successfully at {output_csv_path}")

def parse_yaml_config(yaml_fp:Union[str, Path]) -> dict:
    with open(yaml_fp, "r") as f:
        cfg = yaml.safe_load(f)
    required_keys = ['test_csv_fp', 'safe_dir_fp', 'test_num_prods']
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing required key '{key}' in YAML configuration.")
    return cfg

def run_generate_test_csv():
    logger = SARFILogger().logger
    yaml_cfg_fp = Path(__file__).parent.joinpath("configuration", "slc_matcher_config.yaml")
    cfg = parse_yaml_config(yaml_cfg_fp)
    test_csv_fp = cfg['test_csv_fp']
    safe_dir_fp = cfg['safe_dir_fp']
    test_num_prods = cfg['test_num_prods']
    logger.debug("Parsed YAML configuration successfully.")
    generate_test_csv(safe_dir_fp, test_csv_fp, test_num_prods, logger)

if __name__ == "__main__":
    run_generate_test_csv()