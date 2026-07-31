"""
Defines global test fixtures see https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""
import pytest
from pathlib import Path 
import yaml 
import logging 
import os
import pytest 

@pytest.fixture
def logger():
    """
    NOTE: log formatting is not reflected in pytest command only in log file;
    see https://docs.pytest.org/en/stable/how-to/logging.html#live-logs on how to change the formatting
    """
    log_level = logging.DEBUG
    log_path = Path(__file__).parent.parent.joinpath("data","tmp","tests.log")
    if not log_path.exists():
        os.makedirs(log_path.parent)
    logger = logging.getLogger("tests")
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    if len(logger.handlers) == 0:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(log_level)
    return logger 

@pytest.fixture
def cfg():
    cfg = None
    cfg_fp = Path(__file__).parent.joinpath("configuration.yaml")
    with open(cfg_fp) as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    return cfg
