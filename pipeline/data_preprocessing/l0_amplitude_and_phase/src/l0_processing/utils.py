"""
Project: OpenSAR Insight
Customer: ESA

File: utils.py

Description:
    Utility functions for SAR processing workflows.

Functions:
    read_config: Read and parse YAML configuration files
    
Author : Abdulhameed Yunusa (AHY)
Email: ayunusa@indracompany.com
Date: 2025-07-22

© Copyright INDRA DEIMOS, 2025. All rights reserved.
"""

import yaml
from typing import Any


def read_config(config_file: str) -> dict[str, Any]:
    """
    Read a YAML configuration file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict[str, Any]: Parsed configuration as a dictionary.
    """
    with open(config_file, 'r') as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config
