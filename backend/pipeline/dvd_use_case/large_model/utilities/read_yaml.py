"""
---------------------------------------------------------------------
Project: OpenSAR Insight
---------------------------------------------------------------------
read_yaml.py

Tool: Read YAML configuration files

Author: Abdulhameed Yunusa (ABHY)
E-mail: ayunusa@indracompany.com
Creation Date: 2026-02-26

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""
from pathlib import Path
import yaml 

def read_yaml(file_path: Path) -> dict[str, str]:
	with open(file_path, 'r') as file:
		return yaml.safe_load(file)