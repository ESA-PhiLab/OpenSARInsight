from pathlib import Path
import yaml 
import os 
from typing import Dict 

def get_config(env_name: str) -> Dict:
    cfg_path = os.environ.get(env_name)

    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"Configuration file not found at: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    return cfg