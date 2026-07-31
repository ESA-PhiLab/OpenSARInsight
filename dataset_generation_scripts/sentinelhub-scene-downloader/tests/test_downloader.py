import os
import sys
from dotenv import dotenv_values  # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sentinelHubAPI import CopernicusDataSpace

# Load config
# Determine the correct path to .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
env_path = os.path.join(project_root, ".env")

config = dotenv_values(env_path)  # type: ignore
print(f"Config loaded: {config}")
client = CopernicusDataSpace(
    base_url=config['base_url'],  # type: ignore
    username=config['username'],   # type: ignore
    password=config['password']  # type: ignore
)

# Search and download
client.search_and_download("S1A_IW_SLC__1SDV_20200407T052956_20200407T053026_032017_03B2E2_18FA.SAFE")
