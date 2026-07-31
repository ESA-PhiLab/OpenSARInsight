# Sentinel-1 Scene Downloader

A Python tool for batch downloading Sentinel-1 SAR data from the Copernicus Data Space Ecosystem (CDSE).

## Features

- Batch download of Sentinel-1 scenes from predefined lists
- Support for different scene categories (Dark Vessels, Flood Detection, RFI)
- Automatic authentication with CDSE
- Download progress tracking
- Configurable delays between downloads

## Project Structure
```
├── main.py                    # Main batch download script
├── scenes.py                  # Predefined scene lists
├── savepaths.py              # Download path configuration
├── requirements.txt           # Dependencies
├── .env                      # Credentials (create this)
└── src/sentinelHubAPI/       # Core modules
    ├── auth.py               # Authentication
    ├── query.py              # Product search
    ├── downloader.py         # File download
    ├── client.py             # High-level interface
    └── batchdownloader.py    # Batch processing
```

## Setup

### 1. Create Account
Create an account at [Copernicus Data Space](https://dataspace.copernicus.eu/) and note your credentials.

### 2. Environment Setup (Windows)
```sh
python -m venv env
./env/Scripts/activate
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Configuration
Create a `.env` file in the project root with your credentials:
```
username=your_email@example.com
password=your_password
```

### 5. Configure Download Paths
Edit `savepaths.py` to specify where files should be downloaded:
```python
# Example paths - update these to your preferred locations
dvd_path = "/your/path/to/dark_vessels/"
fld_path = "/your/path/to/flood_detection/"
rfi_path = "/your/path/to/rfi_scenes/"
```

## Usage

### Quick Start
```sh
python main.py
```

This will download all scenes from the predefined lists with 10-minute delays between categories.

### Scene Lists and Paths
The tool uses separate configuration files:
- `scenes.py`: Contains predefined scene lists for each category
  - `dvd`: Dark Vessel Detection scenes
  - `fld`: Flood Detection scenes  
  - `rfi`: Radio Frequency Interference scenes
- `savepaths.py`: Defines download directories for each category

### Custom Downloads
You can also use the individual modules for custom downloads:

```python
from src.sentinelHubAPI.client import SentinelHubClient

client = SentinelHubClient()
client.search_and_download("S1A_IW_SLC__1SDV_20200706T074222_20200706T074242_033331_03DC92_2526.SAFE", 
                          download_dir="/custom/path/")
```
