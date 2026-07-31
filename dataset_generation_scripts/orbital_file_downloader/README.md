# Sentinel-1 Orbit Files Downloader

A simple tool to download orbit files for Sentinel-1 SAR scenes.


## Set Up

Make sure you have the required dependencies installed.  
You may need to install the `s1_orbits` module if it's not already available.

## Usage

### As a Python Module [in script.py]

```python
import s1_orbits

# Download a single orbit file
s1_orbits.fetch_for_scene("scene_name", "storage_path")

# Download multiple orbit files for a list of scenes
orbit_files = [
    s1_orbits.fetch_for_scene(scene, "/path/to/orbit_files/")
    for scene in scene_list
]
```

### From the Command Line

```bash
python main.py
```

This will download orbit files for the scenes specified in your configuration and print the results.

## Project Structure

- `main.py` &mdash; Entry point for downloading orbit files.
- `scenes/` &mdash; Directory containing scene lists for specific categories.
  - `dvd.py` &mdash; Lists of DVD scenes.
  - `flood.py` &mdash; Lists of flood scenes.
  - `rfi.py` &mdash; Lists of RFI scenes.

## Dependencies

- `s1_orbits` &mdash; External module for fetching orbit files.

---