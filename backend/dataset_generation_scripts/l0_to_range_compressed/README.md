
# L0 to Range Compressed SAR Data Pipeline

Python package for processing Sentinel-1 Level-0 (L0) SAR data: matching, range compression, and rescaling.

## Features
- Match L0 patches to headers/calibration files
- Range compress and rescale data
- Process by split/dataset, custom directories, or direct file lists

## Installation

Clone the repo and install package:
```bash
pip install -e .
```

## Usage Examples

## Viewing Function Help & Docstrings

After importing a function, you can view its documentation in Python using:

```python
from l0_to_range_compressed import compress_multiple
help(compress_multiple)  # Shows the docstring and usage
print(compress_multiple.__doc__)  # Prints the docstring only
```

### 1. Compress a Single Patch
```python
from l0_to_range_compressed import compress
compress(
	patch_path, header_path, cal_data_path, cal_header_path, rescaled_output_dir
)
```

### 2. Batch Processing with File List
```python
from l0_to_range_compressed import compress_multiple
matches = [
	{
		'patch': '/path/to/patch.dat',
		'header': '/path/to/header.npy',
		'cal_data': '/path/to/cal_data.npy',
		'cal_header': '/path/to/cal_header.npy',
	},
	# ... more dicts ...
]
compress_multiple(matches, rescaled_output_dir)
```

### 3. Batch Processing by Split/Dataset (Flexible)
```python
from l0_to_range_compressed import compress_split
# Use built-in paths (see paths.py)
compress_split('train', 'vessel', '/output/dir')
# Or use custom directories:
compress_split(
	'train', 'vessel', '/output/dir',
	data_dir='/custom/patches',
	headers_dir='/custom/headers'
)
```

## Output Structure
```
output_dir/
├── s1a-iw-raw-s-vh-...-IW1-VD_1778_rescaled.npy
└── ...
```

## Parameters
- Rescaled size: 512x512 (default)
- Window size: 8 (default)
- Window type: 'hamming' (default)

## Customization
- Edit `paths.py` to set default directories for each dataset
- Use `data_dir` and `headers_dir` in `compress_split` for custom data

## Matching Logic
- Patch to SAFE directory: by timestamp, orbit, mission data take ID
- Header: by swath and polarization
- Calibration: CAL1-IW1 preferred, fallback to CAL2/CAL3