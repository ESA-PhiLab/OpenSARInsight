# L0 Amplitude and Phase Processing

A Python module for generating amplitude and phase information from complex IQ components in Level-0 SAR (Synthetic Aperture Radar) patches.

## Features

- **L0 Data Reading**: Read Level-0 SAR patches from binary files with proper header parsing
- **Amplitude/Phase Computation**: Extract amplitude and phase from complex SAR data
- **Flexible Output**: Support for both radians and degrees for phase output
- **Config Management**: YAML configuration for easy parameter management

## Installation

### Prerequisites

- Python 3.7 or higher
- NumPy >= 1.21.0
- PyYAML
- Matplotlib (for visualization)

### Install from Source

1. Clone or download this repository
2. Navigate to the project directory
3. Install in development mode:

```bash
pip install -e .
```

### Install Dependencies

```bash
pip install numpy pyyaml matplotlib
```

## Project Structure

```
l0_amplitude_and_phase/
├── src/
│   └── l0_processing/
│       ├── __init__.py
│       ├── read_l0_patch.py    # L0 SAR patch reader
│       ├── processing.py       # Amplitude/phase computation
│       └── utils.py           # Utility functions
├── tests/
│   ├── test_read_l0_patch.py  # Unit tests
│   └── test_processing.py     # Processing tests
├── sample_data/               # Sample SAR patch file
├── config.yaml               # Configuration file
├── example.py                # Usage example
├── setup.py                  # Package setup
└── README.md                 # This file
```

## Quick Start

### Basic Usage

```python
from l0_processing.read_l0_patch import read_l0_patch
from l0_processing.processing import get_amplitude_phase

# Read L0 SAR patch
sar_data = read_l0_patch('path/to/your/sardata.dat')

# Compute amplitude and phase
amplitude, phase = get_amplitude_phase(sar_data, return_degrees=True)

print(f"Data shape: {sar_data.shape}")
print(f"Amplitude range: [{amplitude.min():.2f}, {amplitude.max():.2f}]")
print(f"Phase range: [{phase.min():.2f}, {phase.max():.2f}] degrees")
```

### Run the Example

The project includes a complete example that demonstrates reading SAR data and creating visualizations:

```bash
python example.py
```

This will:
- Read the sample SAR data specified in `config.yaml`
- Compute amplitude and phase
- Generate a visualization saved as `sar_amplitude_phase.png`
- Display processing statistics

### Configuration

Edit `config.yaml` to specify your data files:

```yaml
L0_patch:
  sample_data: 'sample_data/your_sar_file.dat'
```

## API Reference

### `read_l0_patch(file_name: str)`

Reads a Level-0 SAR patch from a binary file.

**Parameters:**
- `file_name`: Path to the L0 SAR data file

**Returns:**
- `numpy.ndarray`: Complex-valued matrix of shape (rows, cols)

**File Format:**
- Little Endian binary format
- Header: 8 bytes (2 × uint32 for rows, cols)
- Data: Interleaved real/imaginary float32 values

### `get_amplitude_phase(sar_patch, return_degrees=False)`

Computes amplitude and phase from complex SAR data.

**Parameters:**
- `sar_patch`: Complex-valued NumPy array
- `return_degrees`: If True, return phase in degrees; otherwise radians

**Returns:**
- `amplitude`: Magnitude of the complex data
- `phase`: Phase of the complex data

## Data Format

The L0 SAR data files are expected to be in binary format with:

1. **Header** (8 bytes):
   - `rows` (4 bytes, uint32, little endian)
   - `cols` (4 bytes, uint32, little endian)

2. **Data** (rows × cols × 8 bytes):
   - Interleaved real and imaginary parts
   - Each component is float32, little endian
   - Format: `real₀ imag₀ real₁ imag₁ ...`

## Testing

Run the test suite to verify functionality:

```bash
python -m unittest discover tests/ -v
```

### Test Coverage

- L0 patch reading functionality
- Amplitude and phase computation
- Error handling for invalid inputs
- Data integrity verification

## Example Output

When running the example, you should see output like:

```
Starting L0 SAR patch processing example...
Reading L0 patch from: sample_data/s1a-iw-raw-s-vh-*.dat
Successfully read SAR data with shape: (674, 2879)
Data type: complex64
Computing amplitude and phase...
Amplitude range: [0.00, 51.50]
Phase range: [-172.16, 180.00] degrees
Creating visualization...
Plot saved as: sar_amplitude_phase.png
Example completed successfully!
```