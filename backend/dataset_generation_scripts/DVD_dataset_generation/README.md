# Generate Dataset Pipeline

## Overview

`generate_dataset.py` is an automated pipeline that orchestrates three processing steps to generate a complete dataset from Sentinel-1 SAR products:

1. **DVD Processing** - Runs Dark Vessel Detection to generate labels and SLC/GRD patches
2. **SLC to RAW Conversion** - Converts SLC patches to RAW format using `slc2raw`
3. **L0 Patching** - Extracts corresponding L0 RAW patches using `L0Patcher`

## Prerequisites

### Required Tools

Ensure the following command-line tools are available on your `PATH`:

- `slc2raw` - SLC to RAW converter
- `L0Patcher` - L0 RAW patch extractor
- `L0decoder` - L0 RAW product decoder (required to prepare L0 products)
- `python` (Python 3.9+)

### Required Python Packages

```bash
pip install pyyaml numpy pandas tifffile rasterio pillow pyproj scipy
```

### Directory Structure

```
DVD_Codes_Python/
├── config.yaml              # Main configuration file
├── mapping.csv              # Scene mapping (num;slc;grd;ocn;raw)
├── run.py                   # DVD processing script (robust version)
├── main.py                  # DVD core processing module
├── generate_dataset.py      # Pipeline orchestration script
└── inputs/
    ├── slc/                 # SLC products (*.SAFE)
    ├── grd/                 # GRD products (*.SAFE)
    ├── ocn/                 # OCN products (*.SAFE - optional)
    └── raw/                 # L0 decoded products (*.SAFE - decoded using L0decoder)
```

**Important:** L0 RAW products must be decoded using the `L0decoder` package before placing them in the `raw/` directory. The pipeline expects decoded `.SAFE` folders.

## L0 Product Preparation

Before running the pipeline, L0 RAW products must be decoded:

```bash
# Example: Decode L0 product using L0decoder
L0decoder \
  --input S1A_IW_RAW__0SDV_20210101T000000_20210101T000030_036001_043A01_0001.SAFE \
  --output inputs/raw/

# Verify decoded product exists
ls inputs/raw/S1A_IW_RAW__0SDV_20210101T000000_20210101T000030_036001_043A01_0001.SAFE
```

**L0decoder Documentation:** Refer to the L0decoder package documentation for detailed decoding instructions and parameters.

## Configuration

### 1. Configure `config.yaml`

```yaml
input:
  input_root: "/path/to/DVD_Codes_Python/inputs"
  slc_dirname: "slc"                    # Subdirectory for SLC products
  grd_dirname: "grd"                    # Subdirectory for GRD products
  ocn_dirname: "ocn"                    # Subdirectory for OCN products (optional)
  raw_dirname: "raw"                    # Subdirectory for L0 RAW decoded products
  
  scene_mapping: "scene_mapping.csv"   # Path to scene mapping CSV
  scenes_to_process: [1, 2, "5-10"]    # Scenes to process (or null for all)

output:
  output_root: "/path/to/output"
```

**Key Features:**
- ✅ Supports relative paths (resolved from script directory)
- ✅ Subdirectory names are configurable
- ✅ Scene ranges: `"5-10"` expands to `[5, 6, 7, 8, 9, 10]`
- ✅ Mixed formats: `[1, "3-5", 10]` is valid

### 2. Prepare `scene_mapping.csv`

The CSV must have these columns (semicolon-separated):

```csv
num;slc;grd;ocn;raw
1;S1A_IW_SLC__1SDV_20210101T000000_20210101T000030_036001_043A01_0001;S1A_IW_GRDH_1SDV_20210101T000000_20210101T000030_036001_043A01_0002;S1A_IW_OCN__2SDV_20210101T000000_20210101T000030_036001_043A01_0003;S1A_IW_RAW__0SDV_20210101T000000_20210101T000030_036001_043A01_0004
2;S1B_IW_SLC__1SDV_20210102T120000_20210102T120030_036002_043A02_0001;S1B_IW_GRDH_1SDV_20210102T120000_20210102T120030_036002_043A02_0002;S1B_IW_OCN__2SDV_20210102T120000_20210102T120030_036002_043A02_0003;S1B_IW_RAW__0SDV_20210102T120000_20210102T120030_036002_043A02_0004
```

**Important Notes:**
- Product names can be with or without `.SAFE` suffix
- The script automatically handles `.SAFE` normalization
- Supports UTF-8 with BOM (`encoding='utf-8-sig'`)
- Missing `raw` column: scene will be skipped in step 3
- `ocn` column is optional (can be empty)

### 3. Edit User Settings in `generate_dataset.py`

```python
# Paths to executables (if not in PATH)
SLC2RAW_EXE = "slc2raw"           # Or full path: "/opt/tools/slc2raw"
L0PATCHER_EXE = "L0Patcher"       # Or full path: "/opt/tools/L0Patcher"

# Optional: Override output root from config.yaml
OVERRIDE_OUTPUT_ROOT = None       # Set to Path("/custom/path") to override
```

## Usage

### Basic Usage

```bash
# Process all scenes from mapping.csv
python generate_dataset.py

# Use custom config file
python generate_dataset.py --config custom_config.yaml
```

### Process Specific Scenes

Edit `config.yaml`:

```yaml
input:
  scenes_to_process: [1, 2, 5]  # Process only scenes 1, 2, and 5
```

Or use ranges:

```yaml
input:
  scenes_to_process: ["1-5", 10, "15-20"]  # Scenes 1-5, 10, and 15-20
```

### Process All Scenes

```yaml
input:
  scenes_to_process: null  # Process all scenes in mapping.csv
```

### Command-Line Arguments

```bash
python generate_dataset.py --help

Options:
  --config PATH      Path to config YAML (default: config.yaml)
```

## Output Structure

For each scene, the output directory contains:

```
output/
└── scene_001/
    ├── 01_step1_DONE.txt           # Step 1 completion marker
    ├── 01_step1_run.log            # Step 1 execution log (DVD processing)
    ├── label/                      # DVD-generated labels
    │   ├── DB_OPENSAR_DVD_001.xml
    │   ├── DB_OPENSAR_DVD_002.xml
    │   └── legend_XXXX.txt         # CSV: patch_id,has_vessel,swath_id
    ├── slc/                        # DVD-generated SLC patches (complex64 TIFF)
    │   ├── DB_OPENSAR_DVD_001_SLC_VV.tiff
    │   ├── DB_OPENSAR_DVD_001_SLC_VH.tiff
    │   └── ...
    ├── grd/                        # DVD-generated GRD patches (uint8/uint16 TIFF)
    │   ├── DB_OPENSAR_DVD_001_GRD_VV.tiff
    │   ├── DB_OPENSAR_DVD_001_GRD_VH.tiff
    │   └── ...
    ├── 02_slc2raw/
    │   ├── _DONE.txt               # Step 2 completion marker
    │   ├── 02_step2_driver.log     # Step 2 execution log
    │   ├── traceability_log.txt    # slc2raw detailed log
    │   └── rawH.npy                # Patch info for L0Patcher (numpy array)
    └── 03_raw_patches/
        ├── _DONE.txt               # Step 3 completion marker
        ├── 03_step3_driver.log     # Step 3 execution log
        ├── Logfile.log             # L0Patcher detailed log
        ├── Patch_001.dat           # L0 RAW patch binary data
        ├── Patch_001.xml           # L0 RAW patch metadata
        ├── Patch_002.dat
        ├── Patch_002.xml
        └── ...
```

**Note:** The `Patch_XXX.dat` files are the actual L0 RAW data files. The basename "Patch" is configurable via `L0PATCH_DATA_BASENAME` in `generate_dataset.py`.

### Legend File Format

`legend_XXXX.txt` is a CSV with vessel detection results:

```csv
patch_id,has_vessel,swath_id
1,0,1
2,1,2
3,0,3
```

- `patch_id`: Sequential patch number
- `has_vessel`: 1 if vessel detected, 0 otherwise
- `swath_id`: Sub-swath identifier (1, 2, or 3 for IW mode)

## Features

### Idempotent Processing

- Each step creates a `_DONE.txt` marker upon completion
- Re-running the script skips already-completed steps
- Safe to interrupt and resume at any time
- Marker format: `success,<timestamp>`

### Enhanced Product Discovery

```
[VALIDATE] Checking input directories...
  ✓ SLC         : /path/to/inputs/slc (SAFE folders)
  ✓ GRD         : /path/to/inputs/grd (SAFE folders)
  ✓ RAW         : /path/to/inputs/raw (SAFE folders)
```

- Automatically finds products in subdirectories
- Handles products with/without `.SAFE` extension
- Warns about multiple matches (chooses shortest)
- Clear validation before processing starts

### Partial Success Handling

If `L0Patcher` fails but produces some patches:
- Creates `_PARTIAL_SUCCESS.txt` with error details:
  ```
  exit_code: 1
  patches_created: 42
  stderr: <error messages>
  ```
- Marks step as `partial` in `_DONE.txt`
- Considers the scene partially complete (won't rerun)
- Summary reports partial successes separately

### Automatic Cleanup

- Temporary config files deleted after step 1
- Clean folder structure maintained
- Log files preserved for debugging

### Comprehensive Validation

The script validates:
- ✅ Required tools are on PATH (`slc2raw`, `L0Patcher`)
- ✅ Configuration files exist (`config.yaml`, `mapping.csv`)
- ✅ Input directories exist (slc/, grd/, raw/)
- ✅ Input products exist before processing each scene
- ✅ L0 RAW products are decoded (`.SAFE` folders, not `.zip`)
- ✅ Required columns in mapping CSV
- ✅ Output folders created before writing
- ✅ Scene numbers are integers

### Detailed Progress Tracking

```
===========================================================================
PROCESSING SCENES
===========================================================================

[SCENE 001] (1/10)
  SLC: S1A_IW_SLC__1SDV_20210101T000000_...
  GRD: S1A_IW_GRDH_1SDV_20210101T000000_...
  RAW: S1A_IW_RAW__0SDV_20210101T000000_...
  Output: /path/to/output/scene_001

  ✓ SLC: S1A_IW_SLC__1SDV_20210101T000000_....SAFE
  ✓ GRD: S1A_IW_GRDH_1SDV_20210101T000000_...SAFE
  ✓ RAW: S1A_IW_RAW__0SDV_20210101T000000_...SAFE

[DVD] Processing scene 001...
[DVD] step1 scene 001... ✓ SUCCESS (32.5s)
[SLC2RAW] step2 scene 001... ✓ SUCCESS (5.2s)
[L0PATCH] step3 scene 001... ✓ SUCCESS (8.7s)
  ✓ SUCCESS
```

## Error Handling

### Tool Not Found

```
[FATAL] Tool 'slc2raw' not found on PATH
```

**Solutions:**
1. Install the tool
2. Add to PATH: `export PATH="/path/to/tool:$PATH"`
3. Set full path in `generate_dataset.py`: `SLC2RAW_EXE = "/opt/tools/slc2raw"`

### L0 Product Not Decoded

```
[ERROR] RAW product is not decoded (found .zip instead of .SAFE)
```

**Solution:** Decode L0 products using L0decoder:

```bash
# Decode the L0 RAW product
L0decoder --input S1A_IW_RAW__*.zip --output inputs/raw/

# Verify decoded product
ls -la inputs/raw/*.SAFE
```

### Product Not Found

```
[SCENE 003] (3/10)
  ✗ SLC SAFE not found in /path/to/inputs/slc
  ✗ SKIPPED (missing: SLC)
```

**Solutions:**
1. Verify product exists: `ls /path/to/inputs/slc/*.SAFE`
2. Check product name in `mapping.csv` matches filesystem
3. Ensure `.SAFE` folder is properly extracted (not `.zip`)

### Configuration Error

```
[FATAL] Missing 'input.input_root' in config
```

**Solution:** Add required key to `config.yaml`:

```yaml
input:
  input_root: "/path/to/inputs"
```

### Step Failure

```
[FAIL] step2 scene 001 (exit=1)
       Log: /path/to/output/scene_001/02_slc2raw/02_step2_driver.log
```

**Solutions:**
1. Check detailed log: `cat /path/.../02_step2_driver.log`
2. Check tool's own log: `cat /path/.../traceability_log.txt`
3. Verify input files from step 1 are valid
4. Check disk space and permissions

### Mapping CSV Errors

```
[FATAL] Mapping CSV missing required columns: ['raw']
```

**Solution:** Add missing columns to `mapping.csv`:

```csv
num;slc;grd;ocn;raw
1;product1;product2;product3;product4
```

## Processing Summary

At the end of execution:

```
===========================================================================
PROCESSING SUMMARY
===========================================================================
Total scenes   : 10
Successful     : 7
Failed         : 2
Partial        : 1

Failed scenes:
  - Scene 003: Missing SLC
  - Scene 007: step2 failed

Partial scenes:
  - Scene 009: step3 partial (42 patches created)

[DONE] Processing complete
```

**Exit Codes:**
- `0`: All scenes processed successfully
- `1`: One or more scenes failed (see summary)

## Troubleshooting

### Scene Skipped

```
[WARN] Scene 005 has no RAW name. Skipping.
```

**Solution:** Add the `raw` column value in `mapping.csv` for scene 5.

### L0decoder Not Found

```
[ERROR] L0decoder not found. Cannot decode L0 RAW products.
```

**Solution:** Install L0decoder package and ensure it's on PATH:

```bash
# Check if L0decoder is installed
which L0decoder

# Add to PATH if needed
export PATH="/path/to/L0decoder:$PATH"
```

### Partial Success Warning

```
[WARN] step3 partial: 42 patches created (exit=1)
```

**Meaning:** L0Patcher failed but produced some patches. Check:
1. `03_step3_driver.log` - Driver script log
2. `Logfile.log` - L0Patcher tool log
3. `_PARTIAL_SUCCESS.txt` - Error details

### Label Folder Missing

```
[FAIL] label folder not found: /path/to/scene_001/label
```

**Solution:** DVD processing (step 1) failed:
1. Check `01_step1_run.log`
2. Verify SLC/GRD products are valid
3. Ensure sufficient disk space
4. Check `main.py` for errors

### Multiple SAFE Folders Match

```
[WARN] Multiple SAFE folders match 'S1A_IW_SLC_...': [product1.SAFE, product1_reprocessed.SAFE]
[WARN] Using shortest match: product1.SAFE
```

**Meaning:** Ambiguous product names. The script uses the shortest match.

**Solutions:**
1. Remove duplicate/old products from input folder
2. Use more specific product names in `mapping.csv`

### BOM Issues in CSV

If you see encoding errors:

```python
# config.yaml uses UTF-8-sig by default
# mapping.csv is read with encoding='utf-8-sig'
```

**Solution:** Already handled automatically. If issues persist, convert CSV:

```bash
# Remove BOM
sed '1s/^\xEF\xBB\xBF//' mapping.csv > mapping_clean.csv
```

## Vessel Bounding Box Coordinate Transformation

The pipeline automatically transforms vessel bounding boxes from SLC coordinates to RC (Range Compressed) coordinates during **Step 2 (slc2raw)**.

### How It Works

1. **Step 1 (DVD Processing)**: Creates XML files with vessel bounding boxes in **SLC coordinates only**
   - Each XML contains `<BoundingBox>` elements with Top/Left/Bottom/Right in patch-local SLC coordinates

2. **Step 2 (slc2raw)**: Reads XMLs and adds **RC (Range Compressed) coordinate bounding boxes**
   - Transforms each vessel's bounding box:
     - **Azimuth (lines)**: Transformed using `raw_data_line()`
     - **Range (samples)**: Unchanged (range-compressed data)
   - Adds two new XML elements per vessel:
     - `<BoundingBox_RC_Global>`: RC coordinates in full product space
     - `<BoundingBox_RC_Local>`: RC coordinates relative to the RC patch origin
   - Saves updated XMLs back to the `label/` folder

3. **Step 3 (L0Patcher)**: Uses the RAW coordinates to extract matching L0 patches

### XML Structure After Transformation

```xml
<Ship>
  <Name>Ship_1</Name>
  <Centroid_Position>...</Centroid_Position>
  <Size>25.5</Size>
  
  <!-- Original SLC bounding box (local to SLC patch) -->
  <BoundingBox>
    <Top>150</Top>
    <Left>200</Left>
    <Bottom>175</Bottom>
    <Right>225</Right>
  </BoundingBox>
  
  <!-- RC bounding box local to RC patch (added in Step 2) -->
  <BoundingBox_RC_Local>
    <Top>45</Top>
    <Left>62</Left>
    <Bottom>180</Bottom>
    <Right>197</Right>
  </BoundingBox_RC_Local>
  
  <!-- RC bounding box in global product coordinates (added in Step 2) -->
  <BoundingBox_RC_Global>
    <Top>12450</Top>
    <Left>8320</Left>
    <Bottom>12585</Bottom>
    <Right>8455</Right>
  </BoundingBox_RC_Global>
  
  <is_vessel>true</is_vessel>
  <!-- ... other fields ... -->
</Ship>
```

### Important Notes

- **Centroids are NOT transformed**: A single point in SLC corresponds to a spread signal in RC/raw space (the reference function). The bounding box defines this spread region.

- **Coordinate Systems**:
  - **SLC**: Single Look Complex (processed, focused data)
  - **RC**: Range Compressed (azimuth transformed, range unchanged)
  
- **Transformation Details**:
  - **Azimuth (lines)**: Accounts for burst structure, Doppler, FM rate, and reference function length
  - **Range (samples)**: Unchanged for RC data (range-compressed but not fully raw)

- **Automatic Processing**: No manual intervention required - transformations happen automatically during Step 2

- **Full Raw Coordinates**: If you need **full L0 raw coordinates** (with both azimuth and range transformed), see the `SLC_to_FullRaw` tool in `dataset_generation_scripts/SLC_to_FullRaw/`

### Validation

To validate that RC coordinates were added to your XMLs:

```bash
# Check an XML file for RC bounding boxes
grep -A 4 "BoundingBox_RC" output/scene_001/label/DB_OPENSAR_DVD_001.xml

# Or use the validation script
python validate_raw_coordinates.py output/scene_001/label/
```

The validation script checks:
- ✅ All vessel XMLs contain `BoundingBox_RC_Global` and `BoundingBox_RC_Local`
- ✅ All required fields are present (Top, Left, Bottom, Right)
- ✅ Coordinates are valid integers

**Note:** RC (Range Compressed) coordinates represent the transformation from SLC to range-compressed raw (L0) space. This naming convention matches existing models trained on this dataset format.

### RC vs Full Raw Coordinates

The pipeline creates **RC (Range Compressed)** coordinates, which only transform azimuth:

| Transformation Type | Azimuth (Lines) | Range (Samples) | Created By | Use Case |
|---------------------|-----------------|-----------------|------------|----------|
| **RC** (Range Compressed) | ✅ Transformed | ❌ Unchanged | `generate_dataset.py` (Step 2) | Training on range-compressed data |
| **Full Raw** (L0) | ✅ Transformed | ✅ Transformed | `SLC_to_FullRaw` tool | Training on complete L0 raw data |

If you need full raw coordinates with both directions transformed, use the separate `SLC_to_FullRaw` tool located in `dataset_generation_scripts/SLC_to_FullRaw/`.

## Advanced Usage

### Custom Paths and Names

```python
# In generate_dataset.py

# Override output location
OVERRIDE_OUTPUT_ROOT = Path("/mnt/ssd/fast_storage")

# Custom patch data file basename
L0PATCH_DATA_BASENAME = "VesselPatch"  # Results in VesselPatch_001.dat, etc.

# Custom tool paths
SLC2RAW_EXE = "/opt/sar_tools/bin/slc2raw"
L0PATCHER_EXE = "/opt/sar_tools/bin/L0Patcher"
```

### Process Subset of Scenes Programmatically

```python
# Generate even-numbered scenes 2-100
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['input']['scenes_to_process'] = list(range(2, 101, 2))

with open('config.yaml', 'w') as f:
    yaml.dump(config, f)
```
