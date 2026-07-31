# Dataset Generation Pipeline

## Overview

This pipeline automates the generation of SAR datasets by processing Sentinel-1 products through a three-step workflow:

1. **RFI Processing**: Generates patches and labels from SLC/GRD products
2. **SLC to RAW Conversion**: Converts SLC patches to RAW format using the `slc2raw` tool
3. **L0 Patch Extraction**: Extracts L0 RAW patches from decoded RAW products using the `L0Patcher` tool

The pipeline processes multiple scenes sequentially based on a scene mapping CSV file and generates structured output for each scene.

---

## Case Study Description

### SAR Dataset Generation Pipeline

This pipeline focuses on **automated dataset generation** from SAR imagery.

Objective:

* Process multiple SAR scenes automatically
* Generate SLC patches and labels via RFI processing
* Convert SLC patches to RAW format with full traceability
* Extract corresponding L0 RAW patches from source data

This case study uses a YAML configuration file to control processing of multiple scenes.

---

## Requirements

### System Requirements

* Python 3.x installed
* Command-line tools on PATH:
  - `slc2raw`: Converts SLC patches to RAW format
  - `L0Patcher`: Extracts L0 RAW patches from decoded products

### Python Dependencies

* PyYAML: for configuration file parsing

  ```bash
  pip install pyyaml
  ```

### Required Directory Structure

Your products directory should contain:

```
products_root/
├── slc/
│   └── [SLC_PRODUCT_NAME].SAFE/
└── raw/
    └── [RAW_PRODUCT_NAME].SAFE/
```

### Validation

Before running the pipeline, verify that the required tools are available:

```bash
# Check if tools are on PATH
which slc2raw
which L0Patcher
```

If tools are not found, add them to your PATH or install them according to their respective installation guides.

---

## Configuration File

The pipeline is controlled through a YAML configuration file (default: `config.yaml`).

### Configuration Structure

```yaml
input:
  products_root: /path/to/products/directory
  scene_mapping: /path/to/scene_mapping.csv
  scenes_to_process: [1, 2, 3]  # or ranges like [1-5, 7, 10-12]

output:
  output_root: /path/to/output/directory
```

---

## Input Parameters

### `products_root`

* **Type**: string (path)
* **Description**: Root directory containing SLC and RAW products organized in subdirectories:
  - `slc/`: Contains SLC products as `.SAFE` directories
  - `raw/`: Contains decoded L0 RAW products as `.SAFE` directories
  - `grd/`: Contains GRD products as `.SAFE` directories
  - `binarymask/`: Contains binarymask for each data


### `scene_mapping`

* **Type**: string (path)
* **Description**: Path to CSV file (semicolon-separated) mapping scene numbers to product names. Must contain columns:
  - `num`: Scene identifier (integer)
  - `slc`: SLC product name (e.g., `S1A_IW_SLC__1SDV_...`)
  - `raw`: RAW product name (e.g., `S1A_IW_RAW__0SDV_...`)
* **Example**: `/data/scene_mapping.csv`

### `scenes_to_process`

* **Type**: list of integers or range strings (optional)
* **Description**: Scenes to process from the mapping file. Supports:
  - Individual scenes: `[1, 3, 5]`
  - Ranges (inclusive): `[1-5]` processes scenes 1, 2, 3, 4, 5
  - Mixed: `[1-3, 7, 10-12]` processes scenes 1, 2, 3, 7, 10, 11, 12
  - If omitted or null, **all scenes** from the mapping file will be processed
* **Example**: `[1, 5-10, 15]`

---

## Output Parameters

### `output_root`

* **Type**: string (path)
* **Description**: Root directory where outputs will be organized by scene. Each scene creates a subdirectory `scene_XXX/` containing the results from all processing steps.
* **Example**: `/data/outputs/dataset_v1`

---

## Execution

### Basic Usage

From the directory containing `generate_dataset.py`, run:

```bash
python generate_dataset.py
```

By default, this uses `config.yaml` in the same directory.

### Custom Configuration

To use a different configuration file:

```bash
python generate_dataset.py --config /path/to/custom_config.yaml
```

### Pipeline Steps

The pipeline executes three steps for each scene:

1. **Step 1: DVD Processing**
   - Runs DVD vessel detection workflow (run.py) with per-scene config
   - Generates patches, labels, and masks in `scene_XXX/`
   - Creates: `slc/`, `mask/`, `grd/`, `label/` folders
   - Completion marker: `01_step1_DONE.txt`

2. **Step 2: SLC to RAW Conversion**
   - Converts SLC patches to RAW format using `slc2raw` tool
   - Input: DVD label folder and SLC SAFE product
   - Output: `scene_XXX/02_slc2raw/rawH.npy` (patch info file)
   - Generates traceability log
   - Completion marker: `_DONE.txt`

3. **Step 3: L0 Patch Extraction**
   - Extracts L0 RAW patches using `L0Patcher` tool
   - Input: Decoded L0 SAFE product and patch info from Step 2
   - Output: `.dat` patch files in `scene_XXX/03_raw_patches/`
   - Handles partial success if some patches are created
   - Completion marker: `_DONE.txt`

### Resume Capability

The pipeline automatically skips completed steps:
- Checks for completion markers before each step
- Can safely re-run after interruption
- Only processes incomplete scenes

---

## Outputs

### Directory Structure

Each processed scene creates the following structure:

```
output_root/
└── scene_XXX/
    ├── 01_step1_DONE.txt          # Step 1 completion marker
    ├── 01_step1_run.log           # Step 1 execution log
    ├── slc/                        # Complex patches (from DVD)
    ├── mask/                        # Mask patches (from DVD)
    ├── grd/                       # GRD patches (from DVD)
    ├── label/                     # XML labels (from DVD)
    ├── 02_slc2raw/
    │   ├── rawH.npy              # Patch info file
    │   ├── traceability_log.txt  # SLC2RAW tool log
    │   ├── 02_step2_driver.log   # Step 2 execution log
    │   └── _DONE.txt             # Step 2 completion marker
    └── 03_raw_patches/
        ├── *.dat                  # L0 RAW patch files
        ├── *.xml                  # Patch metadata
        ├── Logfile.log           # L0Patcher tool log
        ├── 03_step3_driver.log   # Step 3 execution log
        └── _DONE.txt             # Step 3 completion marker
```

### Output Files by Type

#### RFI Outputs (Step 1)
- **slc/**: Complex patches in complex64 TIFF format
- **mask/**: Mask patches in PNG format
- **grd/**: GRD patches (VV/VH) in GeoTIFF format
- **label/**: XML metadata files for each patch

#### SLC to RAW Outputs (Step 2)
- **rawH.npy**: NumPy array containing patch information (coordinates, dimensions, etc.)
- **traceability_log.txt**: Detailed conversion log from `slc2raw` tool

#### L0 RAW Patches (Step 3)
- **.dat files**: Binary L0 RAW patch data
- **.xml files**: Patch metadata and parameters
- Patch naming follows base name from SLC product (without `.SAFE` suffix)

### Logs

Each step generates two types of logs:
- **Driver logs** (`XX_stepX_driver.log`): High-level pipeline execution log
- **Tool logs**: Detailed output from external tools (`traceability_log.txt`, `Logfile.log`)

### Processing Summary

At completion, the pipeline prints:
- Total scenes requested
- Successfully completed scenes
- Failed scenes with reasons (if any)

---

## Error Handling

### Validation Checks

The pipeline performs validation at multiple stages:

1. **Initial Validation**
   - Verifies required tools (`slc2raw`, `L0Patcher`) are on PATH
   - Checks config file exists
   - Validates products_root and scene_mapping file exist

2. **Per-Scene Validation**
   - Checks SLC SAFE directory exists
   - Checks decoded L0 RAW SAFE directory exists
   - Verifies DVD outputs (label folder) were created
   - Confirms patch info file was generated

### Partial Success Handling

**Step 3 (L0Patcher)** has special handling:
- If L0Patcher exits with an error but created `.dat` patches, marks as "partial success"
- Creates `_PARTIAL_SUCCESS.txt` with details
- Allows pipeline to continue rather than fail completely

### Failed Scene Tracking

Failed scenes are recorded with specific reasons:
- "Not in mapping file"
- "Missing SLC name" / "Missing RAW name"
- "SLC SAFE missing" / "L0 decoded missing"
- "label folder missing"
- "step1/step2/step3 failed"

Check logs in the respective scene directory for detailed error information. 

---

## Example Configuration

### Minimal Config

```yaml
input:
  products_root: /data/sentinel1/products
  scene_mapping: /data/scene_mapping.csv
  # Omit scenes_to_process to process all scenes

output:
  output_root: /data/outputs/dataset_run1
```

### Selective Processing

```yaml
input:
  products_root: /mnt/storage/sentinel1
  scene_mapping: /home/user/scenes.csv
  scenes_to_process: [1-5, 10, 15-20]  # Process scenes 1,2,3,4,5,10,15,16,17,18,19,20

output:
  output_root: /mnt/storage/outputs/test_run
```

### Scene Mapping CSV Format

```csv
num;slc;raw
1;S1A_IW_SLC__1SDV_20210101T060000_20210101T060030_036000_043000_0001;S1A_IW_RAW__0SDV_20210101T060000_20210101T060030_036000_043000_0001
2;S1A_IW_SLC__1SDV_20210102T060000_20210102T060030_036001_043001_0002;S1A_IW_RAW__0SDV_20210102T060000_20210102T060030_036001_043001_0002
```

**Important**: 
- Delimiter is semicolon (`;`)
- Product names should **not** include `.SAFE` suffix (added automatically)
- All three columns (`num`, `slc`, `raw`) are required

---

## Troubleshooting

### Tool Not Found

**Error**: `slc2raw` or `L0Patcher` not found

**Solution**: 
1. Verify tools are installed
2. Add tool directories to your PATH:
   ```bash
   export PATH=/path/to/tools/bin:$PATH
   ```
3. Or use absolute paths (requires code modification)

### Missing Products

**Error**: SLC SAFE or L0 decoded not found

**Solution**:
1. Check `products_root` path is correct
2. Verify directory structure:
   ```
   products_root/
   ├── slc/[PRODUCT_NAME].SAFE/
   └── raw/[PRODUCT_NAME].SAFE/
   ```
3. Ensure product names in CSV match actual directories
4. Remember `.SAFE` suffix is added automatically

### DVD Step Fails

**Error**: Step 1 failed

**Solution**:
1. Check `01_step1_run.log` in scene directory
2. Verify DVD dependencies are installed
3. Ensure scene exists in mapping CSV
4. Check RFI `run.py` can process individual scenes

### Label Folder Missing

**Error**: Label folder not found after RFI processing

**Solution**:
1. DVD processing may have failed silently
2. Check if scene has any detections (empty scenes may not create labels)
3. Review DVD configuration and input requirements

### L0Patcher Partial Success

**Warning**: L0Patcher exited with error but produced patches

**Explanation**: This is handled gracefully by the pipeline. The scene is marked "partial success" and processing continues. Review `_PARTIAL_SUCCESS.txt` and logs in `03_raw_patches/` for details.

### Configuration Errors

**Common issues**:
- **Missing required fields**: Ensure all required config keys exist
- **Invalid YAML syntax**: Use proper YAML formatting (spaces, not tabs)
- **Path doesn't exist**: Use absolute paths or verify relative paths are correct
- **CSV delimiter**: Ensure scene mapping uses semicolon (`;`)

---

## Advanced Usage

### Processing Single Scene

Edit config to process just one scene:

```yaml
input:
  scenes_to_process: [5]
```

### Resuming After Failure

Simply re-run the same command:
```bash
python generate_dataset.py --config config.yaml
```

The pipeline will skip completed scenes/steps and continue where it left off.

### Monitoring Progress

Watch the console output for:
- Current scene being processed (e.g., "SCENE 003 (2/10)")
- Step-by-step progress within each scene
- Validation checks (✓ for success, ✗ for failure)
- File counts (patches, labels, etc.)


## Notes

* All paths can be absolute or relative to the script directory
* Ensure all input files and products exist before execution
* Each scene is processed independently - failure of one scene doesn't stop others
* Completion markers allow safe resume after interruption
* Logs are preserved for all steps for debugging and traceability

---

## Maintenance

This README reflects the functionality of `generate_dataset.py`. If the pipeline workflow or configuration structure changes, update this document accordingly to maintain accuracy and reproducibility.
