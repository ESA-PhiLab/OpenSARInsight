# SLC to Full Raw Coordinate Transformation Tool

## Overview

This tool transforms vessel bounding boxes from **SLC coordinates to Full Raw (L0) coordinates**, transforming **both azimuth AND range** directions.

### Difference from RC (Range Compressed) Transformation

| Transformation | Azimuth (Lines) | Range (Samples) | Use Case |
|----------------|-----------------|-----------------|----------|
| **RC** (Range Compressed) | ✅ Transformed | ❌ Unchanged | Range-compressed data (azimuth focused) |
| **Full Raw** | ✅ Transformed | ✅ Transformed | Complete L0 raw data |

The RC transformation (done automatically in `generate_dataset.py` Step 2) only transforms azimuth coordinates. This tool provides **full raw transformation** for both directions.

## Prerequisites

- Python 3.9+
- `slc2raw` package installed
- XML files with SLC vessel bounding boxes
- SLC .SAFE product folders

## Usage

### Basic Usage

```bash
python transform_to_fullraw.py <xml_directory> <slc_product> <output_directory>
```

### Examples

#### Transform all XMLs in a directory

```bash
python transform_to_fullraw.py \
    output/scene_001/label/ \
    inputs/slc/S1A_IW_SLC__1SDV_20210101T000000_20210101T000030_036001_043A01_0001.SAFE \
    output/scene_001/label_fullraw/
```

#### Transform a single XML file

```bash
python transform_to_fullraw.py \
    output/scene_001/label/DB_OPENSAR_DVD_001.xml \
    inputs/slc/S1A_IW_SLC__1SDV_20210101T000000_20210101T000030_036001_043A01_0001.SAFE \
    output/scene_001/label_fullraw/
```

#### With verbose output

```bash
python transform_to_fullraw.py \
    output/scene_001/label/ \
    inputs/slc/S1A_IW_SLC__1SDV_*.SAFE \
    output/scene_001/label_fullraw/ \
    -v
```

## Input Requirements

### XML Structure

Input XMLs must contain:
- `<SARData>` with `<SLCSwath>` element
- `<Corner_Coord>` with `<SARData_Sample>` and `<SARData_Line>` 
- `<List_of_ships>` with `<Ship>` elements containing `<BoundingBox>`

Example input XML:
```xml
<Ship>
  <Name>Ship_1</Name>
  <BoundingBox>
    <Top>150</Top>
    <Left>200</Left>
    <Bottom>175</Bottom>
    <Right>225</Right>
  </BoundingBox>
  <!-- Other fields... -->
</Ship>
```

### SLC Product

- Must be a `.SAFE` folder from Sentinel-1
- Must contain `annotation/` subfolder with annotation XML files
- Must match the product referenced in the XML files

## Output

The tool adds two new XML elements to each vessel:

### BoundingBox_FullRaw_Local
Coordinates relative to the raw patch origin:
```xml
<BoundingBox_FullRaw_Local>
  <Top>52</Top>
  <Left>78</Left>
  <Bottom>210</Bottom>
  <Right>235</Right>
</BoundingBox_FullRaw_Local>
```

### BoundingBox_FullRaw_Global
Coordinates in global product space:
```xml
<BoundingBox_FullRaw_Global>
  <Top>12450</Top>
  <Left>8560</Left>
  <Bottom>12610</Bottom>
  <Right>8725</Right>
</BoundingBox_FullRaw_Global>
```

## Command Line Options

```
positional arguments:
  xml_path              XML file or directory containing XML files
  slc_product          Path to SLC .SAFE product folder
  output_dir           Output directory for transformed XMLs

optional arguments:
  -h, --help           show this help message and exit
  -v, --verbose        Verbose output
```

## Transformation Details

### Range (Samples)
- Transforms using `raw_data_pixel()` function
- Accounts for different sampling rates between SLC and raw
- Adjusts for first valid sample offset
- Adds range reference function length

### Azimuth (Lines)
- Transforms using `raw_data_line()` function
- Accounts for burst structure
- Adjusts for Doppler, FM rate
- Adds azimuth reference function margins (±az_len/2)

### Coordinate Systems
- **Input**: SLC bounding boxes (local to 512x512 patch)
- **Output (Local)**: Raw coordinates relative to raw patch origin
- **Output (Global)**: Raw coordinates in full product space

## Important Notes

- **Centroids are NOT transformed**: A single point in SLC corresponds to a spread signal in raw space
- **Bounding boxes only**: The transformation only applies to vessel bounding boxes, not patch corners
- **Both directions**: Unlike RC, this transforms BOTH azimuth and range
- **Non-destructive**: Original SLC bounding boxes are preserved in the XML

## Integration with Pipeline

This tool is **separate** from the main `generate_dataset.py` pipeline:

```
generate_dataset.py (automatic pipeline)
  ├─ Step 1: DVD Processing → Creates XMLs with SLC coords
  ├─ Step 2: slc2raw → Adds RC coords (azimuth only)
  └─ Step 3: L0Patcher → Extracts L0 patches

transform_to_fullraw.py (manual post-processing)
  └─ Adds Full Raw coords (both azimuth and range) to existing XMLs
```

Use `transform_to_fullraw.py` when you need **full L0 raw coordinates** for training models on complete raw data.

## Troubleshooting

### "Annotation file not found"
- Verify the SLC product path points to a `.SAFE` folder
- Check that `annotation/` subfolder exists
- Ensure swath number matches (IW1, IW2, or IW3)

### "No XML files found"
- Check that the XML path contains `.xml` files
- Verify file permissions

### "No vessels found in XML"
- Check that XML contains `<List_of_ships>` with `<Ship>` elements
- Verify each ship has a `<BoundingBox>` element

### Import errors
- Ensure `slc2raw` package is installed
- Verify the path to slc2raw-main is correct
- Check Python version (requires 3.9+)

## Example Workflow

```bash
# 1. Run main pipeline (generates RC coordinates automatically)
python generate_dataset.py

# 2. Transform to Full Raw (optional post-processing)
python transform_to_fullraw.py \
    output/scene_001/label/ \
    inputs/slc/S1A_IW_SLC__1SDV_*.SAFE \
    output/scene_001/label_fullraw/

# 3. Validate results
grep -A 4 "BoundingBox_FullRaw" output/scene_001/label_fullraw/DB_OPENSAR_DVD_001.xml
```

## See Also

- **DVD_dataset_generation/README.md**: Main pipeline documentation
- **slc2raw documentation**: Coordinate transformation functions
- **L0_preparation**: L0 data preparation tools
