Flood Detection Dataset Generation Pipeline (OpenSAR-FD)
=========================================================

This project implements a full flood detection dataset generation pipeline for
Sentinel-1 SLC and GRD products. Starting from raw Sentinel-1 acquisitions and
AOI shapefiles, it generates FD labels, SLC/GRD patches, and optionally L0 RAW
patches.

----------------------------------------------------------------------
1. Pipeline overview
----------------------------------------------------------------------

The pipeline is driven by generate_dataset.py and runs up to three sequential
steps per scene:

  Step 1 — FD patch generation (always runs)
    Invokes run.py internally to produce SLC complex patches, GRD amplitude
    patches, and XML label files.

  Step 2 — SLC to RAW conversion (requires RAW column in mapping CSV)
    Calls the external tool slc2raw to convert SLC patches to RAW format
    (rawH.npy) using the label folder produced in Step 1.

  Step 3 — L0 RAW patch extraction (requires RAW column in mapping CSV)
    Calls the external tool L0Patcher to extract .dat patch files from the
    decoded L0 product using the patch info from Step 2.

Steps 2 and 3 are skipped automatically if no RAW product is listed for a
scene in scene_mapping.csv. Each step writes a done-marker file so re-runs
skip already-completed work.

----------------------------------------------------------------------
2. Required external tools
----------------------------------------------------------------------

Before running the pipeline, ensure the following tools are available on PATH:

  slc2raw     – converts SLC patches to RAW format
  L0Patcher   – extracts L0 RAW patches from a decoded product

The pipeline checks for these at startup and exits immediately if either is
missing.

----------------------------------------------------------------------
3. Project data layout
----------------------------------------------------------------------

The pipeline relies on TWO different data locations:

1) input_root  → dynamic input data (Sentinel-1 scenes, AOIs)
2) shared-data → static auxiliary data used internally by the code

----------------------------------------------------------------------
3.1 input_root (processing input data)
----------------------------------------------------------------------

Defined in config.yaml under paths.input_root. Expected structure:

<input_root>/
└─ <scene_identifier>/
   ├─ <scene_identifier>_mask/
   │    └─ <Aoi_id>/
   │         ├─ aoi        # AOI shapefiles
   │         ├─ event
   │         └─ hydro
   ├─ SLC/
   │   └─ <SLC_product>.SAFE
   └─ GRD/
       └─ <GRD_product>.SAFE

----------------------------------------------------------------------
3.2 shared-data (static auxiliary data)
----------------------------------------------------------------------

The shared-data folder must be present inside the project directory:

project_root/
├─ generate_dataset.py
├─ run.py
├─ config.yaml
└─ shared-data/
   └─ natural_earth/
      └─ land_polygons.shp

This folder contains Natural Earth land polygons used for land/sea/coastal
classification. It is NOT specified in config.yaml and must NOT be mixed with
Sentinel-1 input data.

----------------------------------------------------------------------
4. Configuration file
----------------------------------------------------------------------

All settings are read from a YAML file (default: config.yaml).

Full example matching the current config.yaml:

  processing:
    scenes_to_process: [1, 3]   # list, "1-5" range string, or null (all)

  paths:
    input_root:        "path/to/input"
    output_root:       "path/to/output"
    floods_mapping_csv: "path/to/scene_mapping.csv"

  options:
    USE_AOI_EXACT_POLYGON: false
    SIZEPATCH: 512

scenes_to_process accepts:
  - A list of integers:   [1, 3, 5]
  - A range string:       "1-5"
  - A mixed list:         [1, "3-5"]
  - null / omitted:       process all scenes found in the mapping CSV

----------------------------------------------------------------------
5. Scene mapping CSV
----------------------------------------------------------------------

scene_mapping.csv uses semicolon (;) as delimiter and supports two schemas:

FD schema (used by this project):
  ID ; SCENE ; FLOOD DATE ; AOI ; GRD ; SLC [; RAW]

  - ID, SCENE, AOI may be blank; blank cells inherit the value from the
    previous row (hierarchical / multi-AOI layout).
  - RAW column is optional. If absent or empty, Steps 2 and 3 are skipped
    for that scene and only FD labels/patches are generated.

Full-pipeline schema (alternative):
  num ; slc ; raw ; ...

----------------------------------------------------------------------
6. Running the pipeline
----------------------------------------------------------------------

    python generate_dataset.py --config config.yaml

The --config argument is optional; if omitted the script looks for
config.yaml in the same directory as the script.

----------------------------------------------------------------------
7. Output structure
----------------------------------------------------------------------

<output_root>/
├── <scene_identifier>/
│   └── id_XXX/              # FD step output
│       ├── cp/              # SLC complex patches
│       ├── grd/             # GRD amplitude patches
│       └── label/           # XML metadata and legend
└── scene_XXX/               # pipeline wrapper per logical ID
    ├── 01_step1_DONE.txt
    ├── 02_slc2raw/
    │   ├── rawH.npy          # patch info file
    │   └── traceability_log.txt
    └── 03_raw_patches/
        ├── *.dat             # extracted RAW patch files
        └── *.xml

----------------------------------------------------------------------
8. Summary
----------------------------------------------------------------------

- generate_dataset.py is the single entry point for the full pipeline.
- run.py is invoked internally (Step 1) and should not be called directly.
- RAW steps (2 and 3) are optional and controlled by the mapping CSV.
- Completed steps are skipped on re-runs via done-marker files.
- slc2raw and L0Patcher must be installed and available on PATH.