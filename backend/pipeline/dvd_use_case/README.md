
# Vessel Detection with YOLO (OpenSAR)

YOLO-based vessel detection and pose estimation on OpenSAR datasets.
---

## Project Structure

```
dvd_use_case/
├── large_model/
│   ├── config.yaml                 # Main config (paths, training, inference)
│   ├── pytest.ini                  # Pytest config file
│   ├── data.yaml                   # YOLO dataset config
│   ├── train.py                    # Main training script
│   ├── train_tutorial.ipynb        # Training tutorial
│   ├── data_preprocessing_tutorial.ipynb  # Data preprocessing tutorial
│   ├── patch_vessel_metadata.csv   # Label Summary Used for analysis
│   ├── scripts/
│   │   ├── analysis.py             # Metadata-aware performance analysis
│   │   ├── extract_patch_metadata.py
│   │   ├── filter_dataset.py
│   │   ├── generate_png_from.py
│   │   ├── generate_yolo_labels.py
│   │   ├── inference.py            # Run inference + save prediction labels
│   │   ├── profile_model.py
│   │   └── val.py                  # YOLO validation wrapper
│   ├── tests/                      # Unit tests
│   ├── utilities/                  # Helpers (label conversion, visualization, etc.)
│   └── runs/                       # Training outputs
│
└── small_model/
    ├── utilities/
    │   ├── kd_utilities.py         # Knowledge distillation utilities
    │   └── read_yaml.py             # YAML configuration reader
    ├── __init__.py
    ├── config.yaml
    ├── data.yaml
    └── train_KD.py                 # Knowledge distillation training script
```

---

## Quick Start


1. **Install dependencies and package**
   ```bash
   pip install -e .   # Run this in dvd_use_case
   ```

2. **Interactive Notebooks (Recommended)**
   - **Data Preprocessing Tutorial:**
     - `large_model/train_tutorial.ipynb`
     - Step-by-step guide for preparing the dataset for YOLO-Pose model.
     - Features:
       - convert range compressed to png
       - generate yolodataset
       - filter_dataset (optional)
   - **Training Tutorial:**
     - `large_model/train_tutorial.ipynb`
     - Step-by-step guide for training either the large or small (KD) YOLO-Pose model.
     - Features:
       - Model selection (large/small)
       - Config review and editing
       - Launch training directly from the notebook
       - View logs, validation results, and model profiling (complexity, latency, resource usage)
       - Robust to Docker/Jupyter environments (uses absolute paths)

   - **Inference Tutorial:**
     - `large_model/data_preprocessing_tutorial.ipynb`
     - `large_model/scripts/inference_tutorial.ipynb`
     - `large_model/train_tutorial.ipynb`
     - Interactive demo for running inference on images or batches.
     - Features:
       - Model/config selection
       - Robust to path/config switching
   > **Tip:** These notebooks are designed for use inside a Docker/Jupyter environment and are the easiest way to get started, explore, and validate the pipeline.

2. **Import and use as a package**
  After installation, you can import and use modules from anywhere:
  ```python
  import large_model as lm
  # Access scripts
  # Access utilities
  lm.utilities.rc_labels.LABELS
  ```

  Or import specific modules:
  ```python
  from large_model.utilities import rc_labels
  ```

3. **Configure paths**
  - Edit `config.yaml` for your data locations.
4. **Prepare data**
   - Generate images and labels:
     ```bash
     python -m large_model.scripts.generate_png
     python -m large_model.scripts.generate_yolo_labels
     python -m large_model.scripts.filter_dataset
     ```
   - (Optional) Augment or convert labels as needed.


5. **Train a model**
  ```bash
  python -m large_model.train
  ```


6. **Run inference**
  ```bash
  python -m large_model.scripts.inference
  ```


7. **Run Validation**
  ```bash
```
```
  python -m large_model.scripts.val
  ```


8. **Run Analysis**
  ```bash
  python -m large_model.scripts.analysis
  ```

  With CLI overrides:
  ```bash
  python -m large_model.scripts.analysis \
    --model /path/to/best.pt \
    --images /path/to/images/test \
    --labels /path/to/labels/test \
    --metadata patch_vessel_metadata.csv \
    --output ./test_set \
    --iou 0.5 \
    --conf 0.001
  ```

  All arguments are optional — defaults come from `config.yaml`.


9. **Run all tests**
  ```bash
  pytest tests/
  ```

10. **Run a single test**
  ```bash
  pytest tests/test_case_file.py
  ```

---

## Notes

- **Data**: YOLO format, 1 class (`vessel`), 640×640 images, train/val/test split.
- **Scripts**: All data prep and evaluation scripts are in `scripts/`.
- **Config**: All paths and settings in `config.yaml`.
- **Notebooks**: Use the provided Jupyter notebooks for the most robust and user-friendly workflow. They support dynamic model/config switching, Docker/Jupyter compatibility, and interactive profiling/analysis.

---

## Label Format

Labels use **YOLO-Pose format** with one object per line, all values normalised to `[0, 1]`:

```
class xc yc w h kp_x kp_y kp_vis
```

| Field | Description |
|-------|-------------|
| `class` | Class index (`0` = vessel) |
| `xc`, `yc` | Bounding box centre (normalised) |
| `w`, `h` | Bounding box width and height (normalised) |
| `kp_x`, `kp_y` | Keypoint position — vessel centre (normalised) |
| `kp_vis` | Keypoint visibility (`0` = not labelled, `1` = occluded, `2` = visible) |

**Example** (ground truth):
```
0 0.878906 0.273438 0.050781 0.503906 0.878906 0.273438 2
```

**Example** (prediction output from `inference.py`):
```
0 0.152344 0.354102 0.128906 0.482422 0.155273 0.351563 2
```

Prediction labels are saved to `<output_dir>/labels/` with the same filename stem as the input image.

---

## Analysis Configuration

`analysis.py` runs YOLO `model.val()` on the full test set and on per-bucket subsets, reporting Box and Pose metrics for each. All settings live in the `analysis:` section of `config.yaml`.

### Selecting buckets

Set `buckets` to `'all'` to run every bucket type, or pass a list of specific ones:

```yaml
analysis:
  buckets: 'all'                          # run all bucket types
  # buckets: [polarization, wind]         # run only these two
  # buckets: [density, shore, swath]      # or any combination
```

Available bucket types: `polarization`, `wind`, `vessel_size`, `density`, `shore`, `swath`.

### Modifying thresholds

Each bucket type defines the boundary values that separate its categories. Edit the values to change where the splits occur:

```yaml
analysis:
  buckets: 'all'

  wind:                         # Low: 0 ≤ wind < 4 | Medium: 4 ≤ wind < 8 | High: wind ≥ 8
    low: 4
    medium: 8
    high: ~

  vessel_size:                  # Small: length < 50 | Medium: 50 ≤ length < 150 | Large: length ≥ 150
    small_vessel: 50
    medium_vessel: 150
    large_vessel: ~

  density:                      # Single: n ≤ 1 | Few: 2 ≤ n ≤ 4 | Dense: n ≥ 5
    single_ship: 1
    few_ships: 4
    dense_scene: ~

  shore:                        # Near: 0–5 km | Mid: 5–20 km | Offshore: > 20 km
    near_shore: 5
    mid_shore: 20
    offshore: ~

  swath: {}                     # uses swath ID directly, no thresholds
```

| Bucket | Condition | Range |
|--------|-----------|-------|
| **Wind** | Low | 0 ≤ wind < `low` |
| | Medium | `low` ≤ wind < `medium` |
| | High | wind ≥ `medium` |
| **Vessel size** | Small vessel | length < `small_vessel` |
| | Medium vessel | `small_vessel` ≤ length < `medium_vessel` |
| | Large vessel | length ≥ `medium_vessel` |
| **Density** | Single ship | n ≤ `single_ship` |
| | Few ships | `single_ship` < n ≤ `few_ships` |
| | Dense scene | n > `few_ships` |
| **Shore** | Near shore | dist < `near_shore` km |
| | Mid shore | `near_shore` ≤ dist < `mid_shore` km |
| | Offshore | dist ≥ `mid_shore` km |
| **Swath** | swath_N | grouped by swath ID |
| **Polarization** | VH / VV | extracted from filename |

### Outputs

Analysis results are saved to `<output_dir>/analysis/`:
- Per-metric bar charts (`precision_by_group.png`, `recall_by_group.png`, `mAP50_by_group.png`, `mAP50-95_by_group.png`)
- Console log with per-bucket Box and Pose metrics and average latency

---

## License

This component is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](../main/LICENSE) due to the use of the Ultralytics YOLOv26-Pose model.

The rest of the OpenSAR backend repository is licensed under the MIT License and can be used modularly without AGPL restrictions except where otherwise noted. See the [main LICENSE](../../LICENSE) file for details.
