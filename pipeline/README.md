# OpenSAR Pipeline

This directory contains the modular components of the OpenSAR / AI4SAR processing and inference pipeline. Each block is self-contained and has its own README with detailed usage instructions.

---

## Pipeline Components

### [`main/`](main/README.md) — Main Pipeline Entry Point

`main_pipeline.py` is the unified command-line entry point for the entire pipeline. It provides a single interface to run training, evaluation, inference, and end-to-end profiling for all supported models, as well as auxiliary tools such as label conversion, geocoding, and dataset generation.

**Supported models:** `rfi_large`, `rfi_small`, `vd_large`, `vd_small`

**Supported tools:** `label_conversion`, `geocoding`, `dataset_generation_dvd`, `dataset_generation_rfi`, `dataset_generation_flood`, `range_compression`

See [`main/README.md`](main/README.md) for full usage instructions and CLI reference.

---

### [`data_preprocessing/`](data_preprocessing/README.md) — Data Augmentation and Normalization

PyTorch-compatible augmentation and normalization transforms for SAR data. Supports both SLC amplitude and L0 I/Q (range-compressed) data formats.

Key transforms include:
- **SLC amplitude**: random translation, cropping, resizing, speckle noise simulation
- **L0 I/Q / range-compressed**: amplitude scaling, phase perturbation, narrowband spectral dropout, subsampling and interpolation, azimuth defocus, bandwidth trim
- **Normalization**: log-compression + percentile clipping for SLC; per-channel z-score for L0 I/Q

See [`data_preprocessing/README.md`](data_preprocessing/README.md) for recommended augmentation parameters per use case (Flood, RFI, Vessel Detection) and code examples.

---

### [`dvd_use_case/`](dvd_use_case/README.md) — Vessel Detection (YOLO)

YOLO-based vessel detection and pose estimation on SAR imagery. Contains large and small (knowledge-distilled) model variants.

Key components:
- **`large_model/`**: training, inference, validation, and metadata-aware performance analysis scripts
- **`small_model/`**: knowledge-distillation variant
- Labels use YOLO-Pose format (bounding box + keypoint per vessel)

See [`dvd_use_case/README.md`](dvd_use_case/README.md) for setup, data preparation, training, inference, and analysis instructions.

---

### [`RFI_usecase/`](RFI_usecase/README.md) — RFI Segmentation (UNet)

Binary pixel-wise segmentation of Radio Frequency Interference (RFI) artefacts on range-compressed SAR data. Uses a UNet-based architecture with 4-channel VV/VH I/Q input.

Key components:
- **Input**: 4-channel `[VV_I, VV_Q, VH_I, VH_Q]` tensors from paired VV/VH `.npy` files
- **Output**: binary segmentation mask (RFI vs. background)
- **Loss**: focal loss with ignore-index support for invalid pixels
- **Normalization**: per-dataset z-score (precomputed stats in JSON)

See [`RFI_usecase/README.md`](RFI_usecase/README.md) for dataset format, model architecture, training, and evaluation instructions.

---

### [`geocoding_block/`](geocoding_block/README.md) — SAR Patch Geocoding

Geocodes SLC image patches from the OpenSAR dataset using L1 product metadata and orbit files. Supports vessel, RFI, and flood image patches.

Key inputs:
- L1 SLC product directory
- Orbit files (POEORB)
- Label files and a dataset look-up table (LUT)
- Copernicus GLO-30 DEM

Output is a geocoded GeoTIFF with queryable latitude/longitude per pixel.

See [`geocoding_block/README.md`](geocoding_block/README.md) for configuration, running instructions, and Docker requirements.

---

## Running the Pipeline

All pipeline commands should be run inside the OpenSAR Docker container. See [`docker/README.md`](../docker/README.md) for container setup instructions.

```bash
# General syntax (from pipeline/main/)
python main_pipeline.py [--model <model>] [--mode <mode>] [--tool <tool>] [--extra-args "<args>"] [--list]
```

Refer to [`main/README.md`](main/README.md) for the full CLI reference.

---

## License

**Note:** The `main/` entry point and `dvd_use_case/` (vessel detection) components are licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](main/LICENSE) due to the use of the Ultralytics YOLOv26-Pose model.

All other pipeline components (`data_preprocessing/`, `RFI_usecase/`, `geocoding_block/`) are licensed under the [MIT License](../LICENSE) and can be used modularly without AGPL restrictions.
