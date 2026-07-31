# OpenSAR Main Pipeline

`main_pipeline.py` is the unified entry point for the OpenSAR / AI4SAR pipeline. It provides a single command-line interface to run training, evaluation, inference, and end-to-end complexity profiling for all supported models, as well as executing auxiliary tools such as label conversion, geocoding, and dataset generation.

---

## Table of Contents

- [Dependencies and Requirements](#dependencies-and-requirements)
  - [Docker Image](#docker-image)
  - [Dataset and Artifact Files](#dataset-and-artifact-files)
  - [Configuration Files](#configuration-files)
- [Supported Models](#supported-models)
- [Supported Tools](#supported-tools)
- [Usage](#usage)

---

## Dependencies and Requirements

### Docker Image

All pipeline commands must be run inside the OpenSAR Docker container. The container is based on `nvcr.io/nvidia/pytorch:25.08-py3` and bundles CUDA, GDAL, OpenCV, and all Python dependencies. Refer to ``docker/README.md`` for instructions
on how to install the docker container.

### Dataset and Artifact Files

Ensure that you have the datasets and artifact files downloaded before
running any pipeline commands.


### Configuration Files

Update the paths in the ``.yaml`` configuration files to match your local or shared storage layout before running any pipeline commands.

---

## Supported Models

| Model Key | Description | Train | Evaluate | Inference | End-to-End |
|---|---|:---:|:---:|:---:|:---:|
| `rfi_large` | UNet-based RFI segmentation (large) | ✓ | ✓ | ✓ | ✓ |
| `rfi_small` | UNet-based RFI segmentation (small/KD) | ✓ | ✓ | ✓ | ✓ |
| `vd_large` | YOLO pose-estimation vessel detection | ✓ | — | ✓ | ✓ |
| `vd_small` | Knowledge-distilled vessel detection | ✓ | — | ✓ | ✓ |

---

## Supported Tools

| Tool Key | Description |
|---|---|
| `label_conversion` | Convert timestamped lat/lon labels to SLC coordinates (uses SARFI) |
| `geocoding` | Geocode SAR patch using L1 product metadata and orbit files |
| `dataset_generation_dvd` | Generate the DVD (Vessel Detection) dataset |
| `dataset_generation_rfi` | Generate the RFI dataset |
| `dataset_generation_flood` | Generate the Flood Detection dataset |
| `range_compression` | Run the range compression demo (L0 → RC patches) |

---

## Usage

### General Syntax

```bash
python main_pipeline.py [--model <model>] [--mode <mode>] [--tool <tool>] [--extra-args "<args>"] [--list]
```

Run from within the `pipeline/main/` directory.

---

## License

This component is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE) due to the use of the Ultralytics YOLOv26-Pose model.

The rest of the OpenSAR backend repository is licensed under the MIT License and can be used modularly without AGPL restrictions except where otherwise noted. See the [main LICENSE](../../LICENSE) file for details.
