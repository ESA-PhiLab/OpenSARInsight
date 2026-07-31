# RFI Segmentation Pipeline

This folder contains the training, inference, profiling, and post-processing pipeline for binary segmentation of radio-frequency interference (RFI) in range-compressed SAR data. The pipeline supports:

- A large "U-Net" model used as the main or teacher model
- A lightweight "UNetSmall" model for faster deployment
- Knowledge distillation (KD) training for the small model
- Unified evaluation and CPU/GPU profiling for both model sizes
- Optional post-processing of predicted masks using local SAR power contrast
- Ablation experiments for smaller student variants and scratch training

The input is a 4-channel SAR tensor built from VV/VH complex range-compressed data:

```text
[VV_I, VV_Q, VH_I, VH_Q]
```

## Project structure

```text
RFI_usecase/
|-- models/                           # Model architecture definitions
|   |-- unet.py                       # Large U-Net model
|   |-- unet_small.py                 # Lightweight U-Net student model
|-- utils/                            # Shared dataloading, transforms, plotting, and training utilities
|   |-- rc_dataloader.py              # VV/VH pairing, mask loading, padding, and valid-mask generation
|   |-- augmentation.py               # RC-domain data augmentations
|   |-- normalization.py              # Per-dataset z-score normalization
|   |-- utilities.py                  # General helper functions
|   |-- plotting.py                   # Training-curve plotting utilities
|   |-- utils_distill.py              # Knowledge distillation helper functions
|   |-- compute-rfi-zscore-stat.py    # Script for computing 4-channel normalization statistics
|   |-- logging_setup.py              # Central logging configuration
|-- post_processing/                  # Profiling and RFI mask post-processing utilities
|   |-- profiler.py                   # CPU/GPU inference profiling
|   |-- rfi_postprocessing.py         # Local SAR power-contrast filtering for predicted masks
|-- Ablation/                         # Auxiliary experiments and reduced student-model variants
|   |-- train_rfi_small_scratch.py    # Train small model without KD
|   |-- train_rfi_smallKD_ablation.py # Train smaller KD student variants
|   |-- inference_ablation.py         # Inference script for ablation experiments
|   |-- *.yaml                        # Ablation-specific configuration files
|-- train_rfi_large.py                # Train the large U-Net teacher/reference model
|-- train_rfi_smallKD.py              # Train the lightweight student model with optional KD
|-- inference.py                      # Unified evaluation, profiling, prediction export, and post-processing
|-- inference_tutorial.ipynb          # Notebook for interactive inference inspection
|-- config-rfi-large.yaml             # Large-model training configuration
|-- config-rfi-kd.yaml                # Small/KD model training configuration
|-- transforms.yaml                   # Normalization and augmentation configuration
|-- rfi_zscore_stats.json             # Precomputed 4-channel z-score statistics
|-- README.md                         # Project documentation
```


## Main scripts

### `train_rfi_large.py`

Trains the large U-Net model from scratch on 4-channel range-compressed SAR inputs.

- model: `models/unet.py`
- typical role: teacher model or high-accuracy reference model
- output: `best.pth`, `last.pth`, `epoch_XXX.pth`, `metrics.csv`, training-curve PNGs

### `train_rfi_smallKD.py`

Trains the lightweight student model, optionally with knowledge distillation from the large model.

- model: `models/unet_small.py`
- KD modes: `logits`, `features`, `hybrid`, `none`
- optional Weights & Biases logging
- output: `best_student.pth`, `last_student.pth`, `student_epoch_XXX.pth`, `metrics_student.csv`, training-curve PNGs

### `inference.py`

Unified evaluation and profiling script for both model sizes.

It always:

- evaluates the selected model on the test set
- Profiles CPU inference
- Profiles GPU inference when CUDA is available

It can also:

- generate prediction mask PNGs
- run optional dB-based RFI post-processing

### `post_processing/rfi_postprocessing.py`

Filters connected components in predicted RFI masks using local power contrast computed from the original VV/VH SAR data.

### `ablation/`

Contains auxiliary experiments, including:

- small model trained from scratch
- smaller KD students with reduced `base_channels`
- separate ablation inference script

## Data assumptions

The data loader in `utils/rc_dataloader.py` expects VV and VH `.npy` files that differ only by the `-vv-` / `-vh-` token in the filename.

Example pair:

```text
s1a-iw-raw-s-vv-...-RFI_892_scaled.npy
s1a-iw-raw-s-vh-...-RFI_892_scaled.npy
```

These are combined into one 4-channel sample:

```text
[VV_I, VV_Q, VH_I, VH_Q]
```

Supported `.npy` formats for each polarization are:

- complex array: `(H, W)`
- real channel-last: `(H, W, 2)`
- real channel-first: `(2, H, W)`

### Expected dataset layout

The current mask resolver looks for data under split folders named `train`, `val`, or `test`.

Typical layout:

```text
<dataset_root>/
|-- train/
|   |-- *.npy
|    --  DB_OPENSAR_RFI_<id>_MASK.png
|     
|-- val/
|   |-- *.npy
|    --  DB_OPENSAR_RFI_<id>_MASK.png
|       
 -- test/
    |-- *.npy
     --  DB_OPENSAR_RFI_<id>_MASK.png
```

### Mask handling

The dataloader supports:

- Binary masks stored as {0, 255}, which are converted to {0, 1}
- Integer masks
- 255 as the ignore value during padding and metric computation

If the RC tensor and mask have different spatial sizes, the RC tensor is resized to match the mask size before transforms are applied.

During batching:

- RC tensors are padded to the batch maximum size
- Padding can optionally be aligned to a stride multiple
- A valid_mask is generated to separate real pixels from padded pixels
- Mask padding uses ignore_index = 255

## Models

### Large U-Net

Implemented in `models/unet.py`.

Default large-model config in `config-rfi-large.yaml`:

- `in_channels: 4`
- `out_channels: 1`
- `base_channels: 32`
- `depth: 4`
- `stride_multiple: 16`

### Small model

Implemented in `models/unet_small.py`.

This is a lightweight U-Net variant using depthwise-separable convolutions.

Default small-model config in `config-rfi-kd.yaml`:

- `in_channels: 4`
- `out_channels: 1`
- `base_channels: 16`
- `depth: 3`
- `stride_multiple: 8`

## Normalization and augmentations

The transform pipeline is defined in `transforms.yaml`.

### Normalization

The current setup uses per-dataset z-score normalization with statistics stored in `rfi_zscore_stats.json`.

Statistics are stored in:

```
rfi_zscore_stats.json
 
```

The active normalization profile is:

```yaml
norm:
  active: rfi_zscore
```

### Available augmentations

The codebase includes the following configurable RC-domain augmentations in `utils/augmentation.py`:

- Amplitude/phase perturbation
- Narrowband spectral dropout
- Bandwidth trimming
- Azimuth defocus
- Subsample-and-interpolate

In the current `transforms.yaml`, all augmentation blocks are currently disabled and only normalization is active.

### Recomputing z-score statistics

If the dataset changes, recompute the normalization statistics:

```bash
python utils/compute-rfi-zscore-stat.py \
  --data_dir /path/to/train \
  --output_json /path/to/rfi_zscore_stats.json \
  --batch_size 4 \
  --num_workers 4
```

## Config files path
```
- RFI large model: .../RFI_usecase/config-rfi-large.yaml
- RFI small-KD model: .../RFI_usecase/config-rfi-kd.yaml
- RFI small-scratch ablation: .../RFI_usecase/ablation/config-rfi-small-scratch.yaml
- RFI small-KD ablation: .../RFI_usecase/ablation/config-rfi-kd-ablation.yaml
```



## Environment and setup notes

This project is normally run inside a Docker container rather than directly on the host machine.

The container should provide:

- Python
- PyTorch
- CUDA support when using GPU
- Required Python dependencies
- Access to the mounted SAR dataset path


## Training

### 1. Train the large model

```bash
python train_rfi_large.py --config config-rfi-large.yaml
```

Notes:

- This trains the full U-Net teacher/reference model
- Training and validation transforms are built from `transforms.yaml`
- Checkpoints and plots are written under `save.dir`

Typical outputs:

- `best.pth`
- `last.pth`
- `epoch_XXX.pth`
- `metrics.csv`
- `loss_curve.png`
- `acc_curve.png`
- `dice_curve.png`
- `iou_curve.png`
- `precision_curve.png`
- `recall_curve.png`

### 2. Train the small model with knowledge distillation

```bash
python train_rfi_smallKD.py --config config-rfi-kd.yaml 
```

Useful options:

```bash
python train_rfi_smallKD.py \
  --config config-rfi-kd.yaml \
  --transforms transforms.yaml \
  --device cuda
```

Notes:

- this trains `UNetSmall`
- KD behaviour is controlled by the `distill` block in `config-rfi-kd.yaml`
- The teacher checkpoint path is specified in `distill.teacher.ckpt`
- Optional W&B logging is controlled from `logger.wandb`

Outputs typically include:

- `best_student.pth`
- `last_student.pth`
- `student_epoch_XXX.pth`
- `metrics_student.csv`
- training-curve PNGs

### 3. Run the ablations

Small model from scratch:

```bash
python ablation/train_rfi_small_scratch.py --config ablation/config-rfi-small-scratch.yaml
```

Smaller KD student:

```bash
python ablation/train_rfi_smallKD_ablation.py --config ablation/config-rfi-kd-ablation.yaml --base-channel 6|8
```

## Inference, evaluation, and profiling

Use `inference.py` to run either the large or small model.

### Basic usage

Large model:

```bash
python inference.py --model-size large --config config-rfi-large.yaml  --thr 0.404 --weights /path/to/large/modelweight.pth --test-dir /path/to/testset
```

Small model:

```bash
python inference.py --model-size small --config config-rfi-kd.yaml  --thr 0.325  --weights /path/to/small/modelweight.pth --test-dir /path/to/testset
```

Inference for ablation experiments (Example here is for small model_scratch):

```bash
python inference_ablation.py --config config-rfi-small-scratch.yaml  --base-channels 16  --weights /path/to/smallmodel_scratch_training/modelweight.pth --test-dir /path/to/testset --thr 0.27
```

### What inference does

For the selected model, inference.py:

1. builds the dataset and dataloader
2. loads the selected checkpoint
3. evaluates segmentation metrics on the test set
4. profiles CPU inference
5. profiles GPU inference when CUDA is available
6. optionally generates prediction PNG masks
7. optionally runs post-processing on the prediction masks
8. Required threshold: large model= 0.404 and small modelKD= 0.325
9. Required threshold for ablation experiments: small model scratch= 0.27, small modelKD base-channel_6= 0.41, small modelKD base-channel_8= 0.313

### Profiling notes

The profiler reports:

- parameter count
- MACs/FLOPs when optional libraries are installed
- peak GPU memory
- latency
- fps
- estimated CPU/GPU power

Power estimation details:

- CPU power uses Intel RAPL if available
- GPU power uses `nvidia-smi` if available
- otherwise the script falls back to the CLI defaults:
  - `--fallback-cpu-power-watts 165.0`
  - `--fallback-gpu-power-watts 350.0`

## Generating prediction masks

Prediction mask export is used mainly for post-processing.

Example:

```bash
python inference.py \
  --model-size large \
  --cfg-path config-rfi-large.yaml \
  --weights /path/to/small/modelweight \
  --test-dir /path/to/testset \
  --run-postprocessing \
  --prediction-mask-dir /path/to/prediction_masks \
  --postprocessing-output-dir /path/to/postproc_outputs
```

Saved prediction files follow this pattern:

```text
<vv_filename_stem>_pred.png
```

## Post-processing

The post-processing step in `post_processing/rfi_postprocessing.py` uses local SAR power contrast to filter predicted RFI regions.

### Power image

A combined power image is computed from the original VV/VH data:

```text
|VV|^2 + |VH|^2
```

### Region filtering

For each connected component in the predicted mask, the script:

1. measures the region signal level
2. measures local background power from a ring around the region
3. computes the contrast:

```text
contrast_db = 10 * log10(region_power / background_power)
```

4. keeps or removes the region depending on the selected filter mode

### Main options

- `--threshold-db`
- `--filter-mode keep_high_rfi|remove_high_rfi`
- `--region-stat mean|median|p90`
- `--background-stat mean|median|trimmed_mean`
- `--min-region-area`
- `--ring-inner-iters`
- `--ring-outer-iters`
- `--trim-fraction`

Example:

```bash
python inference.py \
  --model-size small \
  --cfg-path config-rfi-kd.yaml \
  --weights /path/to/student_model.pth \
  --test-dir /path/to/test/range_compressed_rescaled \
  --run-postprocessing \
  --prediction-mask-dir /path/to/prediction_masks_small \
  --postprocessing-output-dir /path/to/postproc_outputs_small \
  --threshold-db 5.0 \
  --filter-mode remove_high_rfi
```

## Notebook

`inference_tutorial.ipynb` provides a notebook-based companion for running and inspecting inference interactively.



