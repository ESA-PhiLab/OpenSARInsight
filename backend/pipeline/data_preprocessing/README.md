# OpenSAR Insight – Data Augmentation and Normalization Module

## Overview
This repository provides PyTorch-compatible data augmentation and normalization techniques for SAR data, supporting both:

- **SLC amplitude data** 
- **L0 I/Q data** 

The module is designed for use in the OpenSAR Insight / AI4SAR project and enables transformations compatible with both focusing and detection heads.

---

## Features
### 1. SLC Amplitude Augmentation:
- **Random Translation** – shifts the image in azimuth & range with configurable padding.
- **Random Cropping** – random cropping.
- **Resizing** – resize back to a fixed size (default: `512x512`).
- **Speckle Noise Simulation** – Gamma-distributed multiplicative noise.

### 2. L0 I/Q Augmentation:
- **Amplitude Scaling** – multiplies per-pixel or per-patch amplitudes by a random factor.
- **Phase Perturbation** – introduces controlled random phase offsets.

### 3. SLC Amplitude Normalization:
- SLC amplitude → log compression → percentile clipping → [0,1] scaling

### 4. L0 I/Q Normalization:
- L0 I/Q (complex) data → per-channel z-score normalization

---

## Input Data Formats

| **Data Type** | **Shape**        | **Content**                     | **File Format** |
|--------------|-------------------|---------------------------------|------------------|
| L0 Raw       | (H, W)            | Complex float32 I/Q data       | `.dat` |
| L0 I/Q       | (H, W, 2)         | Real & Imaginary          | Generated on the fly when reading .dat |
| SLC Amplitude | (H, W)          | Magnitude-only amplitude       | `.tiff` |

---

## Usage

### 1. **For L0 I/Q and SLC Amplitude Normalization**
```python
from torchvision import transforms
from sar_normalizatin import NormalizeL0IQTransform, NormalizeSLCAmpTransform

norm = transforms.Compose([
    NormalizeL0IQTransform(),                 # expects sample["iq"] shape (H, W, 2)
    NormalizeSLCAmpTransform(1.0, 99.0),      # expects sample["slc"] shape (H, W)
])

sample = {"iq": iq_np_or_tensor, "slc": slc_np_or_tensor, "meta": {...}}
sample = norm(sample)
```

### 2. **For SLC Amplitude Augmentation**
```python
from torchvision import transforms
from sar_augmentation import SLCRandomTranslate, SLCRandomCrop, SLCResize, SLCSpeckleNoise

augmentation_pipeline = transforms.Compose([
    SLCRandomTranslate(max_shift_px=(8, 8), mode="reflect"),
    SLCRandomCrop(size=(480, 480), random=True),
    SLCResize(size=(512, 512), mode="bilinear"),
    SLCSpeckleNoise(looks=1.0)
])

sample = {"slc": slc_image}
augmented = augmentation_pipeline(sample)
```


### 3. **For range_compressed data Apmlitude and Phase Perturbation augmentation**
```python
from torchvision import transforms
from sar_augmentation import L0IQAmpPhasePerturb

augmentation_pipeline = transforms.Compose([
    L0IQAmpPhasePerturb(
        amp_scale_range=(0.95, 1.05),
        max_phase_shift=0.05,
        per_pixel=True
    )
])

sample = {"iq": iq_array}
augmented = augmentation_pipeline(sample)
```

### 4. **For range_compressed data Narrow Band Spectral Dropout augmentation**
```python
from torchvision import transforms
from sar_rc_augmentation import RCNarrowbandSpectralDropout

augmentation_pipeline = transforms.Compose([
    RCNarrowbandSpectralDropout(
        bands_range=[(0.48, 0.52)],         # Drop a thin range sub-bands
        bands_azimuth=[],                   # No azimuth dropout for vessel
        soft=True,                          # Smooth raised-cosine taper
        edge_taper=0.15                     # 15% taper inside each band
    )
])

sample = {"rc": rc_array}  # rc_array shape: (H, W, 2) I/Q range-compressed data
augmented = augmentation_pipeline(sample)
```

### 5. **For range_compressed data Subsampling And Interpolation augmentation**
```python
augmentation_pipeline = transforms.Compose([
    RCSubsampleAndInterpolate(
        az_factor=1,        # keep azimuth resolution
        rg_factor=2,        # gentle range softening
        down_mode="avg",    # anti-alias before subsample
        up_mode="bilinear"  # smooth resize back
    )
])

sample = {"rc": rc_array}  # rc_array shape: (H, W, 2) I/Q range-compressed data
augmented = augmentation_pipeline(sample)
```

---

## Recommended Augmentation Parameters for SLC Amplitude

| **Use Case** | **Crop Size** | **Resize** | **Translate (px)** | **Speckle Looks** |
| ------------ | ------------- | ---------- | ------------------ | ----------------- |
| **Flood**    | 480×480       | 512×512    | ±8    applied to time series othewise better not to use | 1.0| 
| **RFI**      | 500×500       | 512×512    | ±4                 | 3.0  |
| **Vessel**   | 512×512 or minimal cropping| 512×512    | ±12   | 1.5  |

---


---

## Recommended Augmentation Parameters for Apmlitude and Phase Perturbation

| **Use Case** | **Amp Range** | **Phase Perturbation**  | **Per Pixel** |  
| ------------ | ------------- | -------- | ---------------| 
| **Flood**    | (0.95, 1.05)  | 0.02     |  False          | 
| **RFI**      | (0.90, 1.10)  | 0.05     |  True          |                
| **Vessel**   | (0.85, 1.15)  | 0.10     |  True          |  

---


---

## Recommended Augmentation Parameters for Narrow Band Spectral Dropout

| **Use Case** | **bands_range** | **bands_azimuth** | **soft** | **edge_taper** |  
| ------------ | -------------   | ----------        | -------- | ---------------| 
| **Flood**    | [(0.48,0.52)]   | []                | True     |  0.15          | 
| **RFI**      | [(0.20,0.22)]   | []                | True     |  0.15          |                
| **Vessel**   | []              | [(0.45,0.55)]     | True     |  0.10          |  

---


---

## Recommended Augmentation Parameters for Subsample And Interpolate

| **Use Case** | **az_factor** | **rg_factor** | **down_mode** | **up_mode** |  **visual effect and why**|
| ------------ | ------------- | ----------  | -------- | ---------------|--------------- |
| **Flood**    | 1   | 2 | "avg"  | "bilinear" | Slight range softening only; preserves azimuth texture important for shorelines. Very mild visual change|
| **RFI**      | 1   | 1 | "avg"  | "bilinear" | Keep it neutral here (no resolution loss); combine with your spectral dropout for RFI. If you must touch resolution: set rg_factor=2, leave az at 1.|              
| **Vessel**   | 2   | 1 | "avg"  | "bilinear" | Tiny azimuth blur to mimic mild motion/resolution loss; targets vessel robustness without washing wakes. |

---

---

## Recommended Augmentation Parameters for Azimuth Defocus

| **Use Case** | **κappa (rad / (cycles/sample)²)** | **visual effect** |  
| ------------ | -------------| --------------------| 
| **Flood**    |         2.0e-4     | Mild azimuth softening; preserves shoreline/textural detail while introducing slight defocus.  |     
| **RFI**      |          0.0         | Keep neutral; azimuth defocus doesn’t help RFI and can smear narrow interference bands. |                  
| **Vessel**   |        6.0e-4                 |Noticeable but not heavy motion-like blur; stresses vessel/wake robustness without washing them out. | 

---

---
## Recommended Augmentation Parameters for Bandwidth Trim

| **Use Case** | **trim_frac_range** | **trim_frac_azimut** | **window** | **visual effect** |
| ------------ | ------------- | ---------- | ------------------ | ----------------- |
| **Flood**    | 0.8           | 0.95       | "hann"  | Slight range-resolution loss while keeping azimuth detail for edges.| 
| **RFI**      | 1.0           | 1.0        | "hann"  | Do not trim; trimming can alter narrow interference patterns you want to detect.|
| **Vessel**   | 0.9           | 0.9        | "hann"  | Mild resolution drop in both axes to test robustness; avoids erasing small vessels.  |

---


## Normalization Shapes and Types Summary 

| **Stage**             | **Key**   | **Expected Shape** | **Dtype In**                   | **Dtype Out**       |
| ----------------- | ----- | -------------- | -------------------------- | --------------- |
| L0 I/Q normalize  | `iq`  | `(H, W, 2)`    | `np.float32/torch.float32` | `torch.float32` |
| SLC amp normalize | `slc` | `(H, W)`       | `np.float32/torch.float32` | `torch.float32` |

---

## Visualization
The provided test scripts allow you to visualize augmentation techniques:

- **SLC amplitude** before & after augmentation.
- **L0 I/Q amplitude and phase**: Intensity, real, and imaginary components for validation.

Example for L0 visualization:
```bash
python test_l0_aug.py --dat sample.dat --amp_min 0.95 --amp_max 1.05 --max_phase 0.05 --per-pixel --save_png l0_visualization.png
```
