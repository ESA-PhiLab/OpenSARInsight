"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: RFI-postprocessing.py

Description:
    Post-process U-Net RFI prediction masks using local SAR power contrast.

    This script takes binary prediction masks produced by a U-Net model and
    compares each connected predicted RFI region against its local background
    power. The local contrast is computed in decibels using:

        contrast_db = 10 * log10(region_power / background_power)

    The SAR power image is computed from matching VV and VH complex
    range-compressed SAR files as:

        combined_power = |VV|^2 + |VH|^2

Key properties
--------------
- Loads U-Net prediction masks from PNG files.
- Resolves the matching original VV/VH complex SAR .npy files.
- Aligns VV and VH SAR arrays to the prediction mask shape using centre-crop
  or centre-padding.
- Computes combined SAR power from VV and VH channels.
- Labels connected regions in the predicted mask.
- Estimates each region's signal level using mean, median, or p90.
- Estimates local background power from a ring around each region.
- Computes region-to-background contrast in dB.
- Supports two filtering modes:
    - keep_high_rfi:
        Keep only regions above the dB threshold.
    - remove_high_rfi:
        Remove regions above the dB threshold and keep lower-contrast regions.
- Saves side-by-side comparison images:
    U-Net prediction | Post-processed mask

Recommended usage
-----------------
- Use this script after U-Net inference has generated prediction mask PNGs.
- Use it to visually inspect the effect of dB-based RFI post-processing.
- Use remove_high_rfi when high-power RFI-like regions should be suppressed.
- Use keep_high_rfi when only strong high-contrast RFI regions should be retained.


History:
    - 2026-04-14:
        First version of RFI post-processing utility.
    - 2026-04-18:
        Final version is ready.

---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2026-04-14

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, label


VALID_FILTER_MODES = {"keep_high_rfi", "remove_high_rfi"}
VALID_FILTER_LEVELS = {"region", "patch"}


@dataclass
class RegionResult:
    """
    Stores measurements for one connected RFI region.
    """
    region_id: int
    area_pixels: int
    region_value: float
    background_value: float
    contrast_db: float
    high_rfi: bool
    kept: bool
    bbox: Tuple[int, int, int, int]


def compute_region_stat(values: np.ndarray, method: str = "median") -> float:
    """
    Compute a representative signal level for a region.
    """
    if values.size == 0:
        raise ValueError("Region values are empty.")

    if method == "mean":
        return float(np.mean(values))

    if method == "median":
        return float(np.median(values))

    if method == "p90":
        return float(np.percentile(values, 90))

    raise ValueError(f"Unsupported region statistic: {method}")


def compute_background_stat(
    values: np.ndarray,
    method: str = "median",
    trim_fraction: float = 0.1,
) -> float:
    """
    Compute a robust estimate of the local background/speckle noise floor.
    """
    if values.size == 0:
        raise ValueError("Background values are empty.")

    if method == "mean":
        return float(np.mean(values))

    if method == "median":
        return float(np.median(values))

    if method == "trimmed_mean":
        if not (0.0 <= trim_fraction < 0.5):
            raise ValueError("trim_fraction must be in [0, 0.5).")

        sorted_vals = np.sort(values)
        n = sorted_vals.size
        k = int(n * trim_fraction)
        trimmed = sorted_vals[k:n - k] if n - 2 * k > 0 else sorted_vals

        return float(np.mean(trimmed))

    raise ValueError(f"Unsupported background statistic: {method}")


def compute_contrast_db(
    region_value: float,
    background_value: float,
    eps: float = 1e-12,
) -> float:
    """
    Compute contrast in dB between region and background for power data.

    Formula:
        10 * log10(region / background)
    """
    region_value = max(region_value, eps)
    background_value = max(background_value, eps)

    return float(10.0 * np.log10(region_value / background_value))


def get_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Return bounding box of a binary mask as:
        min_row, min_col, max_row, max_col
    """
    rows, cols = np.where(mask)

    if rows.size == 0 or cols.size == 0:
        return 0, 0, 0, 0

    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def extract_local_background(
    image: np.ndarray,
    region_mask: np.ndarray,
    full_rfi_mask: np.ndarray,
    ring_inner_iters: int = 3,
    ring_outer_iters: int = 12,
) -> np.ndarray:
    """
    Extract local background pixels from a ring around one predicted RFI region.

    The inner ring is excluded to avoid using pixels too close to the predicted RFI.
    Other predicted RFI pixels are also excluded from the background.
    """
    if ring_outer_iters <= ring_inner_iters:
        raise ValueError("ring_outer_iters must be greater than ring_inner_iters.")

    inner = binary_dilation(region_mask, iterations=ring_inner_iters)
    outer = binary_dilation(region_mask, iterations=ring_outer_iters)

    ring = outer & (~inner)
    background_mask = ring & (~full_rfi_mask.astype(bool))

    return image[background_mask]


def decide_region_keep(high_rfi: bool, filter_mode: str) -> bool:
    """
    Decide whether a region should be kept in the output mask.

    Modes:
        keep_high_rfi:
            Keep only regions above the dB threshold.

        remove_high_rfi:
            Remove regions above the dB threshold.
            Keep lower-RFI predicted regions.
    """
    if filter_mode == "keep_high_rfi":
        return high_rfi

    if filter_mode == "remove_high_rfi":
        return not high_rfi

    raise ValueError(
        f"Unsupported filter_mode={filter_mode}. "
        f"Expected one of: {sorted(VALID_FILTER_MODES)}"
    )


def filter_rfi_patch_by_db(
    power_image: np.ndarray,
    pred_mask: np.ndarray,
    threshold_db: float = 5.0,
    region_stat: str = "p90",
    background_stat: str = "median",
    trim_fraction: float = 0.1,
    filter_mode: str = "remove_high_rfi",
) -> Tuple[np.ndarray, List[RegionResult]]:
    """
    Filter entire patch based on aggregate RFI contrast.
    
    Much faster than region-by-region filtering.
    Returns either the original mask or a zero mask.
    """
    if filter_mode not in VALID_FILTER_MODES:
        raise ValueError(
            f"Unsupported filter_mode={filter_mode}. "
            f"Expected one of: {sorted(VALID_FILTER_MODES)}"
        )

    if power_image.ndim != 2:
        raise ValueError("power_image must be a 2D array.")

    if pred_mask.ndim != 2:
        raise ValueError("pred_mask must be a 2D array.")

    if power_image.shape != pred_mask.shape:
        raise ValueError(
            f"power_image and pred_mask must have the same shape. "
            f"Got power_image={power_image.shape}, pred_mask={pred_mask.shape}"
        )

    power_image = power_image.astype(np.float64)
    pred_mask_bool = pred_mask.astype(bool)

    # Check if there are any RFI pixels
    if not pred_mask_bool.any():
        return np.zeros_like(pred_mask, dtype=np.uint8), []

    # Get all RFI pixels
    rfi_pixels = power_image[pred_mask_bool]
    
    # Compute aggregate RFI power
    try:
        rfi_value = compute_region_stat(rfi_pixels, region_stat)
    except ValueError:
        # No valid RFI pixels
        return np.zeros_like(pred_mask, dtype=np.uint8), []

    # Get all non-RFI pixels as background
    background_pixels = power_image[~pred_mask_bool]
    
    if background_pixels.size == 0:
        # Entire patch is RFI - keep it
        return pred_mask.astype(np.uint8), []

    # Compute background power
    background_value = compute_background_stat(
        background_pixels,
        method=background_stat,
        trim_fraction=trim_fraction,
    )

    # Compute contrast
    contrast_db = compute_contrast_db(
        region_value=rfi_value,
        background_value=background_value,
    )

    high_rfi = contrast_db > threshold_db
    kept = decide_region_keep(high_rfi=high_rfi, filter_mode=filter_mode)

    if kept:
        filtered_mask = pred_mask.astype(np.uint8)
    else:
        filtered_mask = np.zeros_like(pred_mask, dtype=np.uint8)

    # Create a single result entry for the whole patch
    result = RegionResult(
        region_id=1,
        area_pixels=int(pred_mask_bool.sum()),
        region_value=rfi_value,
        background_value=background_value,
        contrast_db=contrast_db,
        high_rfi=high_rfi,
        kept=kept,
        bbox=get_bbox(pred_mask_bool),
    )

    return filtered_mask, [result]


def filter_rfi_regions_by_db(
    power_image: np.ndarray,
    pred_mask: np.ndarray,
    threshold_db: float = 5.0,
    region_stat: str = "p90",
    background_stat: str = "median",
    min_region_area: int = 20,
    ring_inner_iters: int = 3,
    ring_outer_iters: int = 12,
    trim_fraction: float = 0.1,
    filter_mode: str = "remove_high_rfi",
) -> Tuple[np.ndarray, List[RegionResult]]:
    """
    Filter connected RFI regions using local dB contrast.

    This function still returns the filtered mask for visual comparison,
    but the main script only saves the comparison image.
    """
    if filter_mode not in VALID_FILTER_MODES:
        raise ValueError(
            f"Unsupported filter_mode={filter_mode}. "
            f"Expected one of: {sorted(VALID_FILTER_MODES)}"
        )

    if power_image.ndim != 2:
        raise ValueError("power_image must be a 2D array.")

    if pred_mask.ndim != 2:
        raise ValueError("pred_mask must be a 2D array.")

    if power_image.shape != pred_mask.shape:
        raise ValueError(
            f"power_image and pred_mask must have the same shape. "
            f"Got power_image={power_image.shape}, pred_mask={pred_mask.shape}"
        )

    power_image = power_image.astype(np.float64)
    pred_mask = pred_mask.astype(bool)

    labeled, num_regions = label(pred_mask)
    filtered_mask = np.zeros_like(pred_mask, dtype=np.uint8)
    results: List[RegionResult] = []

    for region_id in range(1, num_regions + 1):
        region_mask = labeled == region_id
        area_pixels = int(region_mask.sum())
        bbox = get_bbox(region_mask)

        if area_pixels < min_region_area:
            results.append(
                RegionResult(
                    region_id=region_id,
                    area_pixels=area_pixels,
                    region_value=np.nan,
                    background_value=np.nan,
                    contrast_db=np.nan,
                    high_rfi=False,
                    kept=False,
                    bbox=bbox,
                )
            )
            continue

        region_pixels = power_image[region_mask]

        try:
            region_value = compute_region_stat(region_pixels, region_stat)
        except ValueError:
            results.append(
                RegionResult(
                    region_id=region_id,
                    area_pixels=area_pixels,
                    region_value=np.nan,
                    background_value=np.nan,
                    contrast_db=np.nan,
                    high_rfi=False,
                    kept=False,
                    bbox=bbox,
                )
            )
            continue

        background_pixels = extract_local_background(
            image=power_image,
            region_mask=region_mask,
            full_rfi_mask=pred_mask,
            ring_inner_iters=ring_inner_iters,
            ring_outer_iters=ring_outer_iters,
        )

        if background_pixels.size == 0:
            results.append(
                RegionResult(
                    region_id=region_id,
                    area_pixels=area_pixels,
                    region_value=region_value,
                    background_value=np.nan,
                    contrast_db=np.nan,
                    high_rfi=False,
                    kept=False,
                    bbox=bbox,
                )
            )
            continue

        background_value = compute_background_stat(
            background_pixels,
            method=background_stat,
            trim_fraction=trim_fraction,
        )

        contrast_db = compute_contrast_db(
            region_value=region_value,
            background_value=background_value,
        )

        high_rfi = contrast_db > threshold_db
        kept = decide_region_keep(high_rfi=high_rfi, filter_mode=filter_mode)

        if kept:
            filtered_mask[region_mask] = 1

        results.append(
            RegionResult(
                region_id=region_id,
                area_pixels=area_pixels,
                region_value=region_value,
                background_value=background_value,
                contrast_db=contrast_db,
                high_rfi=high_rfi,
                kept=kept,
                bbox=bbox,
            )
        )

    return filtered_mask, results


def load_mask_png(mask_path: str | Path) -> np.ndarray:
    """
    Load a PNG mask and return a binary mask with values 0 or 1.
    """
    mask = Image.open(mask_path).convert("L")
    mask_array = np.array(mask)

    return (mask_array > 0).astype(np.uint8)


def align_complex_sar_to_mask_shape(
    x: np.ndarray,
    mask_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Align a 2D complex SAR image to the mask shape.

    If image is larger than mask, centre-crop.
    If image is smaller than mask, centre-pad with zeros.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D complex SAR image, got shape {x.shape}")

    img_h, img_w = x.shape
    mask_h, mask_w = mask_shape

    if img_h > mask_h:
        extra = img_h - mask_h
        top = extra // 2
        x = x[top:top + mask_h, :]
    elif img_h < mask_h:
        extra = mask_h - img_h
        pad_top = extra // 2
        pad_bottom = extra - pad_top
        x = np.pad(
            x,
            ((pad_top, pad_bottom), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    img_h, img_w = x.shape

    if img_w > mask_w:
        extra = img_w - mask_w
        left = extra // 2
        x = x[:, left:left + mask_w]
    elif img_w < mask_w:
        extra = mask_w - img_w
        pad_left = extra // 2
        pad_right = extra - pad_left
        x = np.pad(
            x,
            ((0, 0), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )

    return x


def load_combined_power_from_vv_vh_aligned_to_mask(
    npy_path: str | Path,
    mask_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Load matching VV and VH complex SAR files, align to mask shape,
    and compute:

        combined_power = |VV|^2 + |VH|^2
    """
    npy_path = Path(npy_path)
    name_lower = npy_path.name.lower()

    if "-vh-" in name_lower:
        vh_path = npy_path
        vv_path = npy_path.with_name(npy_path.name.replace("-vh-", "-vv-"))
    elif "-vv-" in name_lower:
        vv_path = npy_path
        vh_path = npy_path.with_name(npy_path.name.replace("-vv-", "-vh-"))
    else:
        raise ValueError(
            f"Could not determine VV/VH pair from filename: {npy_path.name}"
        )

    if not vv_path.exists():
        raise FileNotFoundError(f"Matching VV file not found: {vv_path}")

    if not vh_path.exists():
        raise FileNotFoundError(f"Matching VH file not found: {vh_path}")

    vv = np.load(vv_path)
    vh = np.load(vh_path)

    if vv.ndim != 2 or vh.ndim != 2:
        raise ValueError(
            f"Expected 2D complex SAR arrays. Got VV {vv.shape}, VH {vh.shape}"
        )

    if not np.iscomplexobj(vv):
        raise ValueError(f"VV file is not complex: dtype={vv.dtype}")

    if not np.iscomplexobj(vh):
        raise ValueError(f"VH file is not complex: dtype={vh.dtype}")

    vv = align_complex_sar_to_mask_shape(vv, mask_shape)
    vh = align_complex_sar_to_mask_shape(vh, mask_shape)

    vv_power = np.abs(vv) ** 2
    vh_power = np.abs(vh) ** 2

    combined_power = vv_power + vh_power

    return combined_power.astype(np.float64)


def resolve_matching_npy_from_prediction(
    pred_mask_path: str | Path,
    sar_data_dir: str | Path,
    pred_suffix: str = "_pred",
) -> Path:
    """
    Resolve original SAR .npy file from a prediction PNG.

    Expected prediction filename:
        <original_npy_stem>_pred.png
    """
    pred_mask_path = Path(pred_mask_path)
    sar_data_dir = Path(sar_data_dir)

    stem = pred_mask_path.stem

    if stem.endswith(pred_suffix):
        npy_stem = stem[: -len(pred_suffix)]
    else:
        npy_stem = stem

    candidate = sar_data_dir / f"{npy_stem}.npy"

    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Could not find matching SAR .npy for prediction mask: {pred_mask_path}. "
        f"Expected: {candidate}"
    )


def save_unet_and_postprocessed_side_by_side(
    unet_mask: np.ndarray,
    post_mask: np.ndarray,
    output_path: str | Path,
    left_title: str = "U-Net prediction",
    right_title: str = "Post-processed",
) -> None:
    """
    Save a side-by-side PNG:
        U-Net prediction | Post-processed
    """
    unet_img = Image.fromarray((unet_mask > 0).astype(np.uint8) * 255).convert("RGB")
    post_img = Image.fromarray((post_mask > 0).astype(np.uint8) * 255).convert("RGB")

    if unet_img.size != post_img.size:
        post_img = post_img.resize(unet_img.size, resample=Image.Resampling.NEAREST)

    w, h = unet_img.size
    gap = 16
    title_h = max(32, min(64, h // 18))

    canvas_w = (2 * w) + gap
    canvas_h = h + title_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, canvas_w, title_h], fill=(0, 0, 0))

    x_left = 0
    x_right = w + gap

    canvas.paste(unet_img, (x_left, title_h))
    canvas.paste(post_img, (x_right, title_h))

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(14, title_h - 12))
    except Exception:
        font = ImageFont.load_default()

    def centered_x(x_start: int, width: int, text: str) -> int:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        return x_start + max(0, (width - text_w) // 2)

    bbox_ref = draw.textbbox((0, 0), "Hg", font=font)
    text_h = bbox_ref[3] - bbox_ref[1]
    y_text = max(0, (title_h - text_h) // 2)

    draw.text(
        (centered_x(x_left, w, left_title), y_text),
        left_title,
        fill=(255, 255, 255),
        font=font,
    )

    draw.text(
        (centered_x(x_right, w, right_title), y_text),
        right_title,
        fill=(255, 255, 255),
        font=font,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def process_prediction_folder(
    sar_data_dir: str | Path,
    prediction_mask_dir: str | Path,
    output_dir: str | Path,
    threshold_db: float = 5.0,
    region_stat: str = "p90",
    background_stat: str = "median",
    min_region_area: int = 20,
    ring_inner_iters: int = 3,
    ring_outer_iters: int = 12,
    trim_fraction: float = 0.1,
    pred_suffix: str = "_pred",
    filter_mode: str = "remove_high_rfi",
    filter_level: str = "region",
    save_results: bool = True,
) -> Tuple[int, int]:
    """
    Process a directory of U-Net prediction masks by applying dB-based
    post-processing and saving side-by-side comparison images of the
    original prediction and the filtered result.

    For each prediction PNG in prediction_mask_dir this function:
    - resolves the matching .npy SAR power file,
    - loads the prediction mask and combined VV/VH power image aligned to it,
    - applies dB thresholding and connected-region statistics,
    - optionally removes or keeps suspected RFI regions according to
      filter_mode, and
    - writes only a side-by-side comparison image to
      output_dir/comparisons_{filter_mode}.

    Args:
        sar_data_dir (str | Path): Directory containing raw SAR .npy files (VV and VH combined power).
        prediction_mask_dir (str | Path): Directory with prediction PNG masks produced by the U-Net.
        output_dir (str | Path): Directory to write comparison images.
        threshold_db (float): dB threshold used to decide RFI candidate regions.
        region_stat (str): Statistic to compute on detected regions (e.g. "p90").
        background_stat (str): Statistic used for background estimation (e.g. "median").
        min_region_area (int): Minimum area (pixels) for a region to be considered.
        ring_inner_iters (int): Iterations for inner ring morphological operation.
        ring_outer_iters (int): Iterations for outer ring morphological operation.
        trim_fraction (float): Fraction of border to ignore when computing background.
        pred_suffix (str): Suffix used to map prediction PNG names to .npy files.
        filter_mode (str): Filtering behaviour; must be one of VALID_FILTER_MODES.
        filter_level (str): Filtering granularity: "region" for per-region filtering (slower, more accurate),
            "patch" for whole-patch filtering (faster, simpler).
        save_results (bool): Whether to save comparison images.
    """
    if filter_mode not in VALID_FILTER_MODES:
        raise ValueError(
            f"Unsupported filter_mode={filter_mode}. "
            f"Expected one of: {sorted(VALID_FILTER_MODES)}"
        )

    if filter_level not in VALID_FILTER_LEVELS:
        raise ValueError(
            f"Unsupported filter_level={filter_level}. "
            f"Expected one of: {sorted(VALID_FILTER_LEVELS)}"
        )

    sar_data_dir = Path(sar_data_dir)
    prediction_mask_dir = Path(prediction_mask_dir)
    output_dir = Path(output_dir)

    comparison_dir = output_dir / f"comparisons_{filter_mode}"
    if save_results:
        comparison_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = sorted(prediction_mask_dir.glob("*.png"))

    if not pred_paths:
        raise FileNotFoundError(
            f"No prediction PNG files found in: {prediction_mask_dir}"
        )

    print(f"Found {len(pred_paths)} prediction masks.")
    print(f"Filter mode: {filter_mode}")
    print(f"Filter level: {filter_level}")
    print(f"Threshold:   {threshold_db:.2f} dB")
    if save_results:
        print(f"Saving only comparison images to: {comparison_dir}")
    else:
        print("Only post-processing enabled: post-processing will run, but no files will be saved.")

    successful = 0
    failed = 0
    
    # Track patch-level statistics for patch mode
    patch_kept_list = []
    patch_rejected_list = []

    for pred_mask_path in pred_paths:
        print(f"\nProcessing: {pred_mask_path.name}")

        try:
            npy_path = resolve_matching_npy_from_prediction(
                pred_mask_path=pred_mask_path,
                sar_data_dir=sar_data_dir,
                pred_suffix=pred_suffix,
            )

            pred_mask = load_mask_png(pred_mask_path)

            power_image = load_combined_power_from_vv_vh_aligned_to_mask(
                npy_path=npy_path,
                mask_shape=pred_mask.shape,
            )

            if filter_level == "patch":
                filtered_mask, results = filter_rfi_patch_by_db(
                    power_image=power_image,
                    pred_mask=pred_mask,
                    threshold_db=threshold_db,
                    region_stat=region_stat,
                    background_stat=background_stat,
                    trim_fraction=trim_fraction,
                    filter_mode=filter_mode,
                )
            else:  # filter_level == "region"
                filtered_mask, results = filter_rfi_regions_by_db(
                    power_image=power_image,
                    pred_mask=pred_mask,
                    threshold_db=threshold_db,
                    region_stat=region_stat,
                    background_stat=background_stat,
                    min_region_area=min_region_area,
                    ring_inner_iters=ring_inner_iters,
                    ring_outer_iters=ring_outer_iters,
                    trim_fraction=trim_fraction,
                    filter_mode=filter_mode,
                )

            base_name = pred_mask_path.stem

            comparison_path = (
                comparison_dir
                / f"{base_name}_unet_vs_{filter_mode}_th{threshold_db:g}db.png"
            )

            if filter_mode == "keep_high_rfi":
                right_title = f"Kept high RFI > {threshold_db:g} dB"
            else:
                right_title = f"Removed high RFI > {threshold_db:g} dB"

            if save_results:
                save_unet_and_postprocessed_side_by_side(
                    unet_mask=pred_mask,
                    post_mask=filtered_mask,
                    output_path=comparison_path,
                    left_title="U-Net prediction",
                    right_title=right_title,
                )

            valid_contrasts = [
                r.contrast_db for r in results if not np.isnan(r.contrast_db)
            ]

            print(f"Matching NPY:     {npy_path}")
            print(f"Regions found:    {len(results)}")
            print(f"High-RFI regions: {sum(r.high_rfi for r in results)}")
            print(f"Regions kept:     {sum(r.kept for r in results)}")
            if save_results:
                print(f"Comparison:       {comparison_path}")
            else:
                print("Comparison:       not saved images")

            if valid_contrasts:
                print(f"Contrast min:     {np.min(valid_contrasts):.4f}")
                print(f"Contrast max:     {np.max(valid_contrasts):.4f}")
                print(f"Contrast mean:    {np.mean(valid_contrasts):.4f}")
                print(f"Contrast median:  {np.median(valid_contrasts):.4f}")
            else:
                print("No valid contrast values found.")

            # Track patch-level decision for patch mode
            if filter_level == "patch" and results:
                patch_name = pred_mask_path.stem
                if results[0].kept:
                    patch_kept_list.append(patch_name)
                else:
                    patch_rejected_list.append(patch_name)

            successful += 1

        except Exception as exc:
            failed += 1
            print(f"FAILED: {pred_mask_path.name}")
            print(f"Reason: {exc}")

    print("\nFinished folder post-processing.")
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    
    # Print and save patch-level summary for patch mode
    if filter_level == "patch":
        num_kept = len(patch_kept_list)
        num_rejected = len(patch_rejected_list)
        print(f"\nPatch-level filtering summary:")
        print(f"Patches kept:     {num_kept}")
        print(f"Patches rejected: {num_rejected}")
        print(f"Total processed:  {num_kept + num_rejected}")
        
        # Save patch decision list to file
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_file = output_dir / f"patch_decisions_{filter_mode}_th{threshold_db:g}db.txt"
        
        with open(summary_file, "w") as f:
            f.write(f"RFI Post-Processing Patch Decisions\n")
            f.write(f"=====================================\n")
            f.write(f"Filter mode:      {filter_mode}\n")
            f.write(f"Filter level:     {filter_level}\n")
            f.write(f"Threshold (dB):   {threshold_db}\n")
            f.write(f"Total patches:    {num_kept + num_rejected}\n")
            f.write(f"Patches kept:     {num_kept}\n")
            f.write(f"Patches rejected: {num_rejected}\n")
            f.write(f"\n")
            
            f.write(f"KEPT PATCHES ({num_kept}):\n")
            f.write(f"-" * 50 + "\n")
            for patch_name in patch_kept_list:
                f.write(f"{patch_name}\n")
            
            f.write(f"\n")
            f.write(f"REJECTED PATCHES ({num_rejected}):\n")
            f.write(f"-" * 50 + "\n")
            for patch_name in patch_rejected_list:
                f.write(f"{patch_name}\n")
        
        print(f"Patch decisions saved to: {summary_file}")
    
    if save_results:
        print(f"Output dir: {output_dir}")
        print(f"Comparison images: {comparison_dir}")
    else:
        print("Output dir: images not saved")
        print("Comparison images: not saved images")

    return successful, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process U-Net RFI prediction masks using SAR power contrast. "
            "Only side-by-side comparison images are saved."
        )
    )

    parser.add_argument(
        "--sar-data-dir",
        type=Path,
        required=True,
        help="Folder containing original VV/VH complex SAR .npy files.",
    )

    parser.add_argument(
        "--prediction-mask-dir",
        type=Path,
        required=True,
        help="Folder containing generated U-Net prediction PNG masks.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output folder for side-by-side comparison PNG files.",
    )

    parser.add_argument(
        "--filter-mode",
        type=str,
        choices=sorted(VALID_FILTER_MODES),
        default="remove_high_rfi",
        help=(
            "remove_high_rfi removes regions above the dB threshold. "
            "keep_high_rfi keeps only regions above the dB threshold."
        ),
    )

    parser.add_argument(
        "--filter-level",
        type=str,
        choices=sorted(VALID_FILTER_LEVELS),
        default="region",
        help=(
            "region: filter individual connected regions (slower, more precise). "
            "patch: filter entire patch based on aggregate statistics (faster, simpler)."
        ),
    )

    parser.add_argument(
        "--threshold-db",
        type=float,
        default=5.0,
        help="dB threshold used to decide whether a region is high RFI.",
    )

    parser.add_argument(
        "--region-stat",
        type=str,
        choices=["mean", "median", "p90"],
        default="p90",
        help="Statistic used to summarise power inside each predicted region.",
    )

    parser.add_argument(
        "--background-stat",
        type=str,
        choices=["mean", "median", "trimmed_mean"],
        default="median",
        help="Statistic used to estimate local background power.",
    )

    parser.add_argument(
        "--min-region-area",
        type=int,
        default=20,
        help="Minimum connected-region area in pixels. Smaller regions are removed.",
    )

    parser.add_argument(
        "--ring-inner-iters",
        type=int,
        default=3,
        help="Inner dilation iterations for excluding pixels close to the region.",
    )

    parser.add_argument(
        "--ring-outer-iters",
        type=int,
        default=12,
        help="Outer dilation iterations for defining the local background ring.",
    )

    parser.add_argument(
        "--trim-fraction",
        type=float,
        default=0.1,
        help="Trim fraction used only when background_stat=trimmed_mean.",
    )

    parser.add_argument(
        "--pred-suffix",
        type=str,
        default="_pred",
        help="Suffix used in prediction PNG names before .png.",
    )

    parser.add_argument(
    "--no-save-results",
    action="store_true",
    help=(
        "Run post-processing without saving comparison images. "
        "Useful for timing or checking statistics only."
         ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    process_prediction_folder(
        sar_data_dir=args.sar_data_dir,
        prediction_mask_dir=args.prediction_mask_dir,
        output_dir=args.output_dir,
        threshold_db=args.threshold_db,
        region_stat=args.region_stat,
        background_stat=args.background_stat,
        min_region_area=args.min_region_area,
        ring_inner_iters=args.ring_inner_iters,
        ring_outer_iters=args.ring_outer_iters,
        trim_fraction=args.trim_fraction,
        pred_suffix=args.pred_suffix,
        filter_mode=args.filter_mode,
        filter_level=args.filter_level,
        save_results=not args.no_save_results,
    )