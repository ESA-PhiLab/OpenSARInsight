"""
Quick tester for range-compressed SAR augmentations.

- Supports inputs:
    • (H, W, 2) float I/Q
    • (H, W) complex -> (H, W, 2)
    • (1, W) complex or (W,) complex -> (1, W, 2)
- Saves side-by-side "Before vs After" PNGs:
    Amplitude for all augments:
        compare_amp_phase.png
        compare_spectral_dropout.png
        compare_bandwidth_trim.png
        compare_az_defocus.png
        compare_subsample_interp.png
    Phase (ONLY for RCAmplitudePhasePerturb), separate file:
        compare_amp_phase_PHASE.png
"""

from __future__ import annotations

import argparse
import logging
from torch import Tensor
import math
from typing import Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt

from sar_rc_augmentation import (
    RCAmplitudePhasePerturb,
    RCNarrowbandSpectralDropout,
    RCBandwidthTrim,
    RCAzimuthDefocus,
    RCSubsampleAndInterpolate,
)

# ---------------------- Logging ----------------------
def init_logging(level: int = logging.INFO) -> None:
    """Configure root logger once."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

LOGGER = logging.getLogger(__name__)


# ---------------------- I/O helpers ----------------------
def to_hw2_tensor(rc_np: np.ndarray) -> torch.Tensor:
    """
    Convert various RC shapes to a float32 torch.Tensor of shape (H, W, 2).

    Args:
        rc_np: Input RC array (complex or real with last-dim=2).

    Returns:
        Tensor of shape (H, W, 2), dtype float32.

    Raises:
        ValueError: If input shape/dtype is unsupported.
    """
    rc_np = np.asarray(rc_np)
    # Complex inputs → I/Q in last dim
    if np.iscomplexobj(rc_np):
        if rc_np.ndim == 1:
            # (W,) -> (1, W, 2)
            iq = np.stack([rc_np.real, rc_np.imag], axis=-1).astype(np.float32)[None, ...]
            return _to_float_tensor(iq)
        if rc_np.ndim == 2:
            # (H, W) or (1, W)
            if rc_np.shape[0] == 1:
                rc_np = rc_np.squeeze(0)  # (W,)
                iq = np.stack([rc_np.real, rc_np.imag], axis=-1).astype(np.float32)[None, ...]
            else:
                iq = np.stack([rc_np.real, rc_np.imag], axis=-1).astype(np.float32)
            return _to_float_tensor(iq)
        raise ValueError(f"Complex array must be 1D or 2D, got {rc_np.shape}")

    # Real inputs with I/Q in last dim
    if rc_np.ndim == 2 and rc_np.shape[-1] == 2:
        # (W,2) -> (1,W,2)
        return _to_float_tensor(rc_np.astype(np.float32)[None, ...])
    if rc_np.ndim == 3 and rc_np.shape[-1] == 2:
        # (H,W,2)
        return _to_float_tensor(rc_np.astype(np.float32))

    raise ValueError(f"Unsupported input shape/dtype: shape={rc_np.shape}, dtype={rc_np.dtype}")



def _to_float_tensor(arr: Any) -> Tensor:
    """
    Convert input to a float32 torch.Tensor.

    Accepts:
      • torch.Tensor  -> returns .float() (device preserved)
      • numpy.ndarray -> returns float32 CPU tensor.
        Tries zero-copy via from_numpy; if that fails under NumPy>=2
        or due to unsupported/negative strides, falls back to frombuffer,
        then to a copying path as last resort.
    """
    if isinstance(arr, Tensor):
        return arr.to(dtype=torch.float32)

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise TypeError("Input must be torch.Tensor or numpy.ndarray") from exc

    if isinstance(arr, np.ndarray):
        # Ensure float32 and positive, contiguous strides
        np_arr = np.asarray(arr, dtype=np.float32)
        if not (np_arr.flags.c_contiguous and all(s >= 0 for s in np_arr.strides)):
            np_arr = np.ascontiguousarray(np_arr, dtype=np.float32)

        # 1) Fast path: zero-copy if torch build supports current NumPy
        try:
            return _to_float_tensor(np_arr)
        except (RuntimeError, TypeError, ValueError):
            # 2) Buffer protocol path (bypasses NumPy C-API)
            try:
                t = torch.frombuffer(memoryview(np_arr),
                                     dtype=torch.float32,
                                     count=np_arr.size)
                return t.reshape(np_arr.shape)
            except Exception:
                # 3) Copying fallback
                return torch.tensor(np_arr, dtype=torch.float32)

    raise TypeError("Input must be torch.Tensor or numpy.ndarray")


# ---------------------- Plot helpers ----------------------
def _save_compare_1d(y_before: np.ndarray, y_after: np.ndarray, title: str, ylabel: str, out_png: str) -> None:
    plt.figure(figsize=(12, 3.2))
    plt.suptitle(title)
    plt.subplot(1, 2, 1)
    plt.plot(y_before)
    plt.title("Before")
    plt.xlabel("Range samples")
    plt.ylabel(ylabel)
    plt.subplot(1, 2, 2)
    plt.plot(y_after)
    plt.title("After")
    plt.xlabel("Range samples")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ---------------------- Signal helpers ----------------------
def amplitude(iq: torch.Tensor) -> torch.Tensor:
    """
    Compute magnitude sqrt(I^2 + Q^2) from an (..., 2)-last-dim I/Q tensor.

    Args:
        iq: Tensor with last dimension size 2 (I,Q), dtype float/half/double.

    Returns:
        Tensor of shape iq.shape[:-1], same device, float dtype.
    """
    if iq.ndim < 1 or iq.shape[-1] != 2:
        raise ValueError(f"amplitude() expects last dim==2 (I/Q), got shape {tuple(iq.shape)}")
    i = iq[..., 0]
    q = iq[..., 1]
    # torch.hypot is numerically stable for magnitude
    return torch.hypot(i, q).to(torch.float32)


def phase_wrapped(iq: torch.Tensor) -> torch.Tensor:
    """
    Compute wrapped phase angle in [-pi, pi] from an (..., 2)-last-dim I/Q tensor.

    Args:
        iq: Tensor with last dimension size 2 (I,Q).

    Returns:
        Tensor of shape iq.shape[:-1], dtype float32, values in [-pi, pi].
    """
    if iq.ndim < 1 or iq.shape[-1] != 2:
        raise ValueError(f"phase_wrapped() expects last dim==2 (I/Q), got shape {tuple(iq.shape)}")
    i = iq[..., 0]
    q = iq[..., 1]
    return torch.atan2(q, i).to(torch.float32)


from typing import Any  # add this near your imports if it's not present

def _to_float_tensor(arr: Any) -> Tensor:
    """
    Convert input to a float32 torch.Tensor.

    Accepts:
      • torch.Tensor  -> returns .float() (device preserved)
      • numpy.ndarray -> returns float32 CPU tensor.
        Tries zero-copy via from_numpy; if that fails under NumPy>=2
        or due to unsupported/negative strides, falls back to frombuffer,
        then to a copying path as last resort.
    """
    if isinstance(arr, torch.Tensor):
        return arr.to(dtype=torch.float32)

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise TypeError("Input must be torch.Tensor or numpy.ndarray") from exc

    if isinstance(arr, np.ndarray):
        # Ensure float32 and contiguous with non-negative strides
        np_arr = np.asarray(arr, dtype=np.float32)
        if not (np_arr.flags.c_contiguous and all(s >= 0 for s in np_arr.strides)):
            np_arr = np.ascontiguousarray(np_arr, dtype=np.float32)

        # 1) Fast path: zero-copy if ABI is compatible
        try:
            return torch.from_numpy(np_arr)
        except (RuntimeError, TypeError, ValueError):
            # 2) Buffer protocol path (bypasses NumPy C-API)
            try:
                t = torch.frombuffer(memoryview(np_arr), dtype=torch.float32, count=np_arr.size)
                return t.reshape(np_arr.shape)
            except Exception:
                # 3) Copying fallback
                return torch.tensor(np_arr, dtype=torch.float32)

    raise TypeError("Input must be torch.Tensor or numpy.ndarray")


def save_compare_amplitude(iq_before: torch.Tensor, iq_after: torch.Tensor, title: str, out_png: str) -> None:
    """
    Save side-by-side 'Before' vs 'After' amplitude comparison.
    Uses shared percentile clipping for fair brightness when 2D.
    """
    amp_b = amplitude(iq_before).detach().cpu().numpy()
    amp_a = amplitude(iq_after).detach().cpu().numpy()
    H, W = amp_b.shape

    if H == 1:
        _save_compare_1d(amp_b[0], amp_a[0], title, "Amplitude", out_png)
        return

    # 2D image: shared 99.5% percentile for display normalization
    p = float(max(np.percentile(amp_b, 99.5), 1e-9))
    plt.figure(figsize=(10, 4.2))
    plt.suptitle(title)
    plt.subplot(1, 2, 1)
    plt.imshow(np.clip(amp_b / p, 0, 1), cmap="gray", aspect="auto")
    plt.title("Before")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(np.clip(amp_a / p, 0, 1), cmap="gray", aspect="auto")
    plt.title("After")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_compare_phase(iq_before: torch.Tensor, iq_after: torch.Tensor, title: str, out_png: str) -> None:
    """
    Save side-by-side 'Before' vs 'After' PHASE comparison (wrapped [-pi, pi]).
    Uses fixed color scale to allow direct visual comparison.
    """
    ph_b = phase_wrapped(iq_before).detach().cpu().numpy()
    ph_a = phase_wrapped(iq_after).detach().cpu().numpy()
    H, W = ph_b.shape

    if H == 1:
        _save_compare_1d(ph_b[0], ph_a[0], title, "Phase [rad]", out_png)
        return

    vmin, vmax = -math.pi, math.pi
    plt.figure(figsize=(10, 4.2))
    plt.suptitle(title)
    plt.subplot(1, 2, 1)
    plt.imshow(ph_b, cmap="twilight", aspect="auto", vmin=vmin, vmax=vmax)
    plt.title("Before")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(ph_a, cmap="twilight", aspect="auto", vmin=vmin, vmax=vmax)
    plt.title("After")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ---------------------- Main ----------------------
def run(npy_path: str) -> None:
    rc_np = np.load(npy_path, allow_pickle=False)
    LOGGER.info("Loaded %s shape=%s dtype=%s", npy_path, rc_np.shape, rc_np.dtype)

    rc = to_hw2_tensor(rc_np)  # (H, W, 2)
    sample = {"rc": rc}

    # 1) Amplitude & Phase Perturbation
    aug_amp_phase = RCAmplitudePhasePerturb(
        amp_scale_range=(0.85, 1.15),
        max_phase_shift=0.1,
        per_pixel=True,
    )
    out = aug_amp_phase(sample)
    save_compare_amplitude(sample["rc"], out["rc"], "Amplitude & Phase Perturbation (Amplitude)", "compare_amp_phase.png")
    # NEW: separate PHASE comparison
    save_compare_phase(sample["rc"], out["rc"], "Amplitude & Phase Perturbation (Phase)", "compare_amp_phase_PHASE.png")

    # 2) Narrowband Spectral Dropout (center notch as example)
    aug_dropout = RCNarrowbandSpectralDropout(
        bands_range=[(0.45, 0.55)],   # drop a narrow range band
        bands_azimuth=[],             # none in azimuth for simplicity
        soft=True,
        edge_taper=0.1,
    )
    out = aug_dropout(sample)
    save_compare_amplitude(sample["rc"], out["rc"], "Narrowband Spectral Dropout", "compare_spectral_dropout.png")

    # 3) Bandwidth Trim 
    aug_trim = RCBandwidthTrim(
        trim_frac_range=0.96,
        trim_frac_azimuth=0.99,
        window="hann",
    )
    out = aug_trim(sample)
    save_compare_amplitude(sample["rc"], out["rc"], "Bandwidth Trim", "compare_bandwidth_trim.png")

    # 4) Azimuth Defocus (quadratic phase)
    aug_defocus = RCAzimuthDefocus(kappa=6.0e-4)
    out = aug_defocus(sample)
    save_compare_amplitude(sample["rc"], out["rc"], "Azimuth Defocus (κ=6.0e-4)", "compare_az_defocus.png")

    # 5) Subsample & Interpolate (mild downsample both axes)
    aug_sub = RCSubsampleAndInterpolate(
        az_factor=2,
        rg_factor=1,
        down_mode="avg",
        up_mode="bilinear",
    )
    out = aug_sub(sample)
    save_compare_amplitude(
        sample["rc"], out["rc"],
        "Subsample & Interpolate (2×1 avg→bilinear)",
        "compare_subsample_interp.png",
    )

    LOGGER.info("Saved:")
    LOGGER.info("  compare_amp_phase.png")
    LOGGER.info("  compare_amp_phase_PHASE.png")
    LOGGER.info("  compare_spectral_dropout.png")
    LOGGER.info("  compare_bandwidth_trim.png")
    LOGGER.info("  compare_az_defocus.png")
    LOGGER.info("  compare_subsample_interp.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare SAR RC augmentations (Amplitude; Phase separately for Amp/Phase perturb).")
    p.add_argument(
        "--npy",
        help="Path to .npy RC array (complex or I/Q in last dim).",
        
    )
    return p.parse_args()


def main() -> None:
    init_logging(logging.INFO)
    args = parse_args()
    run(args.npy)


if __name__ == "__main__":
    main()


