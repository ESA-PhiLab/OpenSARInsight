"""
augment_only_rc_save_each_aug_no_args.py

Reads range-compressed SAR .npy files (complex (H,W) or I/Q (H,W,2)),
applies each selected augmentation separately (using your classes from
`sar_rc_augmentation`) via torchvision.transforms.Compose, and saves the
augmented I/Q as (H, W, 2) float32 .npy with a suffix per augmentation.

No argparse; edit the CONFIG block below.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

try:
    from torchvision.transforms import Compose as TVCompose
    Compose = TVCompose
except Exception:  # fallback so the script still runs if torchvision isn't installed
    class Compose:  # minimal drop-in
        def __init__(self, transforms: Sequence[Callable[[Mapping[str, Any]], Mapping[str, Any]]]) -> None:
            self.transforms = list(transforms)
        def __call__(self, x: Mapping[str, Any]) -> Mapping[str, Any]:
            out = x
            for t in self.transforms:
                out = t(out)
            return out

# --- Import YOUR augmentation classes ---
from sar_rc_augmentation import (
    RCAmplitudePhasePerturb,
    RCNarrowbandSpectralDropout,
    RCBandwidthTrim,
    RCAzimuthDefocus,
    RCSubsampleAndInterpolate,
)
from dataset_generation_scripts.utils import get_config
cfg = get_config("MODEL_USECASES_PATH")

# =========================
# CONFIG — edit these
# =========================
INPUTS: List[Union[str, Path]] = cfg["dataloader_test"]["input_paths"]
OUT_DIR: Union[str, Path] = cfg["dataloader_test"]["output_path"]
AUG_REPEATS: int = 1   # run each augmentation this many times per input

# Enable/disable branches (each aug saved separately)
ENABLE_AMP_PHASE: bool = True
ENABLE_DROPOUT: bool = True
ENABLE_TRIM: bool = True
ENABLE_DEFOCUS: bool = True
ENABLE_SUBSAMPLE: bool = True

# Amp/Phase params
AMP_SCALE_MIN: float = 0.95
AMP_SCALE_MAX: float = 1.05
MAX_PHASE_SHIFT: float = 0.2

# Dropout params
DROPOUT_BANDS_RANGE: List[Tuple[float, float]] = [(0.48, 0.52)]  # normalized frequency bands to drop in range
DROPOUT_SOFT: bool = False
DROPOUT_EDGE_TAPER: float = 0.1

# Trim params
TRIM_FRAC_RANGE: float = 0.96     # keep fraction in range (0,1]
TRIM_FRAC_AZIMUTH: float = 0.99   # keep fraction in azimuth (0,1]
TRIM_WINDOW: str = "hann"         # "hann" | "hamming" | "kaiser" | "rect"

# Defocus params
KAPPA: float = 2.0e-4

# Subsample/Interpolate params
AZ_FACTOR: int = 2
RG_FACTOR: int = 2
DOWN_MODE: str = "avg"            # "avg" | "nearest"
UP_MODE: str = "bilinear"         # "bilinear" | "nearest"

# Optional: set seeds for reproducibility of stochastic augs (comment out to keep random)
SEED_TORCH: Optional[int] = None
SEED_NUMPY: Optional[int] = None

LOG_LEVEL = logging.INFO
# =========================


def init_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def _safe_torch_from_numpy(a: np.ndarray) -> Tensor:
    """Robust NumPy->torch float32 conversion (NumPy 2.x safe)."""
    a = np.asarray(a, dtype=np.float32)
    if not (a.flags.c_contiguous and all(s >= 0 for s in a.strides)):
        a = np.ascontiguousarray(a, dtype=np.float32)
    try:
        return torch.from_numpy(a)
    except (RuntimeError, TypeError, ValueError):
        try:
            t = torch.frombuffer(memoryview(a), dtype=torch.float32, count=a.size)
            return t.reshape(a.shape)
        except Exception:
            return torch.tensor(a, dtype=torch.float32)


def to_hw2_tensor(rc_np: np.ndarray) -> torch.Tensor:
    """
    Convert range-compressed array to float32 torch.Tensor (H, W, 2).

    Accepts:
      - Complex: (H,W) or (W,) or (1,W)
      - I/Q:     (H,W,2) or (W,2)
    """
    rc_np = np.asarray(rc_np)
    if np.iscomplexobj(rc_np):
        if rc_np.ndim == 1:
            iq = np.stack([rc_np.real, rc_np.imag], axis=-1).astype(np.float32)[None, ...]  # (1,W,2)
            return _safe_torch_from_numpy(iq)
        if rc_np.ndim == 2:
            if rc_np.shape[0] == 1:
                w = rc_np.squeeze(0)
                iq = np.stack([w.real, w.imag], axis=-1).astype(np.float32)[None, ...]      # (1,W,2)
            else:
                iq = np.stack([rc_np.real, rc_np.imag], axis=-1).astype(np.float32)         # (H,W,2)
            return _safe_torch_from_numpy(iq)
        raise ValueError(f"Complex array must be 1D or 2D, got {rc_np.shape}")

    # Real I/Q with last dim==2
    if rc_np.ndim == 2 and rc_np.shape[-1] == 2:
        return _safe_torch_from_numpy(rc_np.astype(np.float32)[None, ...])  # (1,W,2)
    if rc_np.ndim == 3 and rc_np.shape[-1] == 2:
        return _safe_torch_from_numpy(rc_np.astype(np.float32))             # (H,W,2)

    raise ValueError(f"Unsupported input shape/dtype: shape={rc_np.shape}, dtype={rc_np.dtype}")



class RCNpyDataset(Dataset):
    """
    Yields dicts: {'rc': (H,W,2) float32 tensor, 'meta': {'source': str, 'aug_idx': int}}.
    If aug_repeats > 1, each file appears that many times to re-run stochastic augs.
    """
    def __init__(self, paths: Sequence[Union[str, Path]], aug_repeats: int = 1) -> None:
        if aug_repeats < 1:
            raise ValueError("aug_repeats must be >= 1")
        self.items: List[Tuple[Path, int]] = [(Path(p), r) for p in paths for r in range(aug_repeats)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p, r = self.items[idx]
        arr = np.load(str(p), allow_pickle=False)
        rc = to_hw2_tensor(arr)  # (H,W,2)
        return {"rc": rc, "meta": {"source": str(p), "aug_idx": r}}


def build_aug_branches() -> List[Tuple[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]]]:
    """
    Build a list of (suffix_name, transform_callable) for each enabled augmentation.
    Each callable will be wrapped as Compose([aug]) during execution.
    """
    branches: List[Tuple[str, Callable]] = []

    if ENABLE_AMP_PHASE:
        branches.append((
            "amp_phase",
            RCAmplitudePhasePerturb(
                amp_scale_range=(AMP_SCALE_MIN, AMP_SCALE_MAX),
                max_phase_shift=MAX_PHASE_SHIFT,
                per_pixel=False,
            ),
        ))

    if ENABLE_DROPOUT:
        branches.append((
            "dropout",
            RCNarrowbandSpectralDropout(
                bands_range=DROPOUT_BANDS_RANGE,
                bands_azimuth=[],
                soft=DROPOUT_SOFT,
                edge_taper=DROPOUT_EDGE_TAPER,
            ),
        ))

    if ENABLE_TRIM:
        branches.append((
            "trim",
            RCBandwidthTrim(
                trim_frac_range=TRIM_FRAC_RANGE,
                trim_frac_azimuth=TRIM_FRAC_AZIMUTH,
                window=TRIM_WINDOW,
            ),
        ))

    if ENABLE_DEFOCUS:
        branches.append(("defocus", RCAzimuthDefocus(kappa=KAPPA)))

    if ENABLE_SUBSAMPLE:
        branches.append((
            "subinterp",
            RCSubsampleAndInterpolate(
                az_factor=AZ_FACTOR,
                rg_factor=RG_FACTOR,
                down_mode=DOWN_MODE,
                up_mode=UP_MODE,
            ),
        ))

    if not branches:
        raise ValueError("No augmentations enabled. Turn on ENABLE_* toggles in CONFIG.")

    return branches


def save_iq_npy(rc: Tensor, src: str, aug_idx: int, out_dir: Path, suffix: str) -> Path:
    """
    Save (H, W, 2) float32 I/Q tensor as .npy with an augmentation suffix.
    File name: <base>__<suffix>__augXXX.npy
    """
    if rc.ndim == 4 and rc.size(0) == 1:
        rc = rc[0]
    if rc.ndim != 3 or rc.size(-1) != 2:
        raise ValueError(f"Expected rc shape (H,W,2), got {tuple(rc.shape)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(src).stem
    out_path = out_dir / f"{base}__{suffix}__aug{aug_idx:03d}.npy"
    np.save(str(out_path), rc.detach().cpu().numpy().astype(np.float32), allow_pickle=False)
    return out_path


def run() -> None:
    init_logging(LOG_LEVEL)

    # Optional reproducibility
    if SEED_TORCH is not None:
        torch.manual_seed(SEED_TORCH)
    if SEED_NUMPY is not None:
        np.random.seed(SEED_NUMPY)

    inputs = [Path(p) for p in INPUTS]
    for x in inputs:
        if not x.is_file():
            raise FileNotFoundError(f"Not found: {x}")

    branches = build_aug_branches()
    ds = RCNpyDataset(inputs, aug_repeats=AUG_REPEATS)
    loader = DataLoader(ds, batch_size=1, shuffle=False, pin_memory=False, num_workers=0)

    out_dir = Path(OUT_DIR)
    saved_paths: List[str] = []

    for batch in loader:
        rc = batch["rc"]                   # (1,H,W,2)
        meta = batch["meta"]               # dict of lists under default collate
        src = meta["source"][0]
        aug_idx = int(meta["aug_idx"][0])

        for suffix, aug in branches:
            pipeline = Compose([aug])      # requirement: use Compose (even single aug)
            out_sample = pipeline({"rc": rc, "meta": meta})
            out_rc = out_sample["rc"]
            out_path = save_iq_npy(out_rc, src, aug_idx, out_dir, suffix=suffix)
            logging.info("Saved %s", out_path)
            saved_paths.append(str(out_path))

    logging.info("Done. Saved %d files to %s", len(saved_paths), out_dir)


if __name__ == "__main__":
    run()
