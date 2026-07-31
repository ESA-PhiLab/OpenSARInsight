"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: rc_dataloader.py

Description:
Range-Compressed (RC) SAR DataLoader with Compose-friendly external transforms.

- Loads RC SAR .npy as (2, H, W) float32 [I,Q]
- Optional masks (H, W) int labels; padding uses ignore_index (default 255)
- Uses torchvision.transforms.Compose with your custom RC normalization/augs
- Collate pads to batch maxima (optionally to nearest stride multiple)
- Builds per-sample valid_mask for loss/metric masking
- Strict size check ensures mask geometry matches RC after transforms

Usage
-----
python rc_dataloader.py \
  --data_dir /path/to/rc_npy \
  --config   aug.yaml \
  --use_masks true \
  --batch_size 4 \
  --num_workers 4 \
  --stride_multiple 32 \
  --enforce_stride_multiple true \
  --bypass_padding false \
  --max_batches 2

History:
  - 2025-10-01: First version for range-compressed data
  - 2025-10-09: Mask resolver + mask loading added; strict geometry checks


Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-09-30
© INDRA DEIMOS, 2025. All rights reserved.
---------------------------------------------------------------------
"""
from __future__ import annotations
import argparse
import glob
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
import json
from logging_setup import init_logging
from pipeline.data_preprocessing.normalization.sar_normalization import NormalizeRCTransform
from pipeline.data_preprocessing.augmentation.sar_rc_augmentation import (
    RCAmplitudePhasePerturb,
    RCNarrowbandSpectralDropout,
    RCBandwidthTrim,
    RCAzimuthDefocus,
    RCSubsampleAndInterpolate,
)

# ------------------------- Logging ------------------------- #
init_logging(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

# ------------------------- Types --------------------------- #
Sample = Dict[str, Any]  # {"rc": Tensor(2,H,W), "mask": Optional[Tensor(H,W)], "meta": dict}


# ------------------------- Utils --------------------------- #
def shape_tuple(x: torch.Tensor | np.ndarray) -> tuple[int, ...]:
    """Return a plain Python shape tuple of ints."""
    return tuple(int(d) for d in x.shape)


def check_stride_ready(h: int, w: int, stride: int) -> str:
    """Human-readable remainder check for stride alignment."""
    if stride <= 1:
        return "no stride enforcement"
    return f"H%{stride}={(h % stride)} W%{stride}={(w % stride)}"


def tensor_from_npy(path: Path) -> Tensor:
    """
    Load a .npy RC array and return torch.float32 tensor with shape (2, H, W).
    Accepts complex (H,W), channel-last (H,W,2), or channel-first (2,H,W).
    """
    arr = np.load(str(path), mmap_mode="r")

    if np.iscomplexobj(arr) and arr.ndim == 2:
        i = np.asarray(arr.real, dtype=np.float32)
        q = np.asarray(arr.imag, dtype=np.float32)
        rc_np = np.stack([i, q], axis=0)  # (2,H,W)
    elif arr.ndim == 3 and arr.shape[-1] == 2:  # (H,W,2)
        rc_np = arr.astype(np.float32).transpose(2, 0, 1)  # -> (2,H,W)
    elif arr.ndim == 3 and arr.shape[0] == 2:  # (2,H,W)
        rc_np = arr.astype(np.float32)
    else:
        raise ValueError(
            f"Unsupported RC array shape {arr.shape} in {path.name}. "
            "Expected complex (H,W), (H,W,2), or (2,H,W)."
        )

    rc_np = np.nan_to_num(rc_np, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return torch.from_numpy(rc_np)  # (2,H,W) float32


def ceil_to_multiple(x: int, m: int) -> int:
    """Round x up to the nearest multiple of m."""
    if m <= 1:
        return x
    return ((x + m - 1) // m) * m


def compute_valid_mask(h: int, w: int, h_pad: int, w_pad: int, device: torch.device) -> Tensor:
    """Return (1, h_pad, w_pad) mask with ones in [0:h, 0:w] and zeros elsewhere."""
    vm = torch.zeros((1, h_pad, w_pad), dtype=torch.float32, device=device)
    vm[:, :h, :w] = 1.0
    return vm


def magnitude(rc: Tensor, eps: float = 1e-8) -> Tensor:
    """Return magnitude sqrt(I^2+Q^2+eps) from rc=(2,H,W)."""
    return torch.sqrt((rc ** 2).sum(dim=0) + eps)  # (H, W)


def load_yaml_config(path: str | Path) -> dict:
    """Load YAML config as dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping.")
    return cfg


def ensure_channel_first(rc: Union[Tensor, np.ndarray]) -> Tensor:
    """
    Ensure RC has shape (2, H, W). Accepts:
      - torch.Tensor or np.ndarray in (2,H,W), (H,W,2), or complex (H,W).
    Returns torch.float32 (2,H,W).
    """
    if isinstance(rc, np.ndarray):
        arr = rc
        if np.iscomplexobj(arr) and arr.ndim == 2:
            rc_np = np.stack([arr.real.astype(np.float32), arr.imag.astype(np.float32)], axis=0)
        elif arr.ndim == 3 and arr.shape[0] == 2:
            rc_np = arr.astype(np.float32)
        elif arr.ndim == 3 and arr.shape[-1] == 2:
            rc_np = arr.astype(np.float32).transpose(2, 0, 1)  # (H,W,2)->(2,H,W)
        else:
            raise ValueError(f"Unsupported RC array shape {arr.shape}. Expected (2,H,W), (H,W,2), or complex (H,W).")
        rc_np = np.nan_to_num(rc_np, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        return torch.from_numpy(rc_np)

    if not isinstance(rc, torch.Tensor):
        raise TypeError(f"rc must be Tensor or ndarray, got {type(rc)}")

    if rc.is_complex() and rc.ndim == 2:
        rc = torch.stack([rc.real, rc.imag], dim=0).to(torch.float32)
    elif rc.ndim == 3 and rc.shape[0] == 2:
        rc = rc.to(torch.float32)
    elif rc.ndim == 3 and rc.shape[-1] == 2:
        rc = rc.permute(2, 0, 1).contiguous().to(torch.float32)  # (H,W,2)->(2,H,W)
    else:
        raise ValueError(f"Unsupported RC tensor shape {tuple(rc.shape)}. Expected (2,H,W), (H,W,2), or complex (H,W).")

    rc = torch.nan_to_num(rc, nan=0.0, posinf=0.0, neginf=0.0)
    return rc


# ------------------------- Mask resolver ------------------- #
import re
from pathlib import Path
from typing import Optional

def default_mask_resolver(rc_path: Path) -> Optional[Path]:
    """
    Accept masks named with or without a trailing `_scaled`.
    """
    rc_path = Path(rc_path)
    stem = rc_path.stem

    # Find split dir
    split_dir: Optional[Path] = None
    for ancestor in [rc_path.parent] + list(rc_path.parents):
        if ancestor.name.lower() in {"train", "val", "test"}:
            split_dir = ancestor
            break
    if split_dir is None:
        return None

    mask_dir = split_dir / "mask"
    # Try exact stem first, then a version with `_scaled` stripped.
    candidates = [
        mask_dir / f"{stem}_mask.png",
        mask_dir / f"{re.sub(r'_scaled$', '', stem, flags=re.IGNORECASE)}_mask.png",
    ]

    for p in candidates:
        if p.is_file():
            return p
    return None



def load_mask_png(path: Path) -> Tensor:
    """
    Load a mask PNG as a (H, W) torch.int64 tensor.
    - If binary values are {0,255}, remap 255 -> 1 (keeps labels compact).
    - Keeps other integer labels as-is (multi-class).
    """
    arr = np.array(Image.open(path))
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D (H,W); got shape {arr.shape} at {path}")
    # Normalize binary {0,255} -> {0,1}
    uniques = np.unique(arr)
    if set(uniques.tolist()).issubset({0, 255}):
        arr = (arr == 255).astype(np.int64)
    else:
        # Ensure integer dtype
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"Mask must be integer-labeled; got dtype {arr.dtype} at {path}")
        arr = arr.astype(np.int64, copy=False)
    return torch.from_numpy(arr)  # (H,W) int64


# ------------------------- Dataset ------------------------ #
class RCDataset(Dataset):
    """
    Range-Compressed SAR dataset from .npy files.
    Returns dict with: "rc" (2,H,W) float32, "mask" (Optional[H,W] int64), "meta" (dict)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        use_masks: bool = True,
        mask_resolver: Optional[Callable[[Path], Optional[Path]]] = None,
        transform: Optional[Callable[[Sample], Sample]] = None,
        strict_missing_masks: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.use_masks = use_masks
        self.mask_resolver = mask_resolver or default_mask_resolver
        self.transform = transform
        self.strict_missing_masks = strict_missing_masks

        self.paths: List[Path] = sorted([Path(p) for p in glob.glob(os.path.join(str(self.data_dir), "*.npy"))])
        if not self.paths:
            raise FileNotFoundError(f"No .npy files found under {self.data_dir}")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Found %d RC files under %s", len(self.paths), self.data_dir)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Sample:
        path = self.paths[idx]

        # ---- RC load (2,H,W) float32 ----
        rc = tensor_from_npy(path)
        size_pre = shape_tuple(rc)

        # ---- Mask load (optional) ----
        mask: Optional[Tensor] = None
        if self.use_masks:
            mpath = self.mask_resolver(path)
            if mpath is None:
                msg = f"No mask found for {path}"
                if self.strict_missing_masks:
                    raise FileNotFoundError(msg)
                else:
                    LOGGER.warning(msg)
            else:
                mask = load_mask_png(mpath)
        rc_hwc = rc.permute(1, 2, 0).contiguous().float()  
        sample: Sample = {
            "rc": rc_hwc,
            "mask": mask,  # may be None if not found and strict_missing_masks=False
            "meta": {
                "path": str(path),
                "basename": path.stem,
            },
        }

        # ---- Transforms (RC-only; mask untouched) ----
        if self.transform is not None:
            sample = self.transform(sample)

        # ---- Standardize RC layout and record sizes ----
        sample["rc"] = ensure_channel_first(sample["rc"])
        size_post = shape_tuple(sample["rc"])
        H, W = sample["rc"].shape[1], sample["rc"].shape[2]

        # ---- Geometry sanity: mask must match RC after transforms ----
        if self.use_masks and sample["mask"] is not None:
            m = sample["mask"]
            if not isinstance(m, torch.Tensor):
                raise TypeError(f"Mask must be a torch.Tensor, got {type(m)}")
            if m.ndim != 2:
                raise ValueError(f"Mask must be 2D (H,W); got {tuple(m.shape)}")
            if (m.shape[0] != H) or (m.shape[1] != W):
                raise ValueError(
                    f"Mask size {tuple(m.shape)} does not match RC size {(H, W)} for {path}. "
                    "If RC transforms change geometry, mirror them on the mask (nearest)."
                )
            # Ensure int64 dtype for loss/indexing consistency
            if m.dtype != torch.int64:
                sample["mask"] = m.to(torch.int64)

        sample["meta"].update(
            {
                "H": int(H),
                "W": int(W),
                "size_pre_transform": size_pre,    # e.g., (2, 638, 512)
                "size_post_transform": size_post,  # e.g., (2, 642, 512)
            }
        )

        LOGGER.debug(
            "Sample %-24s | pre=%-15s post=%-15s mask=%s",
            os.path.basename(path), str(size_pre), str(size_post),
            "None" if sample["mask"] is None else tuple(sample["mask"].shape),
        )

        return sample


# ------------------------- Collate ------------------------ #
@dataclass
class PaddedBatch:
    """Batched tensors with collate-time padding and valid_mask."""
    rc: Tensor                 # (B, 2, H_pad, W_pad)
    mask: Optional[Tensor]     # (B, H_pad, W_pad) or None
    valid_mask: Tensor         # (B, 1, H_pad, W_pad)
    meta: List[Dict[str, Any]]


def collate_pad_validmask(
    batch: List[Sample],
    enforce_stride_multiple: bool = True,
    stride_multiple: int = 32,
    bypass_padding: bool = False,
    mask_ignore_index: int = 255,
    device: torch.device | None = None,
) -> Union[PaddedBatch, List[Sample]]:
    """
    Collate: pad to batch maxima (optionally to nearest stride multiple); build valid_mask.
    If bypass_padding=True, return the unmodified list of samples.
    """
    if bypass_padding:
        for s in batch:
            rc = ensure_channel_first(s["rc"])
            LOGGER.info(
                "Bypass sample %-24s | rc=%s",
                os.path.basename(s["meta"].get("path", "")),
                str(shape_tuple(rc)),
            )
        return batch

    device = device or torch.device("cpu")

    # Compute batch maxima (H,W) after transforms
    sizes_hw: List[Tuple[int, int]] = []
    for s in batch:
        rc = ensure_channel_first(s["rc"])
        sizes_hw.append((rc.shape[1], rc.shape[2]))
    H_max = max(h for h, _ in sizes_hw)
    W_max = max(w for _, w in sizes_hw)

    # Decide padded size
    if enforce_stride_multiple and stride_multiple > 1:
        H_pad = ceil_to_multiple(H_max, stride_multiple)
        W_pad = ceil_to_multiple(W_max, stride_multiple)
        LOGGER.info(
            "Padding to (H_pad,W_pad)=(%d,%d) with stride_multiple=%d (from max=(%d,%d)) | %s",
            H_pad, W_pad, stride_multiple, H_max, W_max, check_stride_ready(H_pad, W_pad, stride_multiple),
        )
    else:
        H_pad, W_pad = H_max, W_max
        LOGGER.info("Padding to batch maxima (no stride enforcement): (H_pad,W_pad)=(%d,%d)", H_pad, W_pad)

    B = len(batch)
    rc_b = torch.zeros((B, 2, H_pad, W_pad), dtype=torch.float32, device=device)
    valid_b = torch.zeros((B, 1, H_pad, W_pad), dtype=torch.float32, device=device)

    any_mask_present = any(s["mask"] is not None for s in batch)
    mask_b: Optional[Tensor] = None
    if any_mask_present:
        mask_b = torch.full((B, H_pad, W_pad), fill_value=mask_ignore_index, dtype=torch.int64, device=device)

    metas: List[Dict[str, Any]] = []

    for i, s in enumerate(batch):
        rc = ensure_channel_first(s["rc"])
        H, W = rc.shape[1], rc.shape[2]
        rc_b[i, :, :H, :W] = rc.to(device)
        valid_b[i] = compute_valid_mask(H, W, H_pad, W_pad, device=device)

        if any_mask_present and s["mask"] is not None:
            m = s["mask"].to(device)
            if m.ndim != 2 or m.shape != (H, W):
                raise ValueError(f"Mask shape mismatch: got {tuple(m.shape)} expected {(H, W)}")
            mask_b[i, :H, :W] = m

        metas.append(s["meta"])

    LOGGER.info(
        "Batch: B=%d, C=%d, H=%d, W=%d | mask=%s | valid=%s",
        B, 2, H_pad, W_pad,
        "None" if mask_b is None else tuple(mask_b.shape),
        tuple(valid_b.shape),
    )

    return PaddedBatch(rc=rc_b, mask=mask_b, valid_mask=valid_b, meta=metas)


# ------------------------- Transforms --------------------- #
#----------Helpers ------------# 

def _resolve_norm_profile(cfg: Dict, split: str) -> Dict:
    """
    Resolve the active normalization profile for this split.
    Supports:
      norm.active: per_dataset|per_patch
      norm.use.{train,val,test,infer}: profile name
      norm.profiles.{per_dataset,per_patch}: settings
    """
    norm_cfg = cfg.get("norm", {})
    profiles = norm_cfg.get("profiles", {})
    use = norm_cfg.get("use", {})
    active = norm_cfg.get("active")

    profile_name = use.get(split) if use else active
    profile_name = profile_name or "per_patch"
    if profile_name not in profiles:
        raise ValueError(f"norm profile '{profile_name}' not found under norm.profiles")

    prof = dict(profiles[profile_name])  # copy
    prof.setdefault("method", "rms-shared")
    prof.setdefault("mode", "per_patch")   # 'per_patch' | 'per_dataset'
    prof.setdefault("eps", 1e-12)
    prof.setdefault("log_stats", False)
    prof["_name"] = profile_name
    return prof

def _load_rms_value_if_needed(norm_profile: Dict) -> Optional[float]:
    """
    For per-dataset mode, return the RMS scalar from either:
      - profile['rms_value'] (already injected, e.g., from calibration), or
      - JSON at profile['stats_path'] containing key 'rms'.
    Returns None for per-patch mode.
    """
    if norm_profile.get("mode") != "per_dataset":
        return None

    if "rms_value" in norm_profile:   
        rms = float(norm_profile["rms_value"])
        if not (rms > 0.0):
            raise ValueError(f"Invalid rms_value in config: {rms}")
        return rms

    stats_path = norm_profile.get("stats_path")
    if not stats_path:
        raise ValueError("per_dataset mode requires 'rms_value' or 'stats_path' in norm profile.")

    p = Path(stats_path)
    if not p.exists():
        raise FileNotFoundError(f"norm.stats_path not found: {stats_path}")

    with p.open("r", encoding="utf-8") as f:
        js = json.load(f)
    rms = float(js.get("rms", float("nan")))
    if not (rms > 0.0):
        raise ValueError(f"Invalid 'rms' in {stats_path}: {rms}")
    LOGGER.info("Loaded per-dataset RMS from %s: %.8e", stats_path, rms)
    return rms

#----------Main transforms ------------# 
def build_transforms_from_cfg(cfg: dict, split: str = "train") -> Compose:
    """
    Build RC-only transforms from YAML config.
    Rules:
      - Augmentations applied ONLY for split=='train'.
      - Normalizer is always appended after augmentations.
      - Supports per-patch and per-dataset RMS-shared via norm.profiles + norm.active/use.
    """
    tfms = []

    # --- Augmentations (RC-only; train only) ---
    augs = cfg.get("augs", {})
    enable_augs = (split == "train")

    ap = augs.get("amp_phase", {})
    if enable_augs and ap.get("enable", True):
        tfms.append(
            RCAmplitudePhasePerturb(
                amp_scale_range=tuple(ap.get("amp_scale_range", [0.85, 1.15])),
                max_phase_shift=float(ap.get("max_phase_shift", 0.1)),
                per_pixel=bool(ap.get("per_pixel", True)),
            )
        )

    dr = augs.get("dropout", {})
    if enable_augs and dr.get("enable", True):
        tfms.append(
            RCNarrowbandSpectralDropout(
                bands_range=dr.get("bands_range", []),
                bands_azimuth=dr.get("bands_azimuth", []),
                soft=bool(dr.get("soft", True)),
                edge_taper=float(dr.get("edge_taper", 0.1)),
            )
        )

    tr = augs.get("trim", {})
    if enable_augs and tr.get("enable", True):
        tfms.append(
            RCBandwidthTrim(
                trim_frac_range=float(tr.get("trim_frac_range", 0.96)),
                trim_frac_azimuth=float(tr.get("trim_frac_azimuth", 0.99)),
                window=str(tr.get("window", "hann")),
            )
        )

    df = augs.get("defocus", {})
    if enable_augs and df.get("enable", True):
        tfms.append(RCAzimuthDefocus(kappa=float(df.get("kappa", 6.0e-4))))

    ss = augs.get("subsample", {})
    if enable_augs and ss.get("enable", True):
        tfms.append(
            RCSubsampleAndInterpolate(
                az_factor=int(ss.get("az_factor", 2)),
                rg_factor=int(ss.get("rg_factor", 1)),
                down_mode=str(ss.get("down_mode", "avg")),
                up_mode=str(ss.get("up_mode", "bilinear")),
            )
        )

    # --- Normalization (always last) ---
    norm_profile = _resolve_norm_profile(cfg, split)
    method_cfg = norm_profile.get("method", "rms-shared")
    if method_cfg not in ("rms-shared", "rms_shared", "rms_shared_per_column"):
        raise NotImplementedError(f"Unsupported norm.method: {method_cfg}")

    method_arg = "rms_shared" if method_cfg in ("rms-shared", "rms_shared") else method_cfg
    eps = float(norm_profile.get("eps", 1e-12))
    log_stats = bool(norm_profile.get("log_stats", False))
    mode = norm_profile.get("mode", "per_patch")

    rms_value = _load_rms_value_if_needed(norm_profile)  # None for per_patch

    tfms.append(
        NormalizeRCTransform(
            method=method_arg,          # "rms_shared" or "rms_shared_per_column"
            eps=eps,
            log_stats=log_stats,
            mode=mode,                  # "per_patch" | "per_dataset"
            rms_value=rms_value,        # None unless per_dataset
        )
    )

    LOGGER.info(
        "Transforms(split=%s): augs=%s | norm_profile=%s | method=%s | mode=%s | eps=%.2e%s",
        split,
        "on" if enable_augs else "off",
        norm_profile.get("_name"),
        method_arg,
        mode,
        eps,
        f" | rms_value={rms_value:.6e}" if rms_value is not None else "",
    )

    return Compose(tfms)

# ------------------------- Main --------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RC SAR DataLoader.")
    # Data & loader
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing RC .npy tiles (a single split folder).")
    parser.add_argument("--config", type=str, required=True, help="Path to aug.yaml")
    parser.add_argument("--use_masks", type=lambda s: s.lower() == "true", default=True,
                        help="Attempt to load masks via default_mask_resolver.")
    parser.add_argument("--strict_missing_masks", type=lambda s: s.lower() == "true", default=True,
                        help="If True, raise error when a mask is missing; else keep sample with mask=None.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--bypass_padding", type=lambda s: s.lower() == "true", default=False,
                        help="If true, collate returns list of unpadded samples.")
    parser.add_argument("--enforce_stride_multiple", type=lambda s: s.lower() == "true", default=True,
                        help="Pad up to nearest stride multiple for U-Net compatibility.")
    parser.add_argument("--stride_multiple", type=int, default=32, help="UNet total downsampling multiple (e.g., 16/32).")
    parser.add_argument("--mask_ignore_index", type=int, default=255,
                        help="Ignore index used in padded mask regions.")
    # CLI behavior
    parser.add_argument("--max_batches", type=int, default=1, help="Max number of batches to iterate.")
    parser.add_argument("--verbosity", type=int, default=1, help="0=warn, 1=info, 2=debug")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Preferred device; falls back to CPU if CUDA unavailable.")
    return parser.parse_args()


def main() -> None:

    args = parse_args()

    # Logging level
    level = logging.INFO if args.verbosity == 1 else logging.DEBUG if args.verbosity >= 2 else logging.WARN
    LOGGER.setLevel(level)
    # 1) Load YAML
    cfg = load_yaml_config(args.config)

    # 2) Device
    requested = str(cfg.get("device", args.device)).lower()
    device = torch.device("cuda" if (requested == "cuda" and torch.cuda.is_available()) else "cpu")
    LOGGER.info("Selected device: %s (requested=%s, cuda_available=%s)", device, requested, torch.cuda.is_available())

    # 3) Transforms (RC-only)
    transforms = build_transforms_from_cfg(cfg, split="train")

    # 4) Dataset
    ds = RCDataset(
        data_dir=args.data_dir,
        use_masks=bool(args.use_masks),
        mask_resolver=default_mask_resolver,
        transform=transforms,  # RC normalization/augs only
        strict_missing_masks=bool(args.strict_missing_masks),
    )

    # 5) DataLoader
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=lambda batch: collate_pad_validmask(
            batch,
            enforce_stride_multiple=args.enforce_stride_multiple,
            stride_multiple=args.stride_multiple,
            bypass_padding=args.bypass_padding,
            mask_ignore_index=args.mask_ignore_index,
            device=device,
        ),
    )

    # 6) Iterate a few batches
    batches_iterated = 0
    for batch in loader:
        if isinstance(batch, list):
            LOGGER.info("Bypass mode (no padding): %d samples", len(batch))
            for i, s in enumerate(batch):
                rc = ensure_channel_first(s["rc"])
                LOGGER.info(" sample[%d] rc=%s (H=%d, W=%d) mask=%s", i, tuple(rc.shape), rc.shape[1], rc.shape[2],
                            "None" if s["mask"] is None else tuple(s["mask"].shape))
        else:
            B, C, H, W = batch.rc.shape
            LOGGER.info(
                "Batch: B=%d, C=%d, H=%d, W=%d | mask=%s | valid=%s",
                B, C, H, W,
                "None" if batch.mask is None else tuple(batch.mask.shape),
                tuple(batch.valid_mask.shape),
            )

        batches_iterated += 1
        if batches_iterated >= args.max_batches:
            break

LOGGER.info("Done.")


if __name__ == "__main__":
    main()








