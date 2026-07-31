#!/usr/bin/env python3
"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: rc_dataloader.py

Description:
Range-Compressed (RC) SAR DataLoader with Compose-friendly external transforms.

- Loads RC SAR .npy as (2, H, W) float32 [I,Q]
- Optional masks (can be None; future-proof for segmentation)
- Uses torchvision.transforms.Compose with the custom normalization/augs
- Collate pads to batch maxima and (optionally) to nearest stride multiple
- Builds per-sample valid_mask for loss/metric masking

Debug / Verification Additions (2025-10-02)
-------------------------------------------
- PrintRCStats: logs per-sample stats and optional patches
- SnapshotRC: stores an rc copy in sample['meta'] with a tag
- DeltaFromSnapshot: logs L1/L2/L∞ diffs between current rc and a snapshot
- Probes placed at:
    BEFORE           (raw)
    AFTER_NORM       (post normalization, pre augmentation)
    AFTER_ALL        (post augmentation)
- Deltas logged for:
    BEFORE → AFTER_NORM
    AFTER_NORM → AFTER_ALL
    BEFORE → AFTER_ALL

Usage
-----
python rc_dataloader.py \
  --data_dir /path/to/rc_npy \
  --config aug.yaml \
  --batch_size 1 \
  --num_workers 0 \
  --print_before_after true \
  --print_patch 6 \
  --save_arrays false \
  --save_dir ./rc_debug_out \
  --max_batches 3

History:
    - 2025-10-01: First version of dataloader script for range-compressed data.
    - 2025-10-02: Add BEFORE/AFTER_NORM/AFTER_ALL probes and deltas.

---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-09-30

© Copyright INDRA DEIMOS, 2025. All rights reserved.
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
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from logging_setup import init_logging
from data_preprocessing.normalization.sar_normalization import NormalizeRCTransform
from data_preprocessing.augmentation.sar_rc_augmentation import (
    RCAmplitudePhasePerturb,
    RCNarrowbandSpectralDropout,
    RCBandwidthTrim,
    RCAzimuthDefocus,
    RCSubsampleAndInterpolate,
)

# Configure logging once
init_logging(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

# ========================= Types & utils =========================
Sample = Dict[str, Any]  # {"rc": Tensor(2,H,W), "mask": Optional[Tensor(H,W)], "meta": dict}


def shape_tuple(x: torch.Tensor | np.ndarray) -> Tuple[int, ...]:
    """Return a plain Python shape tuple of ints."""
    return tuple(int(d) for d in x.shape)


def check_stride_ready(h: int, w: int, stride: int) -> str:
    """Return stride divisibility info for UNet-like models."""
    if stride <= 1:
        return "no stride enforcement"
    return f"H%{stride}={(h % stride)} W%{stride}={(w % stride)}"


def tensor_from_npy(path: Path) -> Tensor:
    """
    Load a .npy RC array and return torch.float32 tensor with shape (2, H, W).

    Accepts:
        - complex (H, W)
        - channel-last (H, W, 2)
        - channel-first (2, H, W)
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
    """Load YAML config file."""
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


def _rc_stats(rc: Tensor) -> Dict[str, float]:
    """Compute robust stats for rc=(2,H,W)."""
    rc = ensure_channel_first(rc).cpu()
    i = rc[0]
    q = rc[1]
    mag = magnitude(rc)
    return {
        "i_min": float(i.min().item()),
        "i_max": float(i.max().item()),
        "i_mean": float(i.mean().item()),
        "i_std": float(i.std(unbiased=False).item()),
        "q_min": float(q.min().item()),
        "q_max": float(q.max().item()),
        "q_mean": float(q.mean().item()),
        "q_std": float(q.std(unbiased=False).item()),
        "m_min": float(mag.min().item()),
        "m_max": float(mag.max().item()),
        "m_mean": float(mag.mean().item()),
        "m_std": float(mag.std(unbiased=False).item()),
    }


def _patch_str(rc: Tensor, patch: int) -> str:
    """Format a small top-left patch from I and Q as a string."""
    rc = ensure_channel_first(rc).cpu()
    p = max(0, int(patch))
    if p == 0:
        return ""
    i = rc[0, :p, :p]
    q = rc[1, :p, :p]
    np.set_printoptions(precision=4, suppress=True)
    return f"\n  I[0:{p},0:{p}]=\n{i.numpy()}\n  Q[0:{p},0:{p}]=\n{q.numpy()}"


# ========================= Debug transforms =========================
class PrintRCStats:
    """
    A no-op transform that prints (and optionally saves) rc stats.

    Args:
        tag (str): Label printed in logs, e.g., 'BEFORE', 'AFTER_NORM', 'AFTER_ALL'.
        patch (int): If >0, also print a top-left patch×patch array for I and Q.
        save_arrays (bool): If True, saves rc array to save_dir / f"{basename}.{tag}.npy".
        save_dir (Optional[Path]): Output directory when save_arrays=True.
    """

    def __init__(
        self,
        tag: str,
        patch: int = 0,
        save_arrays: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.tag = str(tag).upper()
        self.patch = int(patch)
        self.save_arrays = bool(save_arrays)
        self.save_dir = Path(save_dir) if save_arrays and save_dir is not None else None
        if self.save_arrays and self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, sample: Sample) -> Sample:
        rc = ensure_channel_first(sample["rc"])
        meta = sample.get("meta", {})
        basename = meta.get("basename", meta.get("path", "unknown"))

        stats = _rc_stats(rc)
        LOGGER.info(
            "[%s] %-24s | rc=%s | "
            "I[min=%.4g max=%.4g mean=%.4g std=%.4g] | "
            "Q[min=%.4g max=%.4g mean=%.4g std=%.4g] | "
            "M[min=%.4g max=%.4g mean=%.4g std=%.4g]%s",
            self.tag,
            os.path.basename(str(basename)),
            str(shape_tuple(rc)),
            stats["i_min"], stats["i_max"], stats["i_mean"], stats["i_std"],
            stats["q_min"], stats["q_max"], stats["q_mean"], stats["q_std"],
            stats["m_min"], stats["m_max"], stats["m_mean"], stats["m_std"],
            _patch_str(rc, self.patch),
        )

        if self.save_arrays and self.save_dir is not None:
            out_path = self.save_dir / f"{Path(str(basename)).stem}.{self.tag}.npy"
            np.save(str(out_path), rc.cpu().numpy(), allow_pickle=False)
            LOGGER.info("[%s] saved rc -> %s", self.tag, out_path)

        # Return sample unchanged
        return sample


class SnapshotRC:
    """Store a detached copy of rc in sample['meta'] under the given tag."""

    def __init__(self, tag: str) -> None:
        self.tag = tag.upper()

    def __call__(self, sample: Sample) -> Sample:
        rc = ensure_channel_first(sample["rc"]).detach().cpu().clone()
        sample.setdefault("meta", {})[f"rc_snap_{self.tag}"] = rc
        return sample


class DeltaFromSnapshot:
    """
    Compare current rc to a stored snapshot and log L1/L2/L∞ deltas on I/Q/M.

    from_tag: snapshot key to compare from (e.g., 'BEFORE')
    to_tag:   label used for printing (e.g., 'AFTER_NORM')
    """

    def __init__(self, from_tag: str, to_tag: str) -> None:
        self.from_tag = from_tag.upper()
        self.to_tag = to_tag.upper()

    def __call__(self, sample: Sample) -> Sample:
        rc_now = ensure_channel_first(sample["rc"]).detach().cpu()
        rc_prev = sample["meta"].get(f"rc_snap_{self.from_tag}")
        name = os.path.basename(sample["meta"].get("path", "unknown"))
        if rc_prev is None:
            LOGGER.warning("No snapshot %s found for %s", self.from_tag, name)
            return sample

        di = (rc_now[0] - rc_prev[0]).abs()
        dq = (rc_now[1] - rc_prev[1]).abs()
        dm = (magnitude(rc_now) - magnitude(rc_prev)).abs()

        def stats(x: torch.Tensor) -> Tuple[float, float, float]:
            l1 = float(x.mean().item())
            l2 = float(x.pow(2).mean().sqrt().item())
            linf = float(x.max().item())
            return l1, l2, linf

        mi, li2, li8 = stats(di)
        mq, lq2, lq8 = stats(dq)
        mm, lm2, lm8 = stats(dm)

        LOGGER.info(
            "[DELTA %s→%s] %-24s | "
            "I[L1=%.4g L2=%.4g L∞=%.4g] | "
            "Q[L1=%.4g L2=%.4g L∞=%.4g] | "
            "M[L1=%.4g L2=%.4g L∞=%.4g]",
            self.from_tag, self.to_tag, name,
            mi, li2, li8, mq, lq2, lq8, mm, lm2, lm8
        )
        return sample


# ========================= Dataset =========================
class RCDataset(Dataset):
    """
    Range-Compressed SAR dataset from .npy files.
    Returns dict with: "rc" (2,H,W) float32, "mask" (Optional[H,W] int64), "meta" (dict)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        use_masks: bool = False,
        mask_resolver: Optional[Callable[[Path], Optional[Path]]] = None,
        transform: Optional[Callable[[Sample], Sample]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.use_masks = use_masks
        self.mask_resolver = mask_resolver
        self.transform = transform

        self.paths: List[Path] = sorted([Path(p) for p in glob.glob(os.path.join(str(self.data_dir), "*.npy"))])
        if not self.paths:
            raise FileNotFoundError(f"No .npy files found under {self.data_dir}")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Found %d RC files under %s", len(self.paths), self.data_dir)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Sample:
        path = self.paths[idx]

        # ---- load (standardizes to (2,H,W)) ----
        rc = tensor_from_npy(path)
        size_pre = shape_tuple(rc)

        mask: Optional[Tensor] = None
        meta: Dict[str, Any] = {
            "path": str(path),
            "basename": path.stem,
        }

        sample: Sample = {"rc": rc, "mask": mask, "meta": meta}

        # ---- transforms (may change size/layout) ----
        if self.transform is not None:
            sample = self.transform(sample)

        # Enforce layout and record post-transform size
        sample["rc"] = ensure_channel_first(sample["rc"])
        size_post = shape_tuple(sample["rc"])
        H, W = sample["rc"].shape[1], sample["rc"].shape[2]

        sample["meta"].update({
            "H": int(H),
            "W": int(W),
            "size_pre_transform": size_pre,    # e.g. (2, 638, 512)
            "size_post_transform": size_post,  # e.g. (2, 642, 512)
        })

        LOGGER.debug(
            "Sample %-24s | pre=%-15s post=%-15s",
            os.path.basename(path), str(size_pre), str(size_post)
        )

        return sample


# ========================= Collate with padding & valid masks =========================
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
                str(shape_tuple(rc))
            )
        return batch

    device = device or torch.device("cpu")

    # Per-sample logs before padding
    for s in batch:
        meta = s["meta"]
        LOGGER.info(
            "Sample %-24s | pre=%-12s post=%-12s",
            os.path.basename(meta.get("path", "")),
            str(meta.get("size_pre_transform", "?")),
            str(meta.get("size_post_transform", "?")),
        )

    # Compute batch maxima (H,W) after transforms (and enforce channel-first)
    sizes_hw: List[Tuple[int, int]] = []
    for s in batch:
        rc = ensure_channel_first(s["rc"])
        sizes_hw.append((rc.shape[1], rc.shape[2]))
    H_max = max(h for h, _ in sizes_hw)
    W_max = max(w for _, w in sizes_hw)

    # Decide padded size and log it
    if enforce_stride_multiple and stride_multiple > 1:
        H_pad = ceil_to_multiple(H_max, stride_multiple)
        W_pad = ceil_to_multiple(W_max, stride_multiple)
        LOGGER.info(
            "Padding to (H_pad,W_pad)=(%d,%d) with stride_multiple=%d (from max=(%d,%d)) | %s",
            H_pad, W_pad, stride_multiple, H_max, W_max, check_stride_ready(H_pad, W_pad, stride_multiple)
        )
    else:
        H_pad, W_pad = H_max, W_max
        LOGGER.info(
            "Padding to batch maxima (no stride enforcement): (H_pad,W_pad)=(%d,%d)",
            H_pad, W_pad
        )

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
                raise ValueError("Mask shape mismatch.")
            mask_b[i, :H, :W] = m

        metas.append(s["meta"])

    LOGGER.info("Batch padded rc shape = %s", str(shape_tuple(rc_b)))

    return PaddedBatch(rc=rc_b, mask=mask_b, valid_mask=valid_b, meta=metas)


# ========================= Build Compose (YAML + debug probes) =========================
def build_transforms_from_cfg(
    cfg: dict,
    *,
    print_before_after: bool = False,
    print_patch: int = 0,
    save_arrays: bool = False,
    save_dir: Optional[Union[str, Path]] = None,
) -> Compose:
    """
    Build a Compose pipeline from YAML + optional debug probes for pre/mid/post printing.
    """
    tfms: List[Callable[[Sample], Sample]] = []

    # BEFORE: raw
    if print_before_after:
        tfms.append(PrintRCStats("BEFORE", patch=print_patch, save_arrays=save_arrays, save_dir=save_dir))
        tfms.append(SnapshotRC("BEFORE"))

    # --- Normalization ---
    norm = cfg.get("norm", {})
    if norm.get("method", "rms-shared") in {"rms_shared", "rms-shared"}:
        tfms.append(NormalizeRCTransform(method="rms_shared"))
    # Add other normalization options here as needed

    # AFTER_NORM (pre-augmentation)
    if print_before_after:
        tfms.append(PrintRCStats("AFTER_NORM", patch=print_patch, save_arrays=save_arrays, save_dir=save_dir))
        tfms.append(SnapshotRC("AFTER_NORM"))
        tfms.append(DeltaFromSnapshot("BEFORE", "AFTER_NORM"))

    # --- Augmentation (enable flags control inclusion) ---
    augs = cfg.get("augs", {})

    ap = augs.get("amp_phase", {})
    if ap.get("enable", True):
        tfms.append(
            RCAmplitudePhasePerturb(
                amp_scale_range=tuple(ap.get("amp_scale_range", [0.85, 1.15])),
                max_phase_shift=float(ap.get("max_phase_shift", 0.1)),
                per_pixel=bool(ap.get("per_pixel", True)),
            )
        )

    dr = augs.get("dropout", {})
    if dr.get("enable", True):
        tfms.append(
            RCNarrowbandSpectralDropout(
                bands_range=dr.get("bands_range", []),
                bands_azimuth=dr.get("bands_azimuth", []),
                soft=bool(dr.get("soft", True)),
                edge_taper=float(dr.get("edge_taper", 0.1)),
            )
        )

    tr = augs.get("trim", {})
    if tr.get("enable", True):
        tfms.append(
            RCBandwidthTrim(
                trim_frac_range=float(tr.get("trim_frac_range", 0.96)),
                trim_frac_azimuth=float(tr.get("trim_frac_azimuth", 0.99)),
                window=str(tr.get("window", "hann")),
            )
        )

    df = augs.get("defocus", {})
    if df.get("enable", True):
        tfms.append(
            RCAzimuthDefocus(
                kappa=float(df.get("kappa", 6.0e-4)),
            )
        )

    ss = augs.get("subsample", {})
    if ss.get("enable", True):
        tfms.append(
            RCSubsampleAndInterpolate(
                az_factor=int(ss.get("az_factor", 2)),
                rg_factor=int(ss.get("rg_factor", 1)),
                down_mode=str(ss.get("down_mode", "avg")),
                up_mode=str(ss.get("up_mode", "bilinear")),
            )
        )

    # AFTER_ALL (post-augmentation)
    if print_before_after:
        tfms.append(PrintRCStats("AFTER_ALL", patch=print_patch, save_arrays=save_arrays, save_dir=save_dir))
        tfms.append(DeltaFromSnapshot("AFTER_NORM", "AFTER_ALL"))
        tfms.append(DeltaFromSnapshot("BEFORE", "AFTER_ALL"))

    return Compose(tfms)


# ========================= Mask resolver placeholder =========================
def default_mask_resolver(_: Path) -> Optional[Path]:
    """
    Map an RC file path to its mask path; return None if unknown.
    Replace with your real resolver when masks are available.
    """
    return None


# ========================= Main =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RC SAR DataLoader with debug probes.")

    # Data & loader
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with RC .npy files (each sample).")
    parser.add_argument("--config", type=str, required=True, help="Path to aug.yaml")
    parser.add_argument("--use_masks", type=lambda s: s.lower() == "true", default=False,
                        help="Attempt to load masks.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--bypass_padding", type=lambda s: s.lower() == "true", default=False,
                        help="If true, collate returns list of unpadded samples.")
    parser.add_argument("--enforce_stride_multiple", type=lambda s: s.lower() == "true", default=True,
                        help="Pad up to nearest stride multiple for U-Net compatibility.")
    parser.add_argument("--stride_multiple", type=int, default=32,
                        help="U-Net total downsampling multiple.")
    parser.add_argument("--mask_ignore_index", type=int, default=255,
                        help="Ignore index for padded/missing masks.")

    # Debug / printing
    parser.add_argument("--print_before_after", type=lambda s: s.lower() == "true", default=False,
                        help="If true, print stats and deltas at BEFORE/AFTER_NORM/AFTER_ALL.")
    parser.add_argument("--print_patch", type=int, default=0,
                        help="If >0, also print top-left patch×patch for I and Q.")
    parser.add_argument("--save_arrays", type=lambda s: s.lower() == "true", default=False,
                        help="If true, save pre/mid/post rc arrays to --save_dir as .npy.")
    parser.add_argument("--save_dir", type=str, default="./rc_debug_out",
                        help="Directory to save arrays when --save_arrays true.")

    # CLI behavior
    parser.add_argument("--max_batches", type=int, default=1, help="Max number of batches to iterate.")
    parser.add_argument("--verbosity", type=int, default=1, help="0=warn, 1=info, 2=debug")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Preferred device; if 'cuda' but unavailable, falls back to CPU.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Logging level
    level_map = {0: logging.WARN, 1: logging.INFO, 2: logging.DEBUG}
    LOGGER.setLevel(level_map.get(args.verbosity, logging.INFO))

    # 1) Load YAML
    cfg = load_yaml_config(args.config)

    # 2) Device selection
    config = {"device": str(cfg.get("device", args.device)).lower()}
    device = torch.device("cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    LOGGER.info("Selected device: %s (requested=%s, cuda_available=%s)",
                device, config["device"], torch.cuda.is_available())

    # 3) Build Compose from YAML + debug probes
    transforms = build_transforms_from_cfg(
        cfg,
        print_before_after=bool(args.print_before_after),
        print_patch=int(args.print_patch),
        save_arrays=bool(args.save_arrays),
        save_dir=args.save_dir,
    )

    # 4) Dataset
    ds = RCDataset(
        data_dir=args.data_dir,
        use_masks=False,            # flip to True when you have masks
        mask_resolver=None,         # plug a resolver when ready
        transform=transforms,       # ← the Compose from YAML (+ probes if enabled)
    )

    # 5) Loader
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

    # 6) Iterate
    batches_iterated = 0
    for batch in loader:
        if isinstance(batch, list):
            LOGGER.info("Bypass mode (no padding): %d samples", len(batch))
            for i, s in enumerate(batch):
                rc = ensure_channel_first(s["rc"])
                LOGGER.info(" sample[%d] rc=%s (H=%d, W=%d)", i, tuple(rc.shape), rc.shape[1], rc.shape[2])
        else:
            B, C, H, W = batch.rc.shape
            LOGGER.info(
                "Batch: B=%d, C=%d, H=%d, W=%d | H%%%d=%d W%%%d=%d | mask=%s | valid=%s",
                B, C, H, W,
                args.stride_multiple, H % args.stride_multiple,
                args.stride_multiple, W % args.stride_multiple,
                "None" if batch.mask is None else tuple(batch.mask.shape),
                tuple(batch.valid_mask.shape),
            )

        batches_iterated += 1
        if batches_iterated >= args.max_batches:
            break

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
