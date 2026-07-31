"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: rc_dataloader.py

Description:
RFI Range-Compressed (RC) SAR DataLoader with 4-channel inputs.

- Loads RC SAR .npy VV/VH pairs as (4, H, W) float32 [VV_I, VV_Q, VH_I, VH_Q]
- Optional masks (H, W) int labels; padding uses ignore_index (default 255)
- Collate pads to batch maxima (optionally to nearest stride multiple)
- Builds per-sample valid_mask for loss/metric masking
- Mask geometry can differ from RC geometry; loss should use valid_mask & ignore_index

Assumptions
-----------
RC files look like (examples):

    s1a-iw-raw-s-vh-20200115t170902-20200115t170934-030814-0388f3-IW1-RFI_4_scaled.npy
    s1a-iw-raw-s-vv-20200115t170902-20200115t170934-030814-0388f3-IW1-RFI_4_scaled.npy

We pair files that differ only in the '-vv-' / '-vh-' token.

Masks are named like:

    DB_OPENSAR_RFI_295_MASK.png
    DB_OPENSAR_RFI_892_MASK.png

We extract the numeric id after 'RFI_' from the RC filename and map:

    ...RFI_892_scaled.npy  ->  DB_OPENSAR_RFI_892_MASK.png


History:
    - 2025-12-04:
        First version is ready.
    - 2025-12-05:
        The script is finalized and used for model training.    
    
---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-12-04

© Copyright INDRA DEIMOS, 2025. All rights reserved.
---------------------------------------------------------------------
"""

from __future__ import annotations
import glob
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import re
import numpy as np
import torch
import yaml
from PIL import Image
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from pipeline.RFI_usecase.utils.augmentation import (
    RCAmplitudePhasePerturb,
    RCNarrowbandSpectralDropout,
    RCBandwidthTrim,
    RCAzimuthDefocus,
    RCSubsampleAndInterpolate,
)
from pipeline.RFI_usecase.utils.normalization import NormalizeZScoreRC4PerDataset
from pipeline.RFI_usecase.utils.logging_setup import init_logging


# ------------------------- Logging ------------------------- #
init_logging(level=logging.WARNING)  
LOGGER = logging.getLogger(__name__)
logging.getLogger("pipeline.RFI_usecase.utils.rc_dataloader").setLevel(logging.WARNING)

# ------------------------- Types --------------------------- #
Sample = Dict[str, Any]  # {"rc": Tensor(C,H,W or H,W,C), "mask": Optional[Tensor(H,W)], "meta": dict}


# ------------------------- Utils --------------------------- #
def _resolve_norm_profile(cfg: Dict, split: str) -> Dict:
    """
    Resolve the active normalization profile for this split.
    Expects something like:

    norm:
      active: rfi_zscore
      profiles:
        rfi_zscore:
          method: zscore4
          mode: per_dataset
          eps: 1e-6
          stats_path: /.../rfi_zscore_stats.json
          log_stats: false
    """
    norm_cfg = cfg.get("norm", {})
    profiles = norm_cfg.get("profiles", {})
    use = norm_cfg.get("use", {})
    active = norm_cfg.get("active")

    profile_name = use.get(split) if use else active
    if not profile_name:
        raise ValueError("norm.active or norm.use.<split> must be set for z-score normalization.")

    if profile_name not in profiles:
        raise ValueError(f"norm profile '{profile_name}' not found under norm.profiles")

    prof = dict(profiles[profile_name])  # copy
    prof.setdefault("method", "zscore4")
    prof.setdefault("mode", "per_dataset")
    prof.setdefault("eps", 1e-6)
    prof.setdefault("log_stats", False)
    prof["_name"] = profile_name
    return prof


def build_transforms_from_cfg(cfg: dict, split: str = "train") -> Compose:
    """
    Build RC-only transforms from YAML config.

    - Augmentations: configurable under augs, applied only for split == 'train'.
    - Normalization: ONLY 4-channel per-dataset z-score via NormalizeZScoreRC4PerDataset.
    """
    tfms = []

    # ----------------- Augmentations (train only) -----------------
    augs = cfg.get("augs", {})
    enable_augs = (split == "train")

    ap = augs.get("amp_phase", {})
    if enable_augs and ap.get("enable", False):
        tfms.append(
            RCAmplitudePhasePerturb(
                amp_scale_range=tuple(ap.get("amp_scale_range", [0.98, 1.02])),
                max_phase_shift=float(ap.get("max_phase_shift", 0.0)),
                per_pixel=bool(ap.get("per_pixel", False)),
            )
        )

    dr = augs.get("dropout", {})
    if enable_augs and dr.get("enable", False):
        tfms.append(
            RCNarrowbandSpectralDropout(
                bands_range=dr.get("bands_range", []),
                bands_azimuth=dr.get("bands_azimuth", []),
                soft=bool(dr.get("soft", True)),
                edge_taper=float(dr.get("edge_taper", 0.1)),
            )
        )

    tr = augs.get("trim", {})
    if enable_augs and tr.get("enable", False):
        tfms.append(
            RCBandwidthTrim(
                trim_frac_range=float(tr.get("trim_frac_range", 0.96)),
                trim_frac_azimuth=float(tr.get("trim_frac_azimuth", 0.99)),
                window=str(tr.get("window", "hann")),
            )
        )

    df = augs.get("defocus", {})
    if enable_augs and df.get("enable", False):
        tfms.append(RCAzimuthDefocus(kappa=float(df.get("kappa", 6.0e-4))))

    ss = augs.get("subsample", {})
    if enable_augs and ss.get("enable", False):
        tfms.append(
            RCSubsampleAndInterpolate(
                az_factor=int(ss.get("az_factor", 2)),
                rg_factor=int(ss.get("rg_factor", 1)),
                down_mode=str(ss.get("down_mode", "avg")),
                up_mode=str(ss.get("up_mode", "bilinear")),
            )
        )

    # ----------------- Z-score normalization (always last) ------------------
    norm_profile = _resolve_norm_profile(cfg, split)
    method_cfg = str(norm_profile.get("method", "zscore4")).lower()

    if method_cfg != "zscore4":
        raise NotImplementedError(
            f"Only method='zscore4' is supported now for RFI. Got method={method_cfg!r} "
            f"in norm profile '{norm_profile.get('_name')}'."
        )

    stats_path = norm_profile.get("stats_path")
    if not stats_path:
        raise ValueError(
            f"norm profile '{norm_profile.get('_name')}' uses method='zscore4' "
            f"but has no 'stats_path' defined."
        )

    eps       = float(norm_profile.get("eps", 1e-6))
    log_stats = bool(norm_profile.get("log_stats", False))
    mode      = norm_profile.get("mode", "per_dataset")

    # ---- load mean/std from JSON ----
    with open(stats_path, "r") as f:
        stats = json.load(f)

    # Adjust these keys to match your JSON: e.g. "mean4", "std4", etc.
    mean = stats.get("mean")  # must be length-4
    std  = stats.get("std")   # must be length-4

    if mean is None or std is None:
        raise KeyError(
            f"Expected 'mean' and 'std' in {stats_path}, got keys: {list(stats.keys())}"
        )

    tfms.append(
        NormalizeZScoreRC4PerDataset(
            mean=mean,
            std=std,
            eps=eps,
            log_stats=log_stats,
        )
    )

    LOGGER.info(
        "Transforms(split=%s): augs=%s | norm_profile=%s | method=zscore4 | mode=%s "
        "| eps=%.2e | stats_path=%s",
        split,
        "on" if (enable_augs and any(b.get('enable', False) for b in augs.values())) else "off",
        norm_profile.get("_name"),
        mode,
        eps,
        stats_path,
    )

    return Compose(tfms)



def shape_tuple(x: torch.Tensor | np.ndarray) -> tuple[int, ...]:
    """Return a plain Python shape tuple of ints."""
    return tuple(int(d) for d in x.shape)


def check_stride_ready(h: int, w: int, stride: int) -> str:
    """Human-readable remainder check for stride alignment."""
    if stride <= 1:
        return "no stride enforcement"
    return f"H%{stride}={(h % stride)} W%{stride}={(w % stride)}"


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


def load_yaml_config(path: str | Path) -> dict:
    """Load YAML config as dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping.")
    return cfg


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


def ensure_channel_first(rc: Union[Tensor, np.ndarray]) -> Tensor:
    """
    Ensure RC has shape (C, H, W). Accepts:
      - torch.Tensor or np.ndarray in (C,H,W), (H,W,C), or complex (H,W).
    Supports C in {2,4} for 2-channel or 4-channel RC.
    Returns torch.float32 (C,H,W).
    """
    if isinstance(rc, np.ndarray):
        arr = rc

        # complex (H,W)
        if np.iscomplexobj(arr) and arr.ndim == 2:
            rc_np = np.stack(
                [arr.real.astype(np.float32), arr.imag.astype(np.float32)],
                axis=0,
            )  # (2,H,W)

        elif arr.ndim == 3 and arr.shape[0] in (2, 4):
            # already channel-first (C,H,W)
            rc_np = arr.astype(np.float32)

        elif arr.ndim == 3 and arr.shape[-1] in (2, 4):
            # channel-last (H,W,C) -> (C,H,W)
            rc_np = arr.astype(np.float32).transpose(2, 0, 1)

        else:
            raise ValueError(
                f"Unsupported RC array shape {arr.shape}. "
                "Expected complex (H,W), (C,H,W) with C in {2,4}, or (H,W,C) with C in {2,4}."
            )

        rc_np = np.nan_to_num(rc_np, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        return torch.from_numpy(rc_np)

    # torch.Tensor branch
    if not isinstance(rc, torch.Tensor):
        raise TypeError(f"rc must be Tensor or ndarray, got {type(rc)}")

    if rc.is_complex() and rc.ndim == 2:
        rc = torch.stack([rc.real, rc.imag], dim=0).to(torch.float32)  # (2,H,W)

    elif rc.ndim == 3 and rc.shape[0] in (2, 4):
        # already (C,H,W)
        rc = rc.to(torch.float32)

    elif rc.ndim == 3 and rc.shape[-1] in (2, 4):
        # (H,W,C) -> (C,H,W)
        rc = rc.permute(2, 0, 1).contiguous().to(torch.float32)

    else:
        raise ValueError(
            f"Unsupported RC tensor shape {tuple(rc.shape)}. "
            "Expected complex (H,W), (C,H,W) with C in {2,4}, or (H,W,C) with C in {2,4}."
        )

    rc = torch.nan_to_num(rc, nan=0.0, posinf=0.0, neginf=0.0)
    return rc


# ------------------------- Mask resolver + loader ------------------- #
def rfi_mask_resolver(rc_path: Path) -> Optional[Path]:
    """
    RFI-specific mask resolver.

    RC filename example:
        s1a-iw-raw-s-vv-20210625t020713-20210625t020746-038490-048ac9-IW2-RFI_892_scaled.npy

    Mask filename example:
        DB_OPENSAR_RFI_892_MASK.png

    Logic:
      1) Find the split directory (train/val/test).
      2) Look for 'RFI_<id>' in the RC stem.
      3) Build mask filename: 'DB_OPENSAR_RFI_<id>_MASK.png' under <split>/mask.
    """
    rc_path = Path(rc_path)
    stem = rc_path.stem  # e.g. 's1a-iw-raw-s-vv-...-IW2-RFI_892_scaled'

    # Find split dir (train/val/test)
    split_dir: Optional[Path] = None
    for ancestor in [rc_path.parent] + list(rc_path.parents):
        if ancestor.name.lower() in {"train", "val", "test2"}:
            split_dir = ancestor
            break
    if split_dir is None:
        return None

    mask_dir = split_dir / "mask"

    # Extract numeric id after 'RFI_'
    m = re.search(r"RFI_(\d+)", stem, flags=re.IGNORECASE)
    if not m:
        LOGGER.warning("Could not extract RFI id from RC name: %s", rc_path.name)
        return None

    rfi_id = m.group(1)  # e.g. '892'

    candidates = [
        mask_dir / f"DB_OPENSAR_RFI_{rfi_id}_MASK.png",
        mask_dir / f"db_opensar_rfi_{rfi_id}_mask.png",  # in case of lowercase variant
    ]

    for p in candidates:
        if p.is_file():
            return p

    LOGGER.warning(
        "RFI mask not found for RC %s (tried: %s)",
        rc_path.name,
        ", ".join(str(c) for c in candidates),
    )
    return None


def load_mask_png(path: Path) -> Tensor:
    """
    Load a mask PNG as a (H, W) torch.int64 tensor.
    - If binary values are {0,255}, remap 255 -> 1.
    """
    arr = np.array(Image.open(path))
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D (H,W); got shape {arr.shape} at {path}")
    # Normalize binary {0,255} -> {0,1}
    uniques = np.unique(arr)
    if set(uniques.tolist()).issubset({0, 255}):
        arr = (arr == 255).astype(np.int64)
    else:
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"Mask must be integer-labeled; got dtype {arr.dtype} at {path}")
        arr = arr.astype(np.int64, copy=False)
    return torch.from_numpy(arr)  # (H,W) int64


# ------------------------- RFI VV/VH pairing ------------------------ #
def _build_vv_vh_pairs(data_dir: Path) -> List[Dict[str, Path]]:
    """
    Scan data_dir for *.npy and build VV/VH pairs based on filename pattern
    like:

        s1a-iw-raw-s-vh-20200115t170902-20200115t170934-030814-0388f3-IW1-RFI_4_scaled.npy
        s1a-iw-raw-s-vv-20200115t170902-20200115t170934-030814-0388f3-IW1-RFI_4_scaled.npy

    We capture:
        prefix: 's1a-iw-raw-s-'
        pol   : 'vv' or 'vh'
        suffix: '-20200115t170902-...-IW1-RFI_4_scaled.npy'

    and define base_id = prefix + suffix.

    So vv/vh files with the same prefix+suffix become one pair.
    """
    all_paths = [Path(p) for p in glob.glob(os.path.join(str(data_dir), "*.npy"))]
    if not all_paths:
        raise FileNotFoundError(f"No .npy files found under {data_dir}")

    # base_id -> {"vv": Path, "vh": Path}
    groups: Dict[str, Dict[str, Path]] = {}

    # prefix - (vv|vh) - suffix
    pattern = re.compile(r"^(?P<prefix>.+-)(?P<pol>vv|vh)(?P<suffix>-.+\.npy)$", re.IGNORECASE)

    for p in all_paths:
        m = pattern.match(p.name)
        if not m:
            LOGGER.warning("Skipping file without expected '-vv-'/'-vh-' pattern: %s", p.name)
            continue
        prefix = m.group("prefix")
        pol = m.group("pol").lower()        # 'vv' or 'vh'
        suffix = m.group("suffix")
        base_id = prefix + suffix           # everything except the pol token

        if base_id not in groups:
            groups[base_id] = {}
        if pol in groups[base_id]:
            LOGGER.warning("Duplicate %s file for base '%s': %s", pol, base_id, p)
        groups[base_id][pol] = p

    paired: List[Dict[str, Path]] = []
    for base_id, d in groups.items():
        if "vv" in d and "vh" in d:
            paired.append({"base": base_id, "vv_path": d["vv"], "vh_path": d["vh"]})
        else:
            LOGGER.warning(
                "Base '%s' is missing vv or vh file (have: %s). Skipping.",
                base_id, list(d.keys()),
            )

    if not paired:
        raise RuntimeError(
            f"No vv/vh pairs found under {data_dir}. "
            "Check your filenames match the pattern '...-vv-...npy' and '...-vh-...npy'."
        )

    LOGGER.info("Found %d VV/VH pairs under %s", len(paired), data_dir)
    return paired


# ------------------------- RFI Dataset ------------------------ #
class RFI4ChannelDataset(Dataset):

    """
    Dataset for 4-channel RFI segmentation from paired VV/VH RC SAR data.

    Each sample contains:
        rc:
            torch.float32 tensor with shape (4, H, W), channel order:
            [VV_I, VV_Q, VH_I, VH_Q].

        mask:
            Optional torch.int64 tensor with shape (H, W).

        meta:
            Dictionary containing file paths, base id, and spatial metadata.

    If a mask is available and its geometry differs from the RC geometry, the RC
    tensor is bilinearly resized to match the mask before transforms are applied.

    """
    def __init__(
        self,
        data_dir: Union[str, Path],
        use_masks: bool = True,
        mask_resolver: Optional[Callable[[Path], Optional[Path]]] = None,
        transform: Optional[Callable[[Dict,], Dict]] = None,
        strict_missing_masks: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.use_masks = use_masks
        # RFI-specific resolver by default
        self.mask_resolver = mask_resolver or rfi_mask_resolver
        self.transform = transform
        self.strict_missing_masks = strict_missing_masks

        # Build VV/VH pairs
        self.groups: List[Dict[str, Path]] = _build_vv_vh_pairs(self.data_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("RFI4ChannelDataset initialised with %d samples", len(self.groups))

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Dict:
        g = self.groups[idx]
        vv_path: Path = g["vv_path"]
        vh_path: Path = g["vh_path"]
        base: str = g["base"]

        # ---- 1) Load RC VV / VH: each (2, H_rc, W_rc) ----
        rc_vv = tensor_from_npy(vv_path)  # (2, H_rc, W_rc)
        rc_vh = tensor_from_npy(vh_path)  # (2, H_rc, W_rc)

        if rc_vv.shape != rc_vh.shape:
            raise ValueError(
                f"VV and VH shapes differ for base '{base}': "
                f"vv={tuple(rc_vv.shape)}, vh={tuple(rc_vh.shape)}"
            )

        # Stack into 4 channels: [VV_I, VV_Q, VH_I, VH_Q] -> (4, H_rc, W_rc)
        rc4 = torch.cat([rc_vv, rc_vh], dim=0)
        size_pre = shape_tuple(rc4)

        # ---- 2) Load mask (optional) ----
        mask: Optional[Tensor] = None
        if self.use_masks:
            mpath = self.mask_resolver(vv_path)
            if mpath is None:
                msg = f"No mask found for {vv_path}"
                if self.strict_missing_masks:
                    raise FileNotFoundError(msg)
                else:
                    LOGGER.warning(msg)
            else:
                mask = load_mask_png(mpath)  

                if not isinstance(mask, torch.Tensor):
                    mask = torch.as_tensor(mask)
                if mask.ndim != 2:
                    raise ValueError(f"Mask must be 2D (H,W); got {tuple(mask.shape)}")

                if mask.dtype != torch.int64:
                    mask = mask.to(torch.int64)

        # ---- 3) Align RC to mask geometry if needed ----
        if self.use_masks and mask is not None:
            H_m, W_m = mask.shape
            _, H_rc, W_rc = rc4.shape

            if (H_rc != H_m) or (W_rc != W_m):
                # rc4: (4, H_rc, W_rc) -> (1, 4, H_rc, W_rc) for interpolate
                rc4 = rc4.unsqueeze(0)
                rc4 = F.interpolate(
                    rc4,
                    size=(H_m, W_m),
                    mode="bilinear",
                    align_corners=False,
                )
                rc4 = rc4.squeeze(0)  # back to (4, H_m, W_m)

        # ---- 4) Convert to HWC for transforms (if any) ----
        rc_hwc = rc4.permute(1, 2, 0).contiguous().float()  # (H, W, 4)

        sample: Dict = {
            "rc": rc_hwc,   # (H, W, 4)
            "mask": mask,   # (H, W) or None
            "meta": {
                "base": base,
                "vv_path": str(vv_path),
                "vh_path": str(vh_path),
            },
        }

        # ---- 5) Apply transforms (e.g. z-score norm) ----
        if self.transform is not None:
            sample = self.transform(sample)

        # ---- 6) Standardise RC layout to CHW for the rest of the pipeline ----
        sample["rc"] = ensure_channel_first(sample["rc"])  # -> (4, H_out, W_out) float32
        H_out, W_out = sample["rc"].shape[1], sample["rc"].shape[2]
        size_post = shape_tuple(sample["rc"])

        # Mask sanity checks (now RC and mask should share H×W)
        if self.use_masks and sample["mask"] is not None:
            m = sample["mask"]
            if not isinstance(m, torch.Tensor):
                raise TypeError(f"Mask must be a torch.Tensor, got {type(m)}")
            if m.ndim != 2:
                raise ValueError(f"Mask must be 2D (H,W); got {tuple(m.shape)}")
            if m.dtype != torch.int64:
                sample["mask"] = m.to(torch.int64)

            # Optional: assert geometry match
            H_m, W_m = sample["mask"].shape
            if (H_m != H_out) or (W_m != W_out):
                raise ValueError(
                    f"After alignment: rc shape (4,{H_out},{W_out}) "
                    f"but mask shape ({H_m},{W_m}) for base '{base}'"
                )

        sample["meta"].update(
            {
                "H": int(H_out),
                "W": int(W_out),
                "size_pre_transform": size_pre,
                "size_post_transform": size_post,
            }
        )

        LOGGER.debug(
            "RFI sample %-24s | pre=%-15s post=%-15s mask=%s",
            base,
            str(size_pre),
            str(size_post),
            "None" if sample["mask"] is None else tuple(sample["mask"].shape),
        )

        return sample



# ------------------------- Collate ------------------------ #
@dataclass
class PaddedBatch:
    """Batched tensors with collate-time padding and valid_mask."""
    rc: Tensor                 # (B, C, H_pad, W_pad)
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

    - RC is padded to (H_pad,W_pad) with zeros.
    - valid_mask is 1 where RC is real, 0 in padded regions.
    - Mask is padded with mask_ignore_index.
    - Mask size can differ from RC size; we only enforce 2D.
    """
    if bypass_padding:
        for s in batch:
            rc = ensure_channel_first(s["rc"])
            LOGGER.info(
                "Bypass sample %-24s | rc=%s",
                os.path.basename(s["meta"].get("vv_path", s["meta"].get("base", ""))),
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
    # infer channel count (C) from first sample
    C = ensure_channel_first(batch[0]["rc"]).shape[0]

    rc_b = torch.zeros((B, C, H_pad, W_pad), dtype=torch.float32, device=device)
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
            if m.ndim != 2:
                raise ValueError(f"Mask must be 2D, got {tuple(m.shape)}")
            Hm, Wm = m.shape
            mask_b[i, :Hm, :Wm] = m

        metas.append(s["meta"])

    LOGGER.info(
        "Batch: B=%d, C=%d, H=%d, W=%d | mask=%s | valid=%s",
        B, C, H_pad, W_pad,
        "None" if mask_b is None else tuple(mask_b.shape),
        tuple(valid_b.shape),
    )

    return PaddedBatch(rc=rc_b, mask=mask_b, valid_mask=valid_b, meta=metas)