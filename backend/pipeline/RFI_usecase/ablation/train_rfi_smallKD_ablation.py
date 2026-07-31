"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: train_rfi_smallKD_ablation.py

Description:
- Train a SMALL UNet (student) for RC SAR flood segmentation.
- KD modes: logits | features | hybrid | none (config-selectable).
- Weights & Biases optional logging.

CLI
---
python train_rfi_smallKD_ablation.py --config /path/to/config.yaml  [--base-channels 6|8]  [--transforms /path/to/transforms.yaml] [--device cuda|cpu]

History:
  - 2026-03-05: 
        First working version.
  - 2026-03-20: 
        Final version is ready.

TODO: None.       

---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2026-03-05

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""
from __future__ import annotations
from pathlib import Path
import logging
import copy
import argparse
import sys
from types import SimpleNamespace
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from pipeline.RFI_usecase.utils.utilities import focal_loss_only
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler

# --- optional: wandb ---
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False

# Ensure repo root in sys.path when running directly
_THIS = Path(__file__).resolve()
PIPELINE_DIR = _THIS.parent
BACKEND_DIR = PIPELINE_DIR.parent
for p in (BACKEND_DIR, PIPELINE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pipeline.RFI_usecase.utils.logging_setup import init_logging
from pipeline.RFI_usecase.utils.rc_dataloader import (
    RFI4ChannelDataset, collate_pad_validmask, rfi_mask_resolver,
    build_transforms_from_cfg, load_yaml_config,
)
from pipeline.RFI_usecase.models.unet import UNet
try:
    from pipeline.RFI_usecase.models.unet_small import UNetSmall
    HAS_SMALL = True
except Exception:
    UNetSmall, HAS_SMALL = None, False

from pipeline.RFI_usecase.utils.utilities import (
    set_seed, ensure_dir, make_optimizer, make_scheduler,
    validate_one_epoch, estimate_pos_weight,
    save_weights, load_weights, evaluate_one_epoch_fixed_thr
)
from utils.plotting import save_metrics_csv, plot_training_curves

# KD utils
from pipeline.RFI_usecase.utils.utils_distill import (
    forward_teacher, kd_binary_bce, kd_logit_mse,
    compute_feature_kd
)

# --- logging ---
init_logging(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


#----------------------------mute noisy modules-----------------------------#
for name in ("pipeline.RFI_usecase.utils.rc_dataloader", "pipeline.RFI_usecase.utils.utilities", "PIL", "PIL.PngImagePlugin"):
    lg = logging.getLogger(name); lg.setLevel(logging.ERROR); lg.propagate = False
    for h in list(lg.handlers): h.setLevel(logging.ERROR)


# ----------------------------- CLI -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SMALL UNet with optional KD.")
    p.add_argument("--config", required=True, type=str,
                   help="Path to training config YAML.")
    p.add_argument("--transforms", default=None, type=str,
                   help="Optional path to transforms YAML (overrides config.transforms_cfg).")
    p.add_argument("--device", default=None, choices=["cuda","cpu"],
                   help="Optional device override (overrides config.device).")
    p.add_argument("--base-channels", default=None, type=int,
                   help="Optional student base_channels override. Overrides model.base_channels in config.")

    return p.parse_args()


# ------------------------- Data transformation utilities & build data loader ------------------------- #
def make_val_cfg_from_train_cfg(cfg: dict) -> dict:
    """
    Return a copy of the training transform config with all augmentations disabled.

    This is used to build a deterministic validation transform config from the
    training configuration by setting any block containing an "enable" flag to False.
    """
    val = copy.deepcopy(cfg)
    augs = val.get("augs", {})
    for _, block in augs.items():
        if isinstance(block, dict) and "enable" in block:
            block["enable"] = False 
    return val


def build_dataloader(data_dir: Path, cfg_yaml: dict, batch_size: int, num_workers: int, stride_multiple: int, ignore_index: int, shuffle: bool, split: str):
    """
    Build and return a DataLoader for the RFI dataset.

    Args:
        data_dir: Path to dataset directory.
        cfg_yaml: Transform configuration dictionary.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
        stride_multiple: Padding alignment used in collate function.
        ignore_index: Label value used to ignore padded/invalid pixels.
        shuffle: Whether to shuffle the dataset.
        split: Transform split name (e.g. "train" or "val").

    Returns:
        A torch.utils.data.DataLoader configured with the dataset and collate fn.
    """
    tfms = build_transforms_from_cfg(cfg_yaml, split=split)
    ds = RFI4ChannelDataset(data_dir=str(data_dir), use_masks=True, mask_resolver=rfi_mask_resolver, transform=tfms, strict_missing_masks=True)

    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
        collate_fn=lambda b: collate_pad_validmask(
            b, enforce_stride_multiple=True, stride_multiple=stride_multiple,
            bypass_padding=False, mask_ignore_index=ignore_index, device=torch.device("cpu")),
    )


# ----------------------------- Helper functions -----------------------------
def count_params(model: nn.Module, trainable_only: bool = True) -> float:
    """Return model parameter count in millions. If trainable_only is True, count only parameters that
    require gradients (i.e., trainable parameters)."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return sum(p.numel() for p in model.parameters()) / 1e6

def _unpack_batch_rc(batch, device):
    """Unpack a batch containing radar-channel data.

    Returns a tuple (imgs, masks, valid_mask) where each element is moved to the
    provided device. Handles several batch formats: object with attributes,
    dict, or tuple/list.
    """
    if hasattr(batch, "rc") and hasattr(batch, "valid_mask"):
        imgs = batch.rc.to(device, non_blocking=True)
        masks = None if batch.mask is None else batch.mask.to(device, non_blocking=True).float()
        valid_mask = batch.valid_mask.to(device, non_blocking=True)
        return imgs, masks, valid_mask
    if isinstance(batch, dict):
        imgs = batch["rc"].to(device, non_blocking=True)
        masks = None if batch.get("mask") is None else batch["mask"].to(device, non_blocking=True).float()
        vm = batch.get("valid_mask")
        if vm is not None: vm = vm.to(device, non_blocking=True)
        return imgs, masks, vm
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        imgs = batch[0].to(device, non_blocking=True)
        masks = None if batch[1] is None else batch[1].to(device, non_blocking=True).float()
        vm = batch[2].to(device, non_blocking=True) if len(batch) > 2 and batch[2] is not None else None
        return imgs, masks, vm
    raise TypeError(f"Unsupported batch type: {type(batch)}")

def _to_numpy_img(x: torch.Tensor) -> np.ndarray:
    """
    Convert a torch tensor in [0,1] range to a uint8 HxW or CxHxW numpy image.

    Supports single-channel tensors with shape (1,H,W) or (H,W) and tensors with channel
    first layout. Values are clamped to [0,1], scaled to [0,255] and rounded.
    """
    if x.ndim == 3 and x.shape[0] == 1: x = x[0]
    x = x.detach().float().cpu().clamp(0, 1).numpy()
    return (x * 255.0 + 0.5).astype(np.uint8)

def _make_magnitude(iq: torch.Tensor) -> torch.Tensor:
    """
    Compute a single-channel magnitude image from a 2-channel IQ tensor.

    The input iq is expected to have the real (I) and imaginary (Q)
    components as the first dimension (iq[0], iq[1]). The magnitude is
    computed as sqrt(I^2 + Q^2), contrast-stretched using the 1st and
    99th percentiles, and returned as a clamped tensor in [0,1]
    with a leading channel dimension.
    """
    i = iq[0].float(); q = iq[1].float()
    mag = torch.sqrt(i*i + q*q)
    lo, hi = torch.quantile(mag, torch.tensor([0.01, 0.99], device=mag.device))
    mag = (mag - lo) / (hi - lo + 1e-6)
    return mag.clamp(0,1).unsqueeze(0)

def _overlay_mask(base_gray01: np.ndarray, mask01: np.ndarray, color=(255,0,0), alpha=0.35) -> np.ndarray:
    """
    Overlay a binary mask onto a grayscale base image and return an RGB image.

    base_gray01: HxW uint8 or float grayscale image (0-1 or 0-255).
    mask01: HxW binary mask (0/1 or 0-255).
    color: RGB color tuple for the mask overlay.
    alpha: blending factor for the overlay.
    """
    # normalize mask to 0-255 if provided as 0-1
    if mask01.max() == 1: mask01 = mask01 * 255
    h, w = base_gray01.shape
    rgb = np.stack([base_gray01]*3, axis=-1).astype(np.float32)
    m = (mask01 > 127).astype(np.float32)[..., None]
    overlay = np.zeros_like(rgb, dtype=np.float32)
    overlay[...,0], overlay[...,1], overlay[...,2] = color
    out = rgb * (1 - alpha * m) + overlay * (alpha * m)
    return out.clip(0,255).astype(np.uint8)

@torch.no_grad()
def log_validation_images_wandb(model, val_loader, device, *, resize_to=None, max_images=4, prefix="val"):
    """
    Log validation images with GT and prediction overlays to Weights & Biases.
    """
    if not (_WANDB_AVAILABLE and wandb.run is not None): return
    model.eval(); images_logged, panels = 0, []
    for batch in val_loader:
        imgs, masks, _ = _unpack_batch_rc(batch, device)
        logits = model(imgs)
        if resize_to is not None:
            logits = F.interpolate(logits, size=(resize_to, resize_to), mode="bilinear", align_corners=False)
            masks_rs = None if masks is None else F.interpolate(masks.unsqueeze(1), size=(resize_to, resize_to), mode="nearest").squeeze(1)
        else:
            masks_rs = masks
        preds = (torch.sigmoid(logits) > 0.5).float()
        Ht, Wt = logits.shape[-2:]
        B = imgs.size(0)
        for b in range(B):
            if images_logged >= max_images: break
            mag = _make_magnitude(imgs[b])
            if mag.shape[-2:] != (Ht, Wt):
                mag = F.interpolate(mag.unsqueeze(0), size=(Ht, Wt), mode="bilinear", align_corners=False)[0]
            mag_np = _to_numpy_img(mag); pr_np = _to_numpy_img(preds[b])
            if masks_rs is None: continue
            gt_np = _to_numpy_img(masks_rs[b])
            gt_overlay = _overlay_mask(mag_np, gt_np, color=(0,255,0), alpha=0.35)
            pr_overlay = _overlay_mask(mag_np, pr_np,  color=(255,0,0), alpha=0.35)
            panels.append(wandb.Image(np.concatenate([np.stack([mag_np]*3, axis=-1), gt_overlay, pr_overlay], axis=1),
                                      caption=f"{prefix} | mag | GT overlay (green) | Pred overlay (red)"))
            images_logged += 1
        if images_logged >= max_images: break
    if panels: wandb.log({f"{prefix}/panels": panels})


# ----------------------------- Compute metric function -----------------------------

@torch.no_grad()
def compute_metrics(
    logits: Tensor,                  # [B,1,H,W] (raw logits)
    mask: Tensor,                    # [B,H,W] with {0,1,255}
    valid: Tensor,                   # [B,H,W] or [B,1,H,W]
    thr: Optional[float] = None,     # if None -> sweep to pick best (by select_by)
    thr_grid: Optional[Sequence[float]] = None,
    select_by: str = "dice",
    ignore_index: int = 255,
    return_counts: bool = False,
) -> Dict[str, float]:
    """
    Computes global metrics on VALID, NON-IGNORED pixels only (GPU).

    All metrics (Dice, IoU, Precision, Recall, Accuracy) are computed from
    globally pooled TP/FP/FN/TN across all pixels in logits/mask/valid.

    Returns:
        - dice, iou, precision, recall, acc: global (micro) metrics
        - soft_dice: global threshold-free soft Dice
        - thr: threshold used for hard metrics
        - optionally tp/tn/fp/fn/valid_pix when return_counts=True
    """
    probs = torch.sigmoid(logits)

    # valid mask -> [B,1,H,W]
    v = (valid > 0.5).float()
    if v.ndim == 3:
        v = v.unsqueeze(1)

    # drop ignore_index pixels from the valid region
    not_ign = (mask != ignore_index).unsqueeze(1).float()
    v = v * not_ign

    # targets -> [B,1,H,W]
    y = (mask == 1).float().unsqueeze(1)
    yv = y * v

    eps = 1e-8

    # soft Dice (threshold-free), global
    pv = probs * v
    inter_soft = (pv * yv).sum()
    denom_soft = pv.sum() + yv.sum()
    soft_dice = float((2 * inter_soft + eps) / (denom_soft + eps))

    def hard_metrics_at_threshold(t: float) -> Dict[str, Tensor]:
        pred = (probs >= t).float() * v

        tp = (pred * yv).sum()
        sum_p = pred.sum()
        sum_y = yv.sum()

        fp = (sum_p - tp).clamp_min(0.0)
        fn = (sum_y - tp).clamp_min(0.0)
        union = tp + fp + fn

        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        iou = (tp + eps) / (union + eps)

        prec = tp / (tp + fp + eps) if (tp + fp) > 0 else torch.tensor(0.0, device=logits.device)
        rec = tp / (tp + fn + eps) if (tp + fn) > 0 else torch.tensor(0.0, device=logits.device)

        tn = ((1.0 - pred) * (1.0 - y) * v).sum()
        valid_pix = v.sum()
        acc = (tp + tn) / valid_pix.clamp_min(1.0)

        return {
            "dice": dice,
            "iou": iou,
            "prec": prec,
            "rec": rec,
            "acc": acc,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "valid_pix": valid_pix,
        }

    # threshold selection (optional sweep)
    best_thr = thr
    if thr is None:
        if thr_grid is None:
            thr_grid = torch.linspace(0.01, 0.99, 99, device=logits.device).tolist()

        def score_from_metrics(ms: Dict[str, Tensor], select_by: str) -> float:
            if select_by == "iou":
                return float(ms["iou"].item())
            if select_by == "dice":
                return float(ms["dice"].item())
            if select_by == "f1":
                p = float(ms["prec"].item())
                r = float(ms["rec"].item())
                if (p + r) <= 0.0:
                    return -1.0
                return float(2.0 * p * r / (p + r + 1e-8))
            raise ValueError(f"Unknown select_by='{select_by}'")

        best_thr, best_score = float(thr_grid[0]), -1.0
        for t in thr_grid:
            ms = hard_metrics_at_threshold(float(t))
            score = score_from_metrics(ms, select_by)
            if score > best_score:
                best_score, best_thr = score, float(t)

    ms = hard_metrics_at_threshold(float(best_thr))

    out = {
        "dice": float(ms["dice"].item()),
        "iou": float(ms["iou"].item()),
        "acc": float(ms["acc"].item()),
        "precision": float(ms["prec"].item()),
        "recall": float(ms["rec"].item()),
        "soft_dice": float(soft_dice),
        "thr": float(best_thr),
    }

    if return_counts:
        out.update(
            {
                "tp": float(ms["tp"].item()),
                "tn": float(ms["tn"].item()),
                "fp": float(ms["fp"].item()),
                "fn": float(ms["fn"].item()),
                "valid_pix": float(ms["valid_pix"].item()),
            }
        )

    return out

# ----------------------------- Train Epoch (with KD) -----------------------------

def train_one_epoch_kd(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    *,
    resize_to: Optional[int],
    pos_weight: Optional[float],
    amp: bool,
    grad_clip: float,
    # KD controls:
    use_kd: bool,
    distill_type: str,  # "logits" | "features" | "hybrid" | "none"
    teacher: Optional[nn.Module],
    logits_cfg: Optional[Dict[str, Any]],
    features_cfg: Optional[Dict[str, Any]],
    hybrid_cfg: Optional[Dict[str, Any]],
    metric_thr: float = 0.5,
) -> SimpleNamespace:
    
    """
    Train the student model for one epoch with optional knowledge distillation.

    Performs a full training pass over loader: computes the segmentation (focal)
    loss on student logits, optionally adds KD losses (logits / features / hybrid)
    computed against teacher, runs backprop (with AMP if enabled), optimizer
    step, and aggregates running losses and global confusion counts (tp/tn/fp/fn).

    Returns a SimpleNamespace containing aggregated epoch statistics (loss, hard,
    kd, n, and metric totals like tp/tn/fp/fn/valid_pix) used by the trainer loop.
    """

    model.train()

    # Loss-like values remain sample-weighted averages across the epoch.
    running = {
        "loss": 0.0,
        "hard": 0.0,
        "kd": 0.0,
        "n": 0,
    }

    # Metrics become true epoch-global via accumulated confusion counts.
    metric_totals = {
        "tp": 0.0,
        "tn": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "valid_pix": 0.0,
    }

    posw_tensor = torch.tensor(pos_weight, device=device) if pos_weight else None

    for batch in loader:
        imgs, masks, valid_mask = _unpack_batch_rc(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            student_logits = model(imgs)

            # Common size
            if resize_to is not None:
                student_logits = F.interpolate(
                    student_logits,
                    size=(resize_to, resize_to),
                    mode="bilinear",
                    align_corners=False,
                )
                masks_use = (
                    None
                    if masks is None
                    else F.interpolate(
                        masks.unsqueeze(1),
                        size=(resize_to, resize_to),
                        mode="nearest",
                    ).squeeze(1)
                )
                vm_use = (
                    None
                    if valid_mask is None
                    else F.interpolate(
                        valid_mask.float(),
                        size=(resize_to, resize_to),
                        mode="nearest",
                    ).bool()
                )
            else:
                masks_use = masks
                vm_use = valid_mask

            ##############Hard loss with focal loss###################
            hard = student_logits.new_tensor(0.0)
            if masks_use is not None:
                target = masks_use if masks_use.ndim == 3 else masks_use.squeeze(1)
                hard = focal_loss_only(
                    student_logits,
                    target,
                    vm_use,
                    gamma=2.0,
                    alpha=0.6,
                    ignore_index=255,
                )

            # KD
            kd_val = student_logits.new_tensor(0.0)
            feat_kd = student_logits.new_tensor(0.0)

            if use_kd and distill_type != "none":
                if distill_type in ("logits", "hybrid"):
                    T = float(logits_cfg.get("temperature", 2.0))
                    kd_kind = str(logits_cfg.get("kd_loss", "binary_bce")).lower()

                    with torch.no_grad():
                        tch_logits = forward_teacher(teacher, imgs)

                        if resize_to is not None:
                            tch_logits = F.interpolate(
                                tch_logits,
                                size=(resize_to, resize_to),
                                mode="bilinear",
                                align_corners=False,
                            )

                        if tch_logits.shape[-2:] != student_logits.shape[-2:]:
                            tch_logits = F.interpolate(
                                tch_logits,
                                size=student_logits.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )

                    kd_val = (
                        kd_logit_mse(student_logits, tch_logits, valid_mask=vm_use, T=T)
                        if kd_kind == "logit_mse"
                        else kd_binary_bce(student_logits, tch_logits, valid_mask=vm_use, T=T)
                    )

                if distill_type in ("features", "hybrid"):
                    feat_kd, _ = compute_feature_kd(model, teacher, imgs, features_cfg)

                if distill_type == "logits":
                    alpha_hard = float(logits_cfg.get("alpha_hard", 0.6))
                    total = kd_val if masks_use is None else alpha_hard * hard + (1.0 - alpha_hard) * kd_val
                elif distill_type == "features":
                    alpha_hard = float(features_cfg.get("alpha_hard", 0.6))
                    total = feat_kd if masks_use is None else alpha_hard * hard + (1.0 - alpha_hard) * feat_kd
                else:  # hybrid
                    alpha_logits = float(hybrid_cfg.get("alpha_logits", 0.5))
                    kd_combo = alpha_logits * kd_val + (1.0 - alpha_logits) * feat_kd
                    alpha_hard = float(logits_cfg.get("alpha_hard", 0.6))
                    total = kd_combo if masks_use is None else alpha_hard * hard + (1.0 - alpha_hard) * kd_combo
            else:
                total = hard

        # Optimizer step
        if scaler is not None:
            scaler.scale(total).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Accumulate metrics as raw counts for true epoch-global metrics
        with torch.no_grad():
            if masks_use is not None:
                batch_metrics = compute_metrics(
                    logits=student_logits,
                    mask=masks_use,
                    valid=vm_use,
                    thr=metric_thr,
                    return_counts= True,
                )
                metric_totals["tp"] += batch_metrics["tp"]
                metric_totals["tn"] += batch_metrics["tn"]
                metric_totals["fp"] += batch_metrics["fp"]
                metric_totals["fn"] += batch_metrics["fn"]
                metric_totals["valid_pix"] += batch_metrics["valid_pix"]

        bsz = imgs.size(0)
        running["loss"] += float(total.item()) * bsz
        running["hard"] += float(hard.item()) * bsz
        running["kd"] += float((kd_val + feat_kd).item()) * bsz
        running["n"] += bsz

    n = max(1, running["n"])

    # Final epoch-global metrics from accumulated counts
    eps = 1e-8
    tp = metric_totals["tp"]
    tn = metric_totals["tn"]
    fp = metric_totals["fp"]
    fn = metric_totals["fn"]
    valid_pix = max(metric_totals["valid_pix"], eps)

    if valid_pix > eps:
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        acc = (tp + tn) / valid_pix
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
    else:
        dice = 0.0
        iou = 0.0
        acc = 0.0
        precision = 0.0
        recall = 0.0

    return SimpleNamespace(
        loss=running["loss"] / n,
        hard=running["hard"] / n,
        kd=running["kd"] / n,
        dice=dice,
        iou=iou,
        acc=acc,
        precision=precision,
        recall=recall,
    )


# ----------------------------- Builders -----------------------------
def build_student(cfg: dict, device: torch.device) -> nn.Module:
    """
    Build and return the student model.

    Chooses UNetSmall when available (HAS_SMALL) otherwise UNet.
    Model hyperparameters are taken from the provided cfg dict and the
    final model is moved to the provided device.
    """
    if HAS_SMALL:
        model = UNetSmall(
            in_channels=int(cfg.get("in_channels", 4)),
            out_channels=int(cfg.get("out_channels", 1)),
            base_channels=int(cfg.get("base_channels", 16)),
            depth=int(cfg.get("depth", 3)),
            bilinear=bool(cfg.get("bilinear", True)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    else:
        model = UNet(
            in_channels=int(cfg.get("in_channels", 4)),
            out_channels=int(cfg.get("out_channels", 1)),
            base_channels=int(cfg.get("base_channels", 32)),
            depth=int(cfg.get("depth", 4)),
            bilinear=bool(cfg.get("bilinear", True)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    return model.to(device)

def build_teacher_for_kd(teacher_cfg: dict, device: torch.device) -> nn.Module:
    """
    Build and return a frozen teacher UNet for knowledge distillation.

    The teacher model is instantiated from the provided teacher_cfg, weights
    are loaded from the 'ckpt' path, and the model is set to eval with
    gradients disabled.
    """
    t = UNet(
        in_channels=int(teacher_cfg.get("in_channels", 4)),
        out_channels=int(teacher_cfg.get("out_channels", 1)),
        base_channels=int(teacher_cfg.get("base_channels", 32)),
        depth=int(teacher_cfg.get("depth", 4)),
        bilinear=bool(teacher_cfg.get("bilinear", True)),
        dropout=float(teacher_cfg.get("dropout", 0.0)),
    ).to(device)
    ckpt = teacher_cfg.get("ckpt", None)
    if not ckpt:
        raise ValueError("KD enabled but distill.teacher.ckpt not provided in config.")
    load_weights(t, Path(ckpt), device=device)
    for p in t.parameters(): p.requires_grad_(False)
    t.eval()
    return t


# ----------------------------- wandb utils -----------------------------
def _wandb_setup(cfg: dict) -> Tuple[bool, dict]:
    """
    Set up Weights & Biases logging from configuration.

    Returns a tuple (enabled, log_cfg). If wandb is not enabled or not
    installed, returns (False, {}). When enabled, initializes wandb with
    project/run settings from cfg and returns (True, wandb_config).
    """
    logger_cfg = cfg.get("logger", {})
    log_cfg = logger_cfg.get("wandb", {})
    enable = bool(log_cfg.get("enable", False))
    if not enable: return False, {}
    if not _WANDB_AVAILABLE:
        LOGGER.warning("wandb enabled but not installed; skipping.")
        return False, {}
    wandb.init(
        project=log_cfg.get("project", "opensar-insight"),
        name=log_cfg.get("run_name", None),
        entity=log_cfg.get("entity", None),
        config={ "train": cfg.get("train", {}), "loader": cfg.get("loader", {}),
                 "model": cfg.get("model", {}), "distill": cfg.get("distill", {}),
                 "data": cfg.get("data", {}) },
    )
    return True, log_cfg


# ----------------------------- Main -----------------------------
def main():
    args = parse_args()

    # Load configs/paths
    cfg = load_yaml_config(args.config)
        # Optional CLI overrides
    if args.base_channels is not None:
        cfg.setdefault("model", {})
        cfg["model"]["base_channels"] = int(args.base_channels)
    tfm_cfg_path = args.transforms or cfg.get("transforms_cfg")
    if tfm_cfg_path is None:
        raise ValueError("No transforms file provided (neither in --transforms nor in config.transforms_cfg).")
    tfm_cfg = load_yaml_config(tfm_cfg_path)

    # Device override (CLI has priority)
    device_pref = (args.device or str(cfg.get("device", "cuda"))).lower()

    data_cfg    = cfg.get("data", {})
    loader_cfg  = cfg.get("loader", {})
    train_cfg   = cfg.get("train", {})
    save_cfg    = cfg.get("save", {})
    student_cfg = cfg.get("model", {})
    distill_cfg = cfg.get("distill", {})
    teacher_cfg = distill_cfg.get("teacher", {})

    # W&B
    log_cfg_enabled, wandb_cfg = _wandb_setup(cfg)

    # Paths / device / seed
    train_dir = Path(data_cfg["train_dir"])
    val_dir   = Path(data_cfg["val_dir"])
    save_dir  = Path(save_cfg.get("dir", PIPELINE_DIR / "checkpoints"))
    ensure_dir(save_dir)

    device = torch.device("cuda" if (device_pref == "cuda" and torch.cuda.is_available()) else "cpu")
    LOGGER.info("Device: %s (CUDA available: %s)", device, torch.cuda.is_available())
    set_seed(int(train_cfg.get("seed", 42)))

    # Dataloaders
    batch_size      = int(loader_cfg.get("batch_size"))
    num_workers     = int(loader_cfg.get("num_workers"))
    stride_multiple = int(loader_cfg.get("stride_multiple"))
    ignore_index    = int(loader_cfg.get("ignore_index"))
    train_loader = build_dataloader(train_dir, tfm_cfg, batch_size, num_workers, stride_multiple, ignore_index, True, "train")
    val_loader   = build_dataloader(val_dir, make_val_cfg_from_train_cfg(tfm_cfg), batch_size, num_workers, stride_multiple, ignore_index, False, "val")

    # Models
    student = build_student(student_cfg, device)
    LOGGER.info("Student params: %.2fM", count_params(student))

    use_kd       = bool(distill_cfg.get("enable", False))
    distill_type = str(distill_cfg.get("type", "logits")).lower() if use_kd else "none"
    logits_cfg   = distill_cfg.get("logits", {})
    features_cfg = distill_cfg.get("features", {})
    hybrid_cfg   = distill_cfg.get("hybrid", {})
    teacher      = build_teacher_for_kd(teacher_cfg, device) if use_kd else None
    if use_kd:
        LOGGER.info("KD: type=%s | Teacher params: %.2fM (trainable=%.2fM) | ckpt=%s",
                    distill_type, count_params(teacher, False), count_params(teacher, True), teacher_cfg.get("ckpt", ""))

    # Optim/sched/AMP
    epochs    = int(train_cfg.get("epochs"))
    lr        = float(train_cfg.get("lr"))
    wd        = float(train_cfg.get("weight_decay"))
    optimizer = make_optimizer(student, str(train_cfg.get("optimizer")), lr, wd)
    scheduler = make_scheduler(optimizer, str(train_cfg.get("scheduler")), epochs, int(train_cfg.get("warmup_epochs")))
    amp       = bool(train_cfg.get("amp"))
    grad_clip = float(train_cfg.get("grad_clip"))
    eval_every= int(train_cfg.get("eval_every"))
    resize_to = train_cfg.get("resize_to")
    resize_to = int(resize_to) if (resize_to is not None and str(resize_to).lower() != "none") else None
    scaler    = GradScaler(device="cuda") if (amp and device.type == "cuda") else None

    # pos_weight
    posw = train_cfg.get("pos_weight")
    if isinstance(posw, (int, float)):
        pos_weight_value = float(posw)
    elif isinstance(posw, str) and posw.lower() == "none":
        pos_weight_value = None
    else:
        pos_weight_value = estimate_pos_weight(train_loader, max_batches=10, resize_to=resize_to, device=device)

    if log_cfg_enabled and _WANDB_AVAILABLE:
        wandb.watch(student, log="gradients", log_freq=100)

    # Train loop
    start_epoch, best_dice = 0, 0.0
    history = {k: [] for k in ["train_loss","val_loss","train_acc","val_acc","train_dice","val_dice","train_iou","val_iou","train_precision","val_precision","train_recall","val_recall","train_kd"]}
    rows = []
    log_images_every = int(wandb_cfg.get("log_images_every", 1)) if log_cfg_enabled else 0
    num_val_images   = int(wandb_cfg.get("num_val_images", 4)) if log_cfg_enabled else 0

    for epoch in range(start_epoch, epochs):
        tr = train_one_epoch_kd(
            student, train_loader, optimizer, scaler, device,
            resize_to=resize_to, pos_weight=pos_weight_value, amp=amp, grad_clip=grad_clip,
            use_kd=use_kd, distill_type=distill_type,
            teacher=teacher, logits_cfg=logits_cfg, features_cfg=features_cfg, hybrid_cfg=hybrid_cfg,
        )
        if scheduler:
            scheduler.step()

        LOGGER.info(
            "Epoch %3d/%3d | Train: loss=%.5f (hard=%.5f kd=%.5f)  | LR=%.3e",
            epoch + 1, epochs, tr.loss, tr.hard, tr.kd,
            optimizer.param_groups[0]["lr"]
        )

        if log_cfg_enabled and _WANDB_AVAILABLE:
            wandb.log({
                "train/loss": tr.loss,
                "train/loss_hard": tr.hard,
                "train/loss_kd": tr.kd,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch + 1
            })

        if ((epoch + 1) % max(1, eval_every)) == 0:
            va, val_thr = validate_one_epoch(
                model=student,
                loader=val_loader,
                device=device,
                resize_to=resize_to,
                ignore_index=ignore_index,
                pos_weight=pos_weight_value,
            )


            train_eval = evaluate_one_epoch_fixed_thr(
                model=student,
                loader=train_loader,
                device=device,
                resize_to=resize_to,
                pos_weight=pos_weight_value,
                fixed_thr=val_thr,
                ignore_index=ignore_index,
            )

            LOGGER.info(
                "Epoch %3d/%3d | Valid: loss=%.5f dice=%.4f iou=%.4f acc=%.4f precision=%.4f recall=%.4f | thr=%.3f ",
                epoch + 1, epochs,
                va.loss, va.dice, va.iou, va.acc, va.precision, va.recall,
                val_thr,
            )

            LOGGER.info(
                "Epoch %3d/%3d | Train@val_thr: dice=%.4f iou=%.4f acc=%.4f precision=%.4f recall=%.4f | thr=%.3f",
                epoch + 1, epochs,
                train_eval.dice, train_eval.iou, train_eval.acc,
                train_eval.precision, train_eval.recall,
                val_thr,
            )

            history["train_loss"].append(tr.loss)
            history["val_loss"].append(va.loss)
            history["train_acc"].append(train_eval.acc)
            history["val_acc"].append(va.acc)
            history["train_dice"].append(train_eval.dice)
            history["val_dice"].append(va.dice)
            history["train_iou"].append(train_eval.iou)
            history["val_iou"].append(va.iou)
            history["train_precision"].append(train_eval.precision)
            history["val_precision"].append(va.precision)
            history["train_recall"].append(train_eval.recall)
            history["val_recall"].append(va.recall)
            history["train_kd"].append(tr.kd)

            save_weights(student, save_dir / "last_student.pth")
            save_weights(student, save_dir / f"student_epoch_{epoch + 1:03d}.pth")

            if va.dice > best_dice:
                best_dice = va.dice
                save_weights(student, save_dir / "best_student.pth")

            rows.append({
                "epoch": epoch + 1,
                "train_loss": tr.loss,
                "val_loss": va.loss,
                "train_acc": train_eval.acc,
                "val_acc": va.acc,
                "train_dice": train_eval.dice,
                "val_dice": va.dice,
                "train_iou": train_eval.iou,
                "val_iou": va.iou,
                "train_kd": tr.kd if use_kd else 0.0,
                "train_precision": train_eval.precision,
                "val_precision": va.precision,
                "train_recall": train_eval.recall,
                "val_recall": va.recall,
                "lr": optimizer.param_groups[0]["lr"],
                "thr": val_thr,
            })

            if log_cfg_enabled and _WANDB_AVAILABLE:
                wandb.log({
                    "val/loss": va.loss,
                    "val/dice": va.dice,
                    "val/iou": va.iou,
                    "val/acc": va.acc,
                    "val/precision": va.precision,
                    "val/recall": va.recall,
                    "train_eval/loss": train_eval.loss,
                    "train_eval/dice": train_eval.dice,
                    "train_eval/iou": train_eval.iou,
                    "train_eval/acc": train_eval.acc,
                    "train_eval/precision": train_eval.precision,
                    "train_eval/recall": train_eval.recall,
                    "val/threshold": val_thr,
                    "epoch": epoch + 1,
                })
                if log_images_every > 0 and ((epoch + 1) % log_images_every == 0):
                    log_validation_images_wandb(
                        student, val_loader, device, resize_to=resize_to,
                        max_images=num_val_images, prefix="val"
                    )

    # Persist logs/curves
    save_metrics_csv(save_dir / "metrics_student.csv", rows)
    
    # plot_training_curves(history["train_loss"], history["val_loss"], history["train_acc"], history["val_acc"], history["train_precision"], history["val_precision"], history["train_recall"], history["val_recall"], output_dir=str(save_dir))
    plot_training_curves(
    train_losses=history["train_loss"],
    val_losses=history["val_loss"],
    train_acc=history["train_acc"],
    val_acc=history["val_acc"],
    train_dice=history["train_dice"],
    val_dice=history["val_dice"],
    train_iou=history["train_iou"],
    val_iou=history["val_iou"],
    train_precision=history["train_precision"],
    val_precision=history["val_precision"],
    train_recall=history["train_recall"],
    val_recall=history["val_recall"],
    output_dir=str(save_dir),
)
    LOGGER.info("Training complete. Best validation Dice: %.4f", best_dice)
    if log_cfg_enabled and _WANDB_AVAILABLE and wandb.run is not None: wandb.finish()


if __name__ == "__main__":
    main()
