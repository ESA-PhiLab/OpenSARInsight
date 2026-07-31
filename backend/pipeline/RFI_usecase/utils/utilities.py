"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: utilities.py

Description:
Utility functions are defined here and used in other scripts for RFI model training. 

History:
    - 2025-12-05:
        First version is ready.
    - 2025-12-08:
        The script is finalized to be used in other scripts. 
    
---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-12-05

© Copyright INDRA DEIMOS, 2025. All rights reserved.
---------------------------------------------------------------------
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Sequence
import torch
import torch.nn.functional as F
from torch import Tensor
from pipeline.RFI_usecase.utils.logging_setup import init_logging


# ------------------------- Logging ------------------------- #
init_logging(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


# ------------------------- Utilities ------------------------- #
def save_weights(model: torch.nn.Module, path: Path) -> None:
    """Save a model's state_dict to disk.

    Ensures the parent directory exists then writes the module's state_dict
    to the given path using torch.save.

    Args:
        model: torch.nn.Module whose weights will be saved.
        path: Path where the state_dict will be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def load_weights(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    """Load a saved state_dict from disk and apply it to the given model.

    Args:
        model: The torch.nn.Module instance to receive the weights.
        path: Filesystem path to the saved state_dict (as written by
            save_weights / torch.save).
        device: torch.device (or device spec) used as map_location for loading.
    """
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd, strict=True)

def set_seed(seed: int) -> None:
    """
    Configure pseudo-random number generators for reproducible runs.
    Args:
            seed: Integer seed to initialize all RNGs.

    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(p: Path) -> None:
    """
    Ensure a directory exists at the given path.
    Args:
        p: Path to the directory to ensure exists.
    """
    p.mkdir(parents=True, exist_ok=True)


def estimate_pos_weight(
    loader: torch.utils.data.DataLoader,
    max_batches: Optional[int],
    resize_to: Optional[int],
    device: torch.device,
    cap: float = 14.0,
    ignore_index: int = 255,
) -> float:
    """
    Estimate BCE pos_weight ≈ #neg / #pos over VALID, NON-IGNORED pixels (after resize).

    - Uses both batch.valid_mask and mask != ignore_index to define the valid region.
    - Set max_batches=None to scan the full loader.
    - Caps the result at cap for stability.
    """
    pos_pix = 0.0
    neg_pix = 0.0
    T = int(resize_to)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            # mask: [B,H,W] int {0,1,ignore_index}
            m = batch.mask.to(device, non_blocking=True)            # [B,H,W]
            # valid_mask: [B,1,H,W] float {0,1}
            v = batch.valid_mask.to(device, non_blocking=True)      # [B,1,H,W]

            # Resize to (T,T) with nearest for both
            m_r = F.interpolate(m.unsqueeze(1).float(),
                                size=(T, T),
                                mode="nearest").squeeze(1).to(torch.int64)   # [B,T,T]
            v_r = F.interpolate(v.float(),
                                size=(T, T),
                                mode="nearest")                              # [B,1,T,T]

            # Effective valid region: inside valid_mask AND not ignore_index
            valid_eff = (v_r > 0.5) & (m_r.unsqueeze(1) != ignore_index)     # [B,1,T,T]

            # Positive pixels (label==1) within effective valid region
            pos = (m_r == 1).unsqueeze(1) & valid_eff                        # [B,1,T,T]

            p_count = pos.sum().item()
            v_count = valid_eff.sum().item()  # total valid (pos + neg) pixels

            if v_count == 0:
                continue  # nothing usable in this batch

            pos_pix += p_count
            neg_pix += (v_count - p_count)

    # Fallback if no positives or no negatives seen
    if pos_pix <= 0.0 or neg_pix <= 0.0:
        return 1.0

    w = (neg_pix + 1e-8) / (pos_pix + 1e-8)
    w = float(min(w, cap))
    return w


def make_optimizer(model: torch.nn.Module, name: str, lr: float, wd: float):
    """
    Create an optimizer for the given model.

    Supported optimizers (case-insensitive):
      - "adamw": torch.optim.AdamW
      - "adam": torch.optim.Adam
      - "sgd": torch.optim.SGD (with momentum=0.9, nesterov=True)

    Args:
        model: Module whose parameters will be optimized. Only params with
            requires_grad=True are included.
        name: Name of the optimizer to create (adamw, adam, sgd).
        lr: Base learning rate.
        wd: Weight decay (L2) coefficient.

    Returns:
        An instantiated torch.optim.Optimizer.

    Raises:
        ValueError: If an unsupported optimizer name is provided.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "adam":                                   # <-- add
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {name}")


def make_scheduler(optimizer, name: str, epochs: int, warmup_epochs: int):
    """
    Epoch-level scheduler; extend as needed.
    warmup_epochs: Number of initial epochs used to ramp the learning rate up from near-zero to the base LR. 
    It prevents early instability or divergence.
    """
    name = name.lower()
    if name in {"none", "off"}:
        return None
    if name == "cosine":
        def lr_lambda(ep: int) -> float:
            if ep < warmup_epochs:
                return float(ep + 1) / max(1, warmup_epochs)
            prog = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * prog))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if name == "multistep":
        m1, m2 = max(1, int(0.6 * epochs)), max(1, int(0.85 * epochs))
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[m1, m2], gamma=0.2)
    raise ValueError(f"Unsupported scheduler: {name}")


def resize_for_loss(
    logits: Tensor,
    mask: Tensor,
    valid: Tensor,
    out_size: Optional[int] = None,
    ignore_index: int = 255,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Prepare logits, mask and validity tensors for loss computation.

    This helper performs two related tasks used by training/evaluation code:
      1) Optionally resize the spatial dimensions of logits, mask and valid
         to a square output of size (T, T) when out_size is provided.
      2) Construct a cleaned per-pixel validity mask (va) that combines the
         provided `valid` mask and the `ignore_index` pixels present in
         `mask`.

    Behaviour:
      - If out_size is None:
          * Do not change spatial sizes. Return the inputs (logits, mask)
            unchanged except for producing the cleaned validity map `va`.
      - If out_size is an int T:
          * Resize logits using bilinear interpolation (align_corners=False).
          * Resize mask and valid using nearest neighbour to preserve labels.
          * Return resized tensors and the cleaned validity mask computed on
            the resized mask.

    Args:
        logits: Raw model outputs, shape [B,1,H,W].
        mask: Integer ground-truth mask, shape [B,H,W], values in {0,1,ignore}.
        valid: Validity mask, either [B,H,W] or [B,1,H,W], values in {0,1}.
        out_size: If provided, resize spatial dims to (out_size, out_size).
        ignore_index: Pixel value in `mask` that should be ignored.

    Returns:
        lo: Logits tensor (possibly resized) with shape [B,1,T,T] or original.
        ma: Mask tensor (possibly resized) with shape [B,T,T] or original.
        va: Boolean validity tensor (masked non-ignored pixels), shape [B,T,T]
            or [B,H,W] depending on out_size; True where pixel is valid and
            not equal to ignore_index.
    """
    if out_size is None:
        lo = logits
        ma = mask

        # valid -> [B,1,H,W] float
        v4 = valid.unsqueeze(1).float() if valid.ndim == 3 else valid.float()
        # drop ignore pixels
        va = (v4.squeeze(1) > 0.5) & (ma != ignore_index)
        return lo, ma, va

    # --- original behaviour for fixed T×T ---
    T = int(out_size)

    lo = F.interpolate(logits, size=(T, T), mode="bilinear", align_corners=False)
    ma = F.interpolate(mask.unsqueeze(1).float(), size=(T, T), mode="nearest").squeeze(1).to(mask.dtype)

    v4 = valid.unsqueeze(1).float() if valid.ndim == 3 else valid.float()
    va = F.interpolate(v4, size=(T, T), mode="nearest").squeeze(1) > 0.5
    va = va & (ma != ignore_index)

    return lo, ma, va


# ----------------------- Setting up Focal loss --------------------- #

def focal_loss_only(
    logits: Tensor,                 # [B,1,H,W] float (raw scores)
    mask: Tensor,                   # [B,H,W] in {0,1,ignore_index}
    valid: Tensor,                  # [B,H,W] or [B,1,H,W]
    gamma: float = 2.0,             # focusing parameter
    alpha: Optional[float] = None,  # class balance (None = no reweighting)
    ignore_index: int = 255,
    eps: float = 1e-8,
) -> Tensor:
    """
    Binary focal loss computed from raw logits with masking and optional class reweighting.

    This implements the standard binary focal loss applied to raw model logits
    (no sigmoid pre-applied). The loss is computed only over pixels marked as
    valid and not equal to ignore_index in mask.

    Normalization: the summed loss is divided by the sum of class-weights
    multiplied by the validity mask.

    Args:
        logits: Raw model outputs, shape [B,1,H,W].
        mask: Integer ground-truth mask, shape [B,H,W], values in {0,1,ignore_index}.
        valid: Validity mask, shape [B,H,W] or [B,1,H,W], values in {0,1}.
        gamma: Focusing parameter (>=0). Larger gamma down-weights well-classified
            examples more strongly.
        alpha: Optional class weighting for positives. If None no alpha reweighting
            is applied. If provided, should be a scalar in [0,1].
        ignore_index: Pixel value in ``mask`` that is ignored (default 255).
        eps: Small constant for numerical stability.

    Returns:
        A scalar tensor containing the averaged focal loss.

    Notes:
        - The implementation uses stable computations via log-sigmoid to avoid
          numerical issues when converting logits to probabilities.
        - The returned loss is already reduced and ready for backprop.
    """
    # valid -> [B,1,H,W] float
    w = valid.unsqueeze(1).float() if valid.ndim == 3 else valid.float()
    w = w * (mask.unsqueeze(1) != ignore_index).float()

    # targets -> [B,1,H,W] in {0,1}
    tgt01 = (mask == 1).to(dtype=logits.dtype).unsqueeze(1)

    # Stable pt and logpt
    # log σ(z) for positives; log σ(-z) for negatives
    logpt = F.logsigmoid(logits) * tgt01 + F.logsigmoid(-logits) * (1.0 - tgt01)   # [B,1,H,W]
    pt = logpt.exp().clamp_min(eps)                                                # in (0,1]

    # Optional alpha reweighting (usually you can leave alpha=None)
    if alpha is None:
        at = torch.ones_like(pt)
    else:
        at = alpha * tgt01 + (1.0 - alpha) * (1.0 - tgt01)

    # Focal loss map
    focal_factor = (1.0 - pt).clamp_min(eps).pow(gamma)
    loss_map = - at * focal_factor * logpt                                          # [B,1,H,W]

    # ---- normalization  ----
    num = (loss_map * w).sum()
    den = (at * w).sum().clamp_min(1.0)
    return num / den


# ----------------------- Compute metrics function --------------------- #

@torch.no_grad()
def compute_metrics(
    logits: Tensor,                  # [B,1,H,W] (raw logits)
    mask: Tensor,                    # [B,H,W] with {0,1,255}
    valid: Tensor,                   # [B,H,W] or [B,1,H,W]
    thr: Optional[float] = None,     # if None -> sweep to pick best (by select_by)
    thr_grid: Optional[Sequence[float]] = None,  # e.g., torch.linspace(0.01,0.99,99)
    select_by: str = "dice",           # "dice", "iou" or "f1" (for sweep only)
    ignore_index: int = 255,
) -> Dict[str, float]:
    """
    Computes global metrics on VALID, NON-IGNORED pixels only.

    All metrics (Dice, IoU, Precision, Recall, Accuracy) are computed from
    globally pooled TP/FP/FN/TN across all pixels in logits/mask/valid.

    Returns:
        - dice, iou, precision, recall, acc: global metrics
        - soft_dice: global threshold-free soft Dice
        - thr: threshold used for hard metrics
    """
    probs = torch.sigmoid(logits)                               # [B,1,H,W]

    # valid mask -> [B,1,H,W]
    v = (valid > 0.5).float()
    if v.ndim == 3:
        v = v.unsqueeze(1)

    # drop ignore_index pixels from the valid region
    not_ign = (mask != ignore_index).unsqueeze(1).float()       # [B,1,H,W]
    v = v * not_ign                                             # valid AND not ignore

    # targets -> [B,1,H,W]
    y = (mask == 1).float().unsqueeze(1)                        # positive class only
    yv = y * v                                                  # GT within valid, non-ignore

    eps = 1e-8

    # ---- soft Dice (threshold-free) ----
    # use probabilities, but only in valid & non-ignore region
    pv = probs * v
    inter_soft = (pv * yv).sum()                                
    denom_soft = pv.sum() + yv.sum()
    soft_dice = float((2 * inter_soft + eps) / (denom_soft + eps))

    def hard_metrics_at_threshold(t: float) -> Dict[str, Tensor]:
        """
        Compute global metrics at threshold t.
        Everything is pooled over [B,1,H,W] within valid & non-ignore.
        """
        pred = (probs >= t).float() * v                         # [B,1,H,W], masked to valid & non-ignore

        # Global sums over all pixels
        tp = (pred * yv).sum()                                  # TP
        sum_p = pred.sum()                                      # TP + FP
        sum_y = yv.sum()                                        # TP + FN

        fp = (sum_p - tp).clamp_min(0.0)
        fn = (sum_y - tp).clamp_min(0.0)

        # union = TP + FP + FN (for binary case)
        union = tp + fp + fn

        # Global Dice / IoU
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        iou  = (tp + eps) / (union + eps)

        # Global Precision / Recall
        # Here, if denom is zero, we set metric to 0.0
        prec = tp / (tp + fp + eps) if (tp + fp) > 0 else torch.tensor(0.0, device=logits.device)
        rec  = tp / (tp + fn + eps) if (tp + fn) > 0 else torch.tensor(0.0, device=logits.device)

        # Global Accuracy over valid & non-ignore pixels
        tn = ((1.0 - pred) * (1.0 - y) * v).sum()
        acc = (tp + tn) / v.sum().clamp_min(1.0)

        return {
            "dice": dice,         # scalar tensor
            "iou":  iou,
            "prec": prec,
            "rec":  rec,
            "acc":  acc,
        }

    # ---- threshold selection (optional sweep) ----
    best_thr = thr
    if thr is None:
        if thr_grid is None:
            thr_grid = torch.linspace(0.01, 0.99, 99, device=logits.device).tolist()

        def score_from_metrics(ms: Dict[str, Tensor], select_by: str) -> float:
            # You can extend this as needed
            if select_by == "iou":
                return float(ms["iou"].item())
            elif select_by == "dice":
                return float(ms["dice"].item())
            elif select_by == "f1":
                # F1 from global precision/recall
                p = float(ms["prec"].item())
                r = float(ms["rec"].item())
                if (p + r) <= 0.0:
                    return -1.0
                return float(2.0 * p * r / (p + r + 1e-8))
            else:
                raise ValueError(f"Unknown select_by='{select_by}'")

        best_thr, best_score = float(thr_grid[0]), -1.0
        for t in thr_grid:
            ms = hard_metrics_at_threshold(float(t))
            score = score_from_metrics(ms, select_by)
            if score > best_score:
                best_score, best_thr = score, float(t)

    # ---- final metrics at best_thr (or given thr) ----
    ms = hard_metrics_at_threshold(float(best_thr))

    return {
        "dice":      float(ms["dice"].item()),
        "iou":       float(ms["iou"].item()),
        "acc":       float(ms["acc"].item()),
        "precision": float(ms["prec"].item()),
        "recall":    float(ms["rec"].item()),
        "soft_dice": float(soft_dice),
        "thr":       float(best_thr),
    }

# ----------------------- Train / Val --------------------- #

@torch.no_grad()
def pick_global_threshold_val(
    model,
    loader,
    device: torch.device,
    resize_to: Optional[int],
    thr_grid: Optional[Sequence[float]] = None,
    ignore_index: int = 255,  
) -> Tuple[float, float]:
    """
    Sweep a global threshold over the whole validation set and pick the one
    that maximises global F1 (micro pooled over all valid, non-ignored pixels).

    Returns:
        best_thr, best_f1
    """
    model.eval()

    # Run model once over the loader and store probs, y, v per-batch
    probs_all: list[Tensor] = []
    y_all: list[Tensor] = []
    v_all: list[Tensor] = []

    for batch in loader:
        rc = batch.rc.to(device, non_blocking=True)
        mask = batch.mask.to(device, non_blocking=True)
        valid = batch.valid_mask.to(device, non_blocking=True)

        logits = model(rc)
        lo, ma, va = resize_for_loss(logits, mask, valid, resize_to)

        probs = torch.sigmoid(lo)                      # [B,1,H,W]
        y = (ma == 1).float().unsqueeze(1)             # [B,1,H,W]

        # valid -> [B,1,H,W] robustly
        if va.ndim == 3:
            v = (va > 0.5).float().unsqueeze(1)        # [B,1,H,W]
        elif va.ndim == 4 and va.shape[1] == 1:
            v = (va > 0.5).float()                     # [B,1,H,W]
        else:
            raise ValueError(f"Unexpected valid mask shape: {tuple(va.shape)}")

        v = v * (ma.unsqueeze(1) != ignore_index).float()
        # ----------------------------------------------------------------

        probs_all.append(probs)
        y_all.append(y)
        v_all.append(v)

    if len(probs_all) == 0:
        raise ValueError("Validation loader produced no batches; cannot pick threshold.")

    # Prepare threshold grid
    device0 = probs_all[0].device
    if thr_grid is None:
        thr_grid_t = torch.linspace(0.1, 0.99, 99, device=device0)
    else:
        thr_grid_t = torch.tensor(list(thr_grid), device=device0, dtype=torch.float32)

    eps = 1e-8
    best_t = float(thr_grid_t[0].item())
    best_f1 = -1.0

    # For each threshold, accumulate global TP/FP/FN over all batches
    for t in thr_grid_t:
        tp_sum = torch.tensor(0.0, device=device0)
        fp_sum = torch.tensor(0.0, device=device0)
        fn_sum = torch.tensor(0.0, device=device0)

        for probs, y, v in zip(probs_all, y_all, v_all):
            yv = y * v                                  # GT inside valid & non-ignore
            pred = (probs >= t).float() * v             # prediction inside valid & non-ignore

            tp_sum += (pred * yv).sum()
            fp_sum += (pred * (1.0 - y) * v).sum()
            fn_sum += ((1.0 - pred) * yv).sum()

        prec = tp_sum / (tp_sum + fp_sum + eps)
        rec = tp_sum / (tp_sum + fn_sum + eps)
        f1 = 2.0 * prec * rec / (prec + rec + eps)

        f1_f = float(f1.item())
        if f1_f > best_f1:
            best_f1 = f1_f
            best_t = float(t.item())

    return best_t, float(best_f1)


@dataclass
class EpochStats:
    loss: float
    dice: float
    iou:  float
    acc:  float
    precision: float
    recall: float

def train_one_epoch(
    model, loader, optimizer, device: torch.device,
    resize_to: Optional[int], pos_weight: Optional[float],
    grad_clip: float, metric_thr: None,   # Optional[float] = 0.5
    accumulate_steps: int = 1,   
) -> EpochStats:
    """Train the model for a single epoch and compute epoch metrics.

    Args:
        model: torch.nn.Module to train.
        loader: iterable DataLoader yielding batches with attributes rc, mask, valid_mask.
        optimizer: optimizer used to update model parameters.
        device: torch.device where tensors are placed.
        resize_to: optional size to resize targets for loss computation.
        pos_weight: optional positive-class weight (may be used by some losses).
        grad_clip: max norm for gradient clipping (<=0 disables clipping).
        metric_thr: optional threshold for binarizing predictions for metric computation;
                    if None, soft-dice is used.
        accumulate_steps: number of mini-batches to accumulate gradients over before stepping.

    Returns:
        EpochStats containing aggregated loss and metrics for the epoch.
    """
    import math

    model.train()
    LOGGER.info("[train] using threshold: %s",
                f"{metric_thr:.3f}" if metric_thr is not None else "None (soft-dice)")

    sum_loss = sum_dice = sum_iou = sum_acc = 0.0
    sum_prec = sum_rec = 0.0
    n = 0
    n_prec = n_rec = 0

    optimizer.zero_grad(set_to_none=True)  

    for it, batch in enumerate(loader):
        rc    = batch.rc.to(device, non_blocking=True)
        mask  = batch.mask.to(device, non_blocking=True)
        valid = batch.valid_mask.to(device, non_blocking=True)

        logits = model(rc)
        lo, ma, va = resize_for_loss(logits, mask, valid, resize_to)
        loss = focal_loss_only(lo, ma, va, gamma=2.0, alpha=0.6, ignore_index=255)

        # scale loss for accumulation so total gradient matches a large batch
        loss = loss / max(1, accumulate_steps)
        loss.backward()

        # update only every k steps, or at the very end of the epoch
        do_update = ((it + 1) % accumulate_steps == 0) or (it + 1 == len(loader))
        if do_update:
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # ---- metrics & logging ----
        if metric_thr is None:
            with torch.no_grad():
                p = torch.sigmoid(lo)
                v = (va if va.ndim == 4 else va.unsqueeze(1)).float()
                p = p * v
                y = (ma == 1).float().unsqueeze(1) * v
                inter = (p * y).sum()
                den   = p.sum() + y.sum()
                dice_soft = ((2*inter + 1e-6) / (den + 1e-6)).item()
            m = {"dice": dice_soft, "iou": float("nan"), "acc": float("nan"),
                 "precision": float("nan"), "recall": float("nan")}
        else:
            m = compute_metrics(lo, ma, va, thr=float(metric_thr), select_by="dice")

        sum_loss += float(loss.item() * max(1, accumulate_steps))  # undo scaling for reporting
        sum_dice += m["dice"]; sum_iou += m.get("iou", 0.0); sum_acc += m.get("acc", 0.0)

        prec_train = m.get("precision", float("nan"))
        rec_train  = m.get("recall", float("nan"))
        if not (isinstance(prec_train, float) and math.isnan(prec_train)):
            sum_prec += prec_train; n_prec += 1
        if not (isinstance(rec_train, float) and math.isnan(rec_train)):
            sum_rec  += rec_train;  n_rec  += 1

        n += 1

    return EpochStats(
        loss=sum_loss / max(1, n),
        dice=sum_dice / max(1, n),
        iou =sum_iou  / max(1, n),
        acc =sum_acc  / max(1, n),
        precision=(sum_prec / max(1, n_prec)) if n_prec else float("nan"),
        recall   =(sum_rec  / max(1, n_rec )) if n_rec  else float("nan"),
    )


@torch.no_grad()
def evaluate_one_epoch_fixed_thr(
    model,
    loader,
    device: torch.device,
    resize_to: Optional[int],
    pos_weight: Optional[float],
    fixed_thr: float,
    ignore_index: int = 255,
) -> "EpochStats":
    """
    Evaluate the model on a data loader using a fixed probability threshold.

    Performs a dataset-level evaluation by accumulating true positives,
    false positives, false negatives and true negatives across all batches.

    Args:
        model (torch.nn.Module): The model to evaluate. Should accept input tensor and return logits.
        loader (Iterable): Data loader yielding batches with attributes `rc`, `mask`, and `valid_mask`.
        device (torch.device): Device to run inference on.
        resize_to (Optional[int]): If provided, logits/masks/valid masks are resized to this spatial size.
        pos_weight (Optional[float]): Positional weighting for the loss (unused in metrics calculation).
        fixed_thr (float): Probability threshold in [0,1] for converting sigmoids to binary predictions.
        ignore_index (int): Label value in masks that indicates ignored pixels (default 255).

    Returns:
        EpochStats: Aggregate statistics for the whole loader: loss, dice, iou, acc, precision, recall.
    """
    model.eval()

    eps = 1e-8
    sum_loss = 0.0
    n_batches = 0

    # Global (dataset-level) counts
    tp = fp = fn = tn = 0.0
    valid_pixels = 0.0

    for batch in loader:
        rc = batch.rc.to(device, non_blocking=True)
        mask = batch.mask.to(device, non_blocking=True)
        valid = batch.valid_mask.to(device, non_blocking=True)

        logits = model(rc)
        lo, ma, va = resize_for_loss(logits, mask, valid, resize_to)

        # Loss (same as training/validation)
        loss = focal_loss_only(lo, ma, va, gamma=2.0, alpha=0.6, ignore_index=ignore_index)

        sum_loss += float(loss.item())
        n_batches += 1

        probs = torch.sigmoid(lo)  # [B,1,H,W]
        pred = (probs >= float(fixed_thr))  # bool [B,1,H,W]

        # Ensure mask is [B,1,H,W]
        if ma.ndim == 3:
            ma_ = ma.unsqueeze(1)
        elif ma.ndim == 4:
            ma_ = ma
        else:
            raise ValueError(f"Unexpected mask ndim={ma.ndim}")

        # Ensure valid is [B,1,H,W] boolean
        if va.ndim == 3:
            v_ = va.unsqueeze(1)
        elif va.ndim == 4:
            v_ = va
        else:
            raise ValueError(f"Unexpected valid ndim={va.ndim}")
        v_ = (v_ > 0.5)

        # Exclude ignore_index pixels
        not_ign = (ma_ != ignore_index)
        v_ = v_ & not_ign

        # GT positives
        y = (ma_ == 1)

        # Accumulate counts on valid pixels only
        tp_b = (pred & y & v_).sum().item()
        fp_b = (pred & (~y) & v_).sum().item()
        fn_b = ((~pred) & y & v_).sum().item()
        tn_b = ((~pred) & (~y) & v_).sum().item()

        tp += tp_b
        fp += fp_b
        fn += fn_b
        tn += tn_b
        valid_pixels += v_.sum().item()

    # Derive MICRO metrics from global counts
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)

    acc = (tp + tn) / (valid_pixels + eps) if valid_pixels > 0 else 0.0

    return EpochStats(
        loss=sum_loss / max(1, n_batches),
        dice=float(dice),
        iou=float(iou),
        acc=float(acc),
        precision=float(precision),
        recall=float(recall),
    )


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    device: torch.device,
    resize_to: Optional[int],
    pos_weight: Optional[float],
    ignore_index: int = 255,
) -> Tuple["EpochStats", float]:
    """
    Validate the model for one epoch and compute metrics.

    This first selects a single global threshold on the provided
    validation loader (using pick_global_threshold_val). It then runs the
    model over the loader once using that fixed threshold, computing the
    average loss and accumulating true/false positives/negatives across all
    valid, non-ignored pixels and computes the metrics.

    Args:
        model: the segmentation model to evaluate.
        loader: DataLoader yielding batches with attributes rc, mask, valid_mask.
        device: torch.device to run the model on.
        resize_to: optional integer to resize predictions/targets for loss.
        pos_weight: optional positive class weight (unused here but kept for
            API compatibility).
        ignore_index: label value in targets to ignore during metrics.

    Returns:
        A tuple (EpochStats, float) where EpochStats contains loss and
        micro-metrics, and the float is the picked validation threshold's F1.
    """
    picked_thr, picked_f1 = pick_global_threshold_val(
        model=model,
        loader=loader,
        device=device,
        resize_to=None,           
        ignore_index=ignore_index,
    )
    LOGGER.info("Val global thr=%.3f (max F1=%.4f on val)", picked_thr, picked_f1)

    model.eval()

    eps = 1e-8
    sum_loss = 0.0
    n_batches = 0

    # Global counts (dataset-level)
    tp = fp = fn = tn = 0.0
    valid_pixels = 0.0

    for batch in loader:
        rc = batch.rc.to(device, non_blocking=True)
        mask = batch.mask.to(device, non_blocking=True)
        valid = batch.valid_mask.to(device, non_blocking=True)

        logits = model(rc)
        lo, ma, va = resize_for_loss(logits, mask, valid, resize_to)

        # loss (same as training/validation)
        loss = focal_loss_only(lo, ma, va, gamma=2.0, alpha=0.6, ignore_index=ignore_index)

        sum_loss += float(loss.item())
        n_batches += 1

        # Build predictions at picked_thr
        probs = torch.sigmoid(lo)                      # [B,1,H,W]
        pred = (probs >= float(picked_thr))            # bool [B,1,H,W]

        # Ensure mask is [B,1,H,W]
        if ma.ndim == 3:
            ma_ = ma.unsqueeze(1)
        elif ma.ndim == 4:
            ma_ = ma
        else:
            raise ValueError(f"Unexpected mask ndim={ma.ndim}")

        # Ensure valid is [B,1,H,W] boolean
        if va.ndim == 3:
            v_ = va.unsqueeze(1)
        elif va.ndim == 4:
            v_ = va
        else:
            raise ValueError(f"Unexpected valid ndim={va.ndim}")
        v_ = (v_ > 0.5)

        # Exclude ignore pixels
        v_ = v_ & (ma_ != ignore_index)

        # GT positives
        y = (ma_ == 1)

        # Accumulate counts on valid pixels only
        tp += (pred & y & v_).sum().item()
        fp += (pred & (~y) & v_).sum().item()
        fn += ((~pred) & y & v_).sum().item()
        tn += ((~pred) & (~y) & v_).sum().item()
        valid_pixels += v_.sum().item()

    # Dataset-level metrics
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    acc = (tp + tn) / (valid_pixels + eps) if valid_pixels > 0 else 0.0

    stats = EpochStats(
        loss=sum_loss / max(1, n_batches),
        dice=float(dice),
        iou=float(iou),
        acc=float(acc),
        precision=float(precision),
        recall=float(recall),
    )

    return stats, float(picked_thr)
