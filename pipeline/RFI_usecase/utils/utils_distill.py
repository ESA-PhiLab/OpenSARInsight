"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: utils_distill.py

Key properties
--------------
- Provides utility functions for knowledge distillation between
  teacher and student segmentation models.
- Supports logit-based distillation using binary BCE or MSE losses.
- Supports feature-based distillation on intermediate model feature
  maps using configurable layer taps.
- Includes feature alignment utilities for resizing and channel
  projection before feature comparison.
- Supports multiple feature matching losses, including L2, L1,
  cosine, and attention transfer.

Recommended usage
-----------------
- Use this module in teacher-student training pipelines for model
  compression or efficient student training.
- Useful when transferring knowledge from a larger U-Net model to a
  smaller or more efficient variant.

History:
    - 2026-02-17:
        First version of distillation utility functions.
    - 2026-03-05:    
        Final version is ready.

---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2026-02-17

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
from torch import nn


# ----------------------------- Logit KD -----------------------------
@torch.no_grad()
def forward_teacher(teacher: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """
    Run the teacher model in eval mode on imgs and return its outputs.

    This helper ensures the teacher is set to evaluation mode and
    performs a forward pass without tracking gradients.
    """
    teacher.eval()
    return teacher(imgs)


def kd_binary_bce(student_logits, teacher_logits, valid_mask=None, T: float = 1.0):
    """
    Compute knowledge distillation loss between student and teacher logits

    Uses temperature-scaled binary cross-entropy with logits. If provided,
    valid_mask is used to ignore invalid spatial locations.
    """

    s = student_logits / T
    with torch.no_grad():
        t_prob = torch.sigmoid(teacher_logits / T)

    kd = F.binary_cross_entropy_with_logits(s, t_prob, reduction="none")

    if valid_mask is not None:
        vm = valid_mask
        while vm.ndim < kd.ndim:
            vm = vm.unsqueeze(1)
        vm = vm.float()
        kd = (kd * vm).sum() / vm.expand_as(kd).sum().clamp_min(1.0)
    else:
        kd = kd.mean()

    return kd * (T * T)


def kd_logit_mse(student_logits, teacher_logits, valid_mask=None, T: float = 1.0):
    """
    Compute mean-squared error between student and teacher logits with optional
    temperature scaling and spatial masking.

    Args:
        student_logits: Student model raw logits (B,C,H,W or similar).
        teacher_logits: Teacher model raw logits (same shape as student).
        valid_mask: Optional mask to exclude invalid spatial locations.
        T: Temperature scalar applied to logits before computing MSE.

    Returns:
        Scalar MSE loss multiplied by T^2 to account for temperature scaling.
    """

    s = student_logits / T
    t = teacher_logits / T
    kd = (s - t).pow(2)
    if valid_mask is not None:
        vm = valid_mask
        while vm.ndim < kd.ndim:
            vm = vm.unsqueeze(1)
        kd = kd * vm
    return kd.mean() * (T * T)


# ----------------------------- Feature KD -----------------------------
def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Resize tensor x spatially to match the height and width of ref.

    If x already has the same spatial dimensions as ref, it is returned
    unchanged. Otherwise x is resized using bilinear interpolation.
    """
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

def feat_projector(ch_in: int, ch_out: int, kind: str = "1x1") -> nn.Module:
    """
    Create a projector module to map features from ch_in to ch_out.

    Returns an identity when channels match, a small MLP-style projector
    when kind=='mlp', or a 1x1 convolution otherwise.
    """
    if ch_in == ch_out:
        return nn.Identity()
    if kind == "mlp":
        mid = max(ch_in // 2, ch_out)
        return nn.Sequential(
            nn.Conv2d(ch_in, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch_out, 1, bias=False),
        )
    return nn.Conv2d(ch_in, ch_out, 1, bias=False)  # 1x1

def loss_l2(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compute the L2 loss (mean squared error) between two tensors.

    Args:
        s: Student tensor.
        t: Teacher tensor.

    Returns:
        Mean squared difference between s and t.
    """
    return (s - t).pow(2).mean()

def loss_l1(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compute the L1 loss (mean absolute error) between two tensors.

    Args:
        s: Student tensor.
        t: Teacher tensor.

    Returns:
        Mean absolute difference between s and t.
    """
    return (s - t).abs().mean()

def loss_cosine(s: torch.Tensor, t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the cosine similarity loss between two tensors.

    Args:
        s: Student tensor.
        t: Teacher tensor.
        eps: Small value to avoid division by zero.

    Returns:
        Mean cosine similarity loss between s and t.
    """
    s = s.flatten(2)   # B,C,HW
    t = t.flatten(2)
    s = F.normalize(s, dim=2, eps=eps)
    t = F.normalize(t, dim=2, eps=eps)
    return (1 - (s * t).sum(dim=2)).mean()

def loss_attention_transfer(s: torch.Tensor, t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the attention transfer loss between two tensors.

    Args:
        s: Student tensor.
        t: Teacher tensor.
        eps: Small value to avoid division by zero.

    Returns:
        Mean attention transfer loss between s and t.
    """
    # Zagoruyko & Komodakis (2017)
    def A(x):
        a = x.pow(2).sum(dim=1, keepdim=True)   # B,1,H,W
        a = a / (a.flatten(1).sum(dim=1, keepdim=True).unsqueeze(-1) + eps)
        return a
    return (A(s) - A(t)).pow(2).mean()


class FeatureTaps:
    """
    This collects intermediate feature maps from UNet modules via
    forward hooks for use in feature-based knowledge distillation.

    Register forward hooks on named UNet modules to collect feature maps.
    Available names (for the provided UNet): enc1, enc2, enc3, enc4, bottleneck, dec3, dec2
    """
    def __init__(self, model: nn.Module, tap_names: List[str]):
        self.model = model
        self.tap_names = tap_names
        self.features: Dict[str, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

        name_to_module = {
            "enc1": getattr(model, "inc", None),
            "enc2": getattr(model, "down_blocks", [None])[0] if hasattr(model, "down_blocks") and len(model.down_blocks) > 0 else None,
            "enc3": getattr(model, "down_blocks", [None, None])[1] if hasattr(model, "down_blocks") and len(model.down_blocks) > 1 else None,
            "enc4": getattr(model, "down_blocks", [None, None, None])[2] if hasattr(model, "down_blocks") and len(model.down_blocks) > 2 else None,
            "bottleneck": model.down_blocks[-1] if hasattr(model, "down_blocks") and len(model.down_blocks) > 0 else None,
            "dec3": model.up_blocks[-1] if hasattr(model, "up_blocks") and len(model.up_blocks) > 0 else None,
            "dec2": model.up_blocks[-2] if hasattr(model, "up_blocks") and len(model.up_blocks) > 1 else None,
        }

        for n in tap_names:
            m = name_to_module.get(n, None)
            if m is None:
                raise KeyError(f"Tap '{n}' not found. Available: {list(name_to_module.keys())}")
            self.handles.append(m.register_forward_hook(self._make_hook(n)))

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            self.features[name] = out
        return hook

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.features.clear()


def compute_feature_kd(
    student: nn.Module,
    teacher: nn.Module,
    imgs: torch.Tensor,
    taps_cfg: Dict,
) -> Tuple[torch.Tensor, List[float]]:
    """
    Compute feature-based knowledge distillation loss between a student
    and teacher by matching intermediate feature maps at configured taps.

    Args:
        student: Student model (nn.Module).
        teacher: Teacher model (nn.Module).
        imgs: Input batch of images (torch.Tensor).
        taps_cfg: Configuration dict with keys like "taps", "proj",
                  "norm", "per_layer_weights", "weight", and
                  "detach_teacher".

    Returns:
        A tuple (total_loss, per_layer_losses) where total_loss is a
        scalar tensor and per_layer_losses is a list of floats.
    """
    tap_names = taps_cfg["taps"]
    proj_kind = taps_cfg.get("proj", "1x1")
    norm      = taps_cfg.get("norm", "l2")      # l2 | l1 | cosine | at
    per_w     = taps_cfg.get("per_layer_weights", None)
    global_w  = float(taps_cfg.get("weight", 1.0))
    detach_teacher = bool(taps_cfg.get("detach_teacher", True))

    s_taps = FeatureTaps(student, tap_names)
    t_taps = FeatureTaps(teacher, tap_names)

    with torch.no_grad() if detach_teacher else torch.enable_grad():
        _ = teacher(imgs)     # fills t_taps.features
    _ = student(imgs)         # fills s_taps.features

    losses = []
    proj_cache: Dict[tuple, nn.Module] = {}

    def get_proj(sC, tC, device):
        key = (sC, tC, proj_kind)
        if key not in proj_cache:
            proj_cache[key] = feat_projector(sC, tC, kind=proj_kind).to(device)
        return proj_cache[key]

    for i, name in enumerate(tap_names):
        sf = s_taps.features[name]   # B,Cs,Hs,Ws
        tf = t_taps.features[name]   # B,Ct,Ht,Wt
        sf = _resize_like(sf, tf)
        proj = get_proj(sf.shape[1], tf.shape[1], sf.device)
        sf = proj(sf)

        if   norm == "l2":     l = loss_l2(sf, tf)
        elif norm == "l1":     l = loss_l1(sf, tf)
        elif norm == "cosine": l = loss_cosine(sf, tf)
        elif norm == "at":     l = loss_attention_transfer(sf, tf)
        else: raise ValueError(f"Unknown feature norm: {norm}")

        w = per_w[i] if per_w is not None else 1.0
        losses.append(w * l)

    s_taps.close(); t_taps.close()

    total = global_w * sum(losses) if len(losses) else imgs.new_tensor(0.0)
    return total, [float(x.item()) for x in losses]
