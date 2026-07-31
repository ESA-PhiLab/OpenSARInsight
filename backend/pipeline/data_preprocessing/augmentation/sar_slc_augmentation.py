"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: sar_slc_augmentation.py

Description:
    PyTorch-compatible data augmentation transforms for SAR data.

    SLC amplitude:
        • Random translation (range/azimuth shifts with padding)
        • Random cropping (random or center)
        • Resize back to a fixed size (default 512x512) via bilinear interpolation
        • Multiplicative speckle noise (Gamma-distributed)


    All transforms accept and return a dict-like sample (e.g., {"slc": ...}) and pass through untouched keys, making them compatible with
    torchvision.transforms.Compose.

History:
    - 2025-08-21:
        First version with slc translate/crop/resize augmentation.
    - 2025-08-27:
        More SLC augmentations were added.

---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-08-21

© Copyright INDRA DEIMOS, 2025. All rights reserved.
---------------------------------------------------------------------
"""

from __future__ import annotations
from typing import Any, Dict, Mapping, Optional, Tuple
import logging
import torch
from torch import Tensor
import torch.nn.functional as F
from logging_setup import init_logging

# Configure logging once
init_logging(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

__all__ = [
    "SLCRandomTranslate",
    "SLCRandomCrop",
    "SLCResize",
    "SLCSpeckleNoise",
]

EPSILON: float = 1e-12


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _to_float_tensor(arr: Any) -> Tensor:
    """
        Convert input to torch.float32 tensor.

        Args:
            arr (Any): Input array (torch.Tensor or numpy.ndarray).
    """
    if isinstance(arr, Tensor):
        return arr.float()

    try:
        import numpy as np  # delayed import
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise TypeError("Input must be torch.Tensor or numpy.ndarray") from exc

    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr).float()

    raise TypeError("Input must be torch.Tensor or numpy.ndarray")


def _ensure_2d(name: str, x: Tensor) -> None:
    """
        Ensure an input tensor is 2D (H, W).

        Args:
            name (str): Field name for error messages.
            x (Tensor): Input tensor to validate.
    """
    if x.ndim != 2:
        raise ValueError(f'"{name}" must have shape (H, W); got {tuple(x.shape)}')


# ---------------------------------------------------------------------
# SLC amplitude augmentations
# ---------------------------------------------------------------------
class SLCRandomTranslate:
    """
    A class to translate SLC amplitude by random integer offsets
    in azimuth (rows) and range (cols), padding new areas.

    Attributes:
        max_shift_px (Tuple[int, int]): Maximum absolute shift (dy, dx) in pixels.
        mode (str): Padding mode: "constant", "reflect", or "replicate".
        fill (float): Constant value used when mode="constant".
    """

    def __init__(
        self,
        *,
        max_shift_px: Tuple[int, int] = (8, 8),
        mode: str = "reflect",
        fill: float = 0.0,
    ) -> None:
        """
            Initialise a new SLCRandomTranslate instance.

            Args:
                max_shift_px (Tuple[int, int]): Maximum absolute shift (dy, dx).
                mode (str): "constant", "reflect", or "replicate".
                fill (float): Fill value when mode="constant".

            Raises:
                ValueError: If an unsupported padding mode is provided.
        """
        if mode not in {"constant", "reflect", "replicate"}:
            raise ValueError('mode must be one of {"constant", "reflect", "replicate"}')
        self.max_shift_px = (int(max_shift_px[0]), int(max_shift_px[1]))
        self.mode = mode
        self.fill = float(fill)

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        """
            Translate the "slc" field by random (dy, dx) within max_shift_px.

            Args:
                sample (Mapping[str, Any]): Input dict with key "slc" (H, W).

            Returns:
                Dict[str, Any]: Output dict with translated "slc".
        """
        if "slc" not in sample:
            return dict(sample)

        slc = _to_float_tensor(sample["slc"])
        _ensure_2d("slc", slc)

        H, W = slc.shape
        max_dy, max_dx = self.max_shift_px
        dy = int(torch.randint(-max_dy, max_dy + 1, (1,)).item()) if max_dy > 0 else 0
        dx = int(torch.randint(-max_dx, max_dx + 1, (1,)).item()) if max_dx > 0 else 0
        if dy == 0 and dx == 0:
            return dict(sample)

        pad_top, pad_bottom = max(dy, 0), max(-dy, 0)
        pad_left, pad_right = max(dx, 0), max(-dx, 0)

        if self.mode == "constant":
            padded = F.pad(slc, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=self.fill)
        elif self.mode == "reflect":
            padded = F.pad(slc, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
        else:  # replicate
            padded = F.pad(slc, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

        y0 = pad_top - dy
        x0 = pad_left - dx
        slc_out = padded[y0:y0 + H, x0:x0 + W]

        out = dict(sample)
        out["slc"] = slc_out
        LOGGER.info("SLCRandomTranslate | dy=%d dx=%d mode=%s", dy, dx, self.mode)
        return out


class SLCRandomCrop:
    """
    A class to crop SLC amplitude to a target size, randomly or centered.

    Attributes:
        size (Tuple[int, int]): Output crop size (out_h, out_w).
        random (bool): If True, random crop; else center crop.
    """

    def __init__(self, *, size: Tuple[int, int], random: bool = True) -> None:
        """
            Initialise a new SLCRandomCrop instance.

            Args:
                size (Tuple[int, int]): Output crop size (out_h, out_w).
                random (bool): If True, choose a random crop; else center crop.

            Raises:
                ValueError: If requested size is invalid (<=0).
        """
        oh, ow = size
        if oh <= 0 or ow <= 0:
            raise ValueError("size must be (out_h>0, out_w>0)")
        self.size = (int(oh), int(ow))
        self.random = bool(random)

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        """
            Crop the "slc" field to the configured size.

            Args:
                sample (Mapping[str, Any]): Input dict with key "slc" (H, W).

            Returns:
                Dict[str, Any]: Output dict with cropped "slc".

            Raises:
                ValueError: If the input is smaller than the requested crop.
        """
        if "slc" not in sample:
            return dict(sample)

        slc = _to_float_tensor(sample["slc"])
        _ensure_2d("slc", slc)

        H, W = slc.shape
        oh, ow = self.size
        if oh > H or ow > W:
            raise ValueError(f"crop size {self.size} exceeds input {(H, W)}")

        if self.random:
            y0 = int(torch.randint(0, H - oh + 1, (1,)).item())
            x0 = int(torch.randint(0, W - ow + 1, (1,)).item())
        else:
            y0 = (H - oh) // 2
            x0 = (W - ow) // 2

        slc_out = slc[y0:y0 + oh, x0:x0 + ow]

        out = dict(sample)
        out["slc"] = slc_out
        LOGGER.info("SLCRandomCrop | y0=%d x0=%d size=(%d,%d)", y0, x0, oh, ow)
        return out


class SLCResize:
    """
    A class to resize SLC amplitude images to a fixed size using interpolation.

    Attributes:
        size (Tuple[int, int]): Output size (out_h, out_w), default (512, 512).
        mode (str): Interpolation mode ("bilinear", "nearest", "bicubic").
    """

    def __init__(self, *, size: Tuple[int, int] = (512, 512), mode: str = "bilinear") -> None:
        """
            Initialise a new SLCResize instance.

            Args:
                size (Tuple[int, int]): Desired output size (out_h, out_w).
                mode (str): Interpolation mode ("bilinear", "nearest", "bicubic").

            Raises:
                ValueError: If the interpolation mode is unsupported.
        """
        if mode not in {"bilinear", "nearest", "bicubic"}:
            raise ValueError('mode must be one of {"bilinear", "nearest", "bicubic"}')
        self.size = (int(size[0]), int(size[1]))
        self.mode = mode

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        """
            Resize the "slc" field to the configured size via interpolation.

            Args:
                sample (Mapping[str, Any]): Input dict with key "slc" (H, W).

            Returns:
                Dict[str, Any]: Output dict with resized "slc".
        """
        if "slc" not in sample:
            return dict(sample)

        slc = _to_float_tensor(sample["slc"])
        _ensure_2d("slc", slc)

        slc_4d = slc.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        slc_resized = F.interpolate(slc_4d, size=self.size, mode=self.mode, align_corners=False if self.mode != "nearest" else None)
        slc_out = slc_resized.squeeze(0).squeeze(0)

        out = dict(sample)
        out["slc"] = slc_out
        LOGGER.info("SLCResize | size=%s mode=%s", self.size, self.mode)
        return out


class SLCSpeckleNoise:
    """
    A class to add multiplicative speckle noise to SLC amplitude.

    Attributes:
        looks (float): Equivalent number of looks L (>0). Variance of U is 1/L.
    """

    def __init__(self, *, looks: float = 1.0) -> None:
        """
            Initialise a new SLCSpeckleNoise instance.

            Args:
                looks (float): Number of looks L (>0).

            Raises:
                ValueError: If looks <= 0.
        """
        if looks <= 0:
            raise ValueError("looks must be > 0")
        self.looks = float(looks)

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        """
            Apply multiplicative speckle to the "slc" field (amplitude domain).

            Args:
                sample (Mapping[str, Any]): Input dict with key "slc" (H, W), non-negative.

            Returns:
                Dict[str, Any]: Output dict with noisy "slc".
        """
        if "slc" not in sample:
            return dict(sample)

        slc = _to_float_tensor(sample["slc"])
        _ensure_2d("slc", slc)

        # U ~ Gamma(L, scale=1/L) => E[U]=1, Var[U]=1/L; amplitude uses sqrt(U)
        L = torch.tensor(self.looks, dtype=torch.float32, device=slc.device)
        U = torch.distributions.Gamma(concentration=L, rate=L).sample(slc.shape).to(slc.device)
        mult = torch.sqrt(U).clamp_min(0.0)
        slc_out = slc * mult

        out = dict(sample)
        out["slc"] = slc_out
        LOGGER.info("SLCSpeckleNoise | looks=%.3f", self.looks)
        return out

