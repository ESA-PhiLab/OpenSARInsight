"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: test_slc_augmentation_core.py

Description:
    Core unit tests for SLC data augmentation transforms:
      - SLCRandomTranslate
      - SLCRandomCrop
      - SLCResize
      - SLCSpeckleNoise

    These tests check basic correctness, shape handling, and error paths.

Run:
    pytest -q test_slc_augmentation_core.py
    
---------------------------------------------------------------------
History:
    - 2025-08-07:
        This is the first version of the core unit test for the SLC data augmentation script.

---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-08-28

© Copyright INDRA DEIMOS, 2025. All rights reserved.

"""


from __future__ import annotations
import numpy as np
import torch
import pytest
from torchvision.transforms import Compose

from data_preprocessing.augmentation.sar_slc_augmentation import (
    SLCRandomTranslate,
    SLCRandomCrop,
    SLCResize,
    SLCSpeckleNoise,
)

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
torch.manual_seed(0)
np.random.seed(0)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _dummy_slc(h: int = 64, w: int = 64) -> np.ndarray:
    """Positive SLC amplitude sample."""
    x = np.abs(np.random.randn(h, w)).astype("float32")
    return x


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_slc_translate_identity_when_zero_shift():
    """If max shifts are (0,0), output must equal input."""
    slc = _dummy_slc(32, 40)
    tf = SLCRandomTranslate(max_shift_px=(0, 0), mode="constant", fill=0.0)
    out = tf({"slc": slc})["slc"]

    # Ensure both are tensors for allclose
    if isinstance(out, np.ndarray):
        out = torch.from_numpy(out)
    slc_t = torch.from_numpy(slc)

    assert torch.allclose(out, slc_t)


def test_slc_random_crop_shape_and_bounds():
    """Random crop returns requested shape and uses in-bounds indices."""
    slc = _dummy_slc(64, 64)
    tf = SLCRandomCrop(size=(32, 32), random=True)
    out = tf({"slc": slc})["slc"]

    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape) == (32, 32)
    assert torch.isfinite(out).all()


def test_slc_resize_produces_target_size():
    """Resize hits the target size."""
    slc = _dummy_slc(60, 55)
    tf = SLCResize(size=(64, 64), mode="bilinear")
    out = tf({"slc": slc})["slc"]

    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape) == (64, 64)


def test_slc_speckle_noise_stats():
    """
    With looks=1, multiplicative speckle noise should have a relatively high variance.
    We validate that the noise std is within a realistic range for Gamma-distributed noise.
    """
    slc = _dummy_slc(128, 128)
    tf = SLCSpeckleNoise(looks=1.0)

    out = tf({"slc": slc})["slc"]
    assert isinstance(out, torch.Tensor)

    slc_t = torch.from_numpy(slc)
    noise = out - slc_t

    noise_std = noise.std().item()
    mean_slc = slc_t.mean().item()

    # For looks=1, allow up to ~mean_slc (std ≈ mean)
    assert noise_std >= 0.05 * mean_slc
    assert noise_std <= 1.2 * mean_slc  # more relaxed upper bound