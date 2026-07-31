"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: test_rc_subsample_and_interpolate.py

Description:
    Unit tests for RCSubsampleAndInterpolate. These tests validate:
    • Input validation (missing 'rc', wrong shapes, invalid params)
    • Shape/dtype/device preservation and passthrough of extra metadata
    • Identity behavior for factors == 1 (no-op)
    • Downsample-then-upsample degrades details (small, controlled change)
    • Axis-specific blur (az_factor affects vertical detail, rg_factor horizontal)
    • Behavior across modes: down_mode ∈ {stride, avg}, up_mode ∈ {nearest, bilinear}
    • Batched input handling (N, H, W, 2)
    • Numerical stability (no NaN/Inf)

Run:
    pytest -q test_rc_subsample_and_interpolate.py
---------------------------------------------------------------------
History:
    - 2025-09-05:
        First version of unit tests for RCSubsampleAndInterpolate.
---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-09-05

© Copyright INDRA DEIMOS, 2025. All rights reserved.
---------------------------------------------------------------------
"""

from __future__ import annotations
import math
from typing import Tuple
import pytest
import torch
import torch.nn.functional as F
from sar_rc_augmentation import RCSubsampleAndInterpolate
import logging
from logging_setup import init_logging

# Configure logging once
init_logging(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


# =====================================================
# Helper Functions
# =====================================================

def make_rc(H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    Create synthetic I/Q data with broad spectral content:
    smooth structure + added noise to populate frequencies.
    Returns (H, W, 2) float32 tensor on device.
    """
    g = torch.Generator(device=device).manual_seed(42)
    u = torch.linspace(0, 1, steps=H, device=device)
    v = torch.linspace(0, 1, steps=W, device=device)
    U, V = torch.meshgrid(u, v, indexing="ij")
    phase = 2 * math.pi * (2.0 * U + 5.0 * V)
    amp = 1.0 + 0.25 * torch.sin(2 * math.pi * 3.0 * U) + 0.30 * torch.sin(2 * math.pi * 4.0 * V)
    z = amp * torch.exp(1j * phase)
    z = z + 0.10 * (torch.randn(H, W, generator=g, device=device) + 1j * torch.randn(H, W, generator=g, device=device))
    iq = torch.stack([z.real, z.imag], dim=-1).to(torch.float32)
    return iq

def grad_energy(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute simple finite-difference gradient energy along y (rows) and x (cols)
    over amplitude image derived from I/Q: amp = sqrt(I^2 + Q^2).
    Returns (Ey, Ex) scalar tensors.
    """
    amp = torch.sqrt(torch.clamp(img[..., 0] ** 2 + img[..., 1] ** 2, min=1e-12))
    dy = amp[1:, :] - amp[:-1, :]
    dx = amp[:, 1:] - amp[:, :-1]
    Ey = (dy ** 2).mean()
    Ex = (dx ** 2).mean()
    return Ey, Ex


# =====================================================
# Test Cases
# =====================================================

def test_invalid_params_raise():
    with pytest.raises(ValueError):
        RCSubsampleAndInterpolate(az_factor=0)
    with pytest.raises(ValueError):
        RCSubsampleAndInterpolate(rg_factor=0)
    with pytest.raises(ValueError):
        RCSubsampleAndInterpolate(down_mode="bad")
    with pytest.raises(ValueError):
        RCSubsampleAndInterpolate(up_mode="bad")

def test_missing_key_raises_keyerror():
    aug = RCSubsampleAndInterpolate()
    with pytest.raises(KeyError):
        aug({"not_rc": torch.zeros((8, 8, 2))})

def test_wrong_shape_raises_valueerror():
    aug = RCSubsampleAndInterpolate()
    with pytest.raises(ValueError):
        aug({"rc": torch.zeros((8, 8, 3))})
    with pytest.raises(ValueError):
        aug({"rc": torch.zeros((8, 8))})


def test_identity_when_factors_one():
    rc = make_rc(64, 96, device=torch.device("cpu"))
    aug = RCSubsampleAndInterpolate(az_factor=1, rg_factor=1, down_mode="avg", up_mode="bilinear")
    out = aug({"rc": rc})["rc"]
    assert out.shape == rc.shape
    # identity within small numerical tolerance
    mse = F.mse_loss(out, rc).item()
    assert mse < 1e-10, f"Expected near-identity when factors=1, got MSE={mse:.3e}"

def test_passthrough_and_device_cpu():
    rc = make_rc(32, 48, device=torch.device("cpu"))
    sample = {"rc": rc, "file": "foo.npy", "id": 123}
    out = RCSubsampleAndInterpolate(az_factor=1, rg_factor=2) (sample)
    assert out["rc"].device.type == "cpu"
    assert out["file"] == "foo.npy" and out["id"] == 123
    assert out["rc"].shape == rc.shape and out["rc"].dtype == torch.float32

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_cuda_preserved():
    rc = make_rc(32, 48, device=torch.device("cuda"))
    out = RCSubsampleAndInterpolate(az_factor=2, rg_factor=1) ({"rc": rc})
    assert out["rc"].device.type == "cuda"


@pytest.mark.parametrize("down_mode,up_mode", [("avg", "bilinear"), ("avg", "nearest"), ("stride", "bilinear"), ("stride", "nearest")])
def test_small_factors_change_data_slightly(down_mode: str, up_mode: str):
    """
    With factor 2 on one axis, output should differ (non-zero MSE) but not explode.
    """
    rc = make_rc(64, 96, device=torch.device("cpu"))
    aug = RCSubsampleAndInterpolate(az_factor=1, rg_factor=2, down_mode=down_mode, up_mode=up_mode)
    out = aug({"rc": rc})["rc"]

    mse = F.mse_loss(out, rc).item()
    assert mse > 0.0, "No change detected; expected some difference after down/up sampling."
    # Not too large for gentle settings (empirical loose cap)
    assert mse < 0.5, f"Change seems too large for gentle factors: MSE={mse:.3f}"

def test_axis_specific_blur_rg_affects_horizontal_detail_more():
    """
    rg_factor=2 should reduce horizontal gradients (Ex) more than vertical (Ey),
    since columns are subsampled then interpolated.
    """
    rc = make_rc(96, 128, device=torch.device("cpu"))
    Ey0, Ex0 = grad_energy(rc)
    out = RCSubsampleAndInterpolate(az_factor=1, rg_factor=2, down_mode="avg", up_mode="bilinear")({"rc": rc})["rc"]
    Ey1, Ex1 = grad_energy(out)
    # Horizontal gradient (x) should fall by at least ~5% relative
    assert Ex1 < 0.95 * Ex0, f"Expected noticeable reduction in horizontal detail, Ex1={Ex1:.3e}, Ex0={Ex0:.3e}"
    # Vertical gradient may change less; do not assert strong change.
    assert Ey1 <= Ey0 * 1.05  # allow small increase due to interpolation artifacts, but not large spike

def test_axis_specific_blur_az_affects_vertical_detail_more():
    """
    az_factor=2 should reduce vertical gradients (Ey) more than horizontal (Ex).
    """
    rc = make_rc(96, 128, device=torch.device("cpu"))
    Ey0, Ex0 = grad_energy(rc)
    out = RCSubsampleAndInterpolate(az_factor=2, rg_factor=1, down_mode="avg", up_mode="bilinear")({"rc": rc})["rc"]
    Ey1, Ex1 = grad_energy(out)
    assert Ey1 < 0.95 * Ey0, f"Expected noticeable reduction in vertical detail, Ey1={Ey1:.3e}, Ey0={Ey0:.3e}"
    assert Ex1 <= Ex0 * 1.05


def test_batch_input_handling():
    H, W, N = 64, 96, 4
    rc = make_rc(H, W, device=torch.device("cpu"))
    batch = torch.stack([rc for _ in range(N)], dim=0)  # (N,H,W,2)
    out = RCSubsampleAndInterpolate(az_factor=2, rg_factor=1, down_mode="avg", up_mode="bilinear")({"rc": batch})["rc"]
    assert out.shape == batch.shape
    # Per-item outputs should not be all identical in content to input
    mse0 = F.mse_loss(out[0], batch[0]).item()
    assert mse0 > 0.0


def test_no_nan_inf_in_output():
    rc = make_rc(64, 96, device=torch.device("cpu"))
    out = RCSubsampleAndInterpolate(az_factor=2, rg_factor=2, down_mode="avg", up_mode="bilinear")({"rc": rc})["rc"]
    assert torch.isfinite(out).all()
