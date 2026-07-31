"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: test_rc_amplitude_phase_perturb.py

Description:
    Unit tests for RCAmplitudePhasePerturb. These tests validate:
    • Input validation (shape, missing keys, dtype)
    • Preservation of shape, dtype, and extra metadata
    • Correct amplitude scaling bounds
    • Correct phase perturbation bounds
    • Behavior for per-pixel vs per-patch perturbations
    • Batched input handling
    • Device preservation (CPU & CUDA)
    • Randomness differences across seeds

Run:
    pytest -q test_rc_amplitude_phase_perturb.py
---------------------------------------------------------------------
History:
    - 2025-09-04:
        This is the first version of the core unit test for the RCAmplitudePhasePerturb

---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-09-04

© Copyright INDRA DEIMOS, 2025. All rights reserved.

"""

from __future__ import annotations
import math
from typing import Tuple
import numpy as np
import pytest
import torch
from sar_rc_augmentation import RCAmplitudePhasePerturb, EPSILON


# =====================================================
# Helper Functions
# =====================================================

def amplitude(iq: torch.Tensor) -> torch.Tensor:
    """
    Compute amplitude from I/Q channels.

    Args:
        iq (torch.Tensor): Input tensor of shape (H, W, 2) or (N, H, W, 2).

    Returns:
        torch.Tensor: Amplitude tensor with shape (H, W) or (N, H, W).
    """
    return torch.sqrt(torch.clamp(iq[..., 0] ** 2 + iq[..., 1] ** 2, min=EPSILON))


def phase_wrapped(iq: torch.Tensor) -> torch.Tensor:
    """
    Compute wrapped phase from I/Q channels in radians.

    Args:
        iq (torch.Tensor): Input tensor of shape (H, W, 2) or (N, H, W, 2).

    Returns:
        torch.Tensor: Phase tensor in the range [-π, π], same leading shape as input.
    """
    return torch.atan2(iq[..., 1], iq[..., 0])


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """
    Wrap angles to the range [-π, π].

    Args:
        x (torch.Tensor): Input tensor of unwrapped phase values.

    Returns:
        torch.Tensor: Phase values wrapped into [-π, π].
    """
    return (x + math.pi) % (2 * math.pi) - math.pi


def make_rc(shape: Tuple[int, int, int] = (32, 48, 2), device: torch.device | None = None) -> torch.Tensor:
    """
    Generate a random synthetic RC tensor for testing.

    Args:
        shape (tuple): Shape of RC data, default (32, 48, 2).
        device (torch.device, optional): Device to create tensor on.

    Returns:
        torch.Tensor: Random RC tensor with both I and Q components.
    """
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(123)
    rc = torch.randn(shape, device=device, dtype=torch.float32)
    rc += 0.05 * torch.sign(rc)  # Avoid pure zeros
    return rc


# =====================================================
# Test Cases
# =====================================================

def test_missing_key_raises_keyerror():
    """
    Verify that missing 'rc' key in the sample raises KeyError.
    """
    aug = RCAmplitudePhasePerturb()
    with pytest.raises(KeyError):
        aug({"not_rc": torch.zeros((8, 8, 2))})


def test_wrong_shape_raises_valueerror():
    """
    Verify that invalid 'rc' shapes raise ValueError.
    """
    aug = RCAmplitudePhasePerturb()
    # Wrong last dim
    with pytest.raises(ValueError):
        aug({"rc": torch.zeros((8, 8, 3))})
    # Not 3D
    with pytest.raises(ValueError):
        aug({"rc": torch.zeros((8, 8))})


@pytest.mark.parametrize("per_pixel", [True, False])
def test_shape_dtype_and_passthrough(per_pixel: bool):
    """
    Verify that shape, dtype, and extra metadata are preserved after augmentation.
    """
    rc = make_rc((16, 20, 2))
    sample = {"rc": rc.clone(), "meta": "keep_me", "id": 42}

    aug = RCAmplitudePhasePerturb(
        amp_scale_range=(0.8, 1.2),
        max_phase_shift=0.3,
        per_pixel=per_pixel,
    )
    out = aug(sample)

    # Metadata keys must remain untouched
    assert "rc" in out and "meta" in out and "id" in out
    assert out["meta"] == "keep_me"
    assert out["id"] == 42

    # Shape and dtype preserved
    assert isinstance(out["rc"], torch.Tensor)
    assert out["rc"].dtype == torch.float32
    assert out["rc"].shape == rc.shape


def test_device_is_preserved_cpu():
    """
    Ensure augmentation preserves CPU device.
    """
    rc = make_rc((12, 18, 2), device=torch.device("cpu"))
    aug = RCAmplitudePhasePerturb()
    out = aug({"rc": rc})
    assert out["rc"].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_is_preserved_cuda():
    """
    Ensure augmentation preserves CUDA device when available.
    """
    rc = make_rc((12, 18, 2), device=torch.device("cuda"))
    aug = RCAmplitudePhasePerturb()
    out = aug({"rc": rc})
    assert out["rc"].device.type == "cuda"


@pytest.mark.parametrize("per_pixel", [True, False])
def test_scaling_and_phase_bounds(per_pixel: bool):
    """
    Check that amplitude scaling ratios and phase offsets
    remain within the configured bounds.
    """
    a_min, a_max = 0.8, 1.2
    max_dphi = 0.3
    tol = 1e-3

    rc = make_rc((24, 32, 2))
    amp0 = amplitude(rc)
    ph0 = phase_wrapped(rc)

    out = RCAmplitudePhasePerturb(
        amp_scale_range=(a_min, a_max),
        max_phase_shift=max_dphi,
        per_pixel=per_pixel,
    )({"rc": rc})["rc"]

    amp1 = amplitude(out)
    ph1 = phase_wrapped(out)

    # Check amplitude ratios within range
    ratio = amp1 / torch.clamp(amp0, min=EPSILON)
    assert torch.all(ratio >= (a_min - tol))
    assert torch.all(ratio <= (a_max + tol))

    # Check phase differences within limit
    dphi = wrap_to_pi(ph1 - ph0).abs()
    assert torch.all(dphi <= (max_dphi + tol))


def test_per_patch_is_constant_within_sample():
    """
    If per_pixel=False, alpha and delta_phi must be constant across all pixels.
    """
    a_min, a_max = 0.9, 1.1
    max_dphi = 0.25

    rc = make_rc((30, 22, 2))
    amp0 = amplitude(rc)
    ph0 = phase_wrapped(rc)

    out = RCAmplitudePhasePerturb(
        amp_scale_range=(a_min, a_max),
        max_phase_shift=max_dphi,
        per_pixel=False,
    )({"rc": rc})["rc"]

    amp1 = amplitude(out)
    ph1 = phase_wrapped(out)

    # Expect near-constant scaling and phase offset
    ratio = (amp1 / torch.clamp(amp0, min=EPSILON)).flatten()
    dphi = wrap_to_pi(ph1 - ph0).flatten()
    assert torch.var(ratio) < 1e-6
    assert torch.var(dphi) < 1e-6


def test_batched_input_supported_and_bounds_hold():
    """
    Verify that batched inputs (N, H, W, 2) are processed correctly
    and amplitude & phase constraints are respected.
    """
    N = 3
    H, W = 16, 16
    rc = torch.stack([make_rc((H, W, 2)) for _ in range(N)], dim=0)

    a_min, a_max = 0.85, 1.15
    max_dphi = 0.2
    tol = 1e-3

    amp0 = amplitude(rc)
    ph0 = phase_wrapped(rc)

    out = RCAmplitudePhasePerturb(
        amp_scale_range=(a_min, a_max),
        max_phase_shift=max_dphi,
        per_pixel=True,
    )({"rc": rc})["rc"]

    # Check output shape matches
    assert out.shape == rc.shape

    amp1 = amplitude(out)
    ph1 = phase_wrapped(out)

    # Verify constraints per batch
    ratio = amp1 / torch.clamp(amp0, min=EPSILON)
    dphi = wrap_to_pi(ph1 - ph0).abs()
    assert torch.all(ratio >= (a_min - tol))
    assert torch.all(ratio <= (a_max + tol))
    assert torch.all(dphi <= (max_dphi + tol))


def test_random_seed_changes_output():
    """
    Sanity check that different random seeds produce different perturbations.
    """
    rc = make_rc((20, 20, 2))

    torch.manual_seed(1)
    out1 = RCAmplitudePhasePerturb(per_pixel=True)({"rc": rc})["rc"]

    torch.manual_seed(2)
    out2 = RCAmplitudePhasePerturb(per_pixel=True)({"rc": rc})["rc"]

    # Different seeds should (very likely) produce different outputs
    assert not torch.allclose(out1, out2)
