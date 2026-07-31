"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: test_rc_narrowband_spectral_dropout.py

Description:
    Unit tests for RCNarrowbandSpectralDropout. These tests validate:
    • Input validation (missing 'rc', wrong shapes)
    • Preservation of shape, dtype, device, and passthrough metadata
    • Correct behavior for range- and azimuth-band dropout (hard & soft)
    • Multiple disjoint band handling
    • Batched input handling (N, H, W, 2)
    • Soft-taper ramp directions and monotonicity (1→0 enter, 0 core, 0→1 exit)
    • Numerical stability (no NaN/Inf)

Run:
    pytest -q test_rc_narrowband_spectral_dropout.py
---------------------------------------------------------------------
History:
    - 2025-09-05:
        First version of the unit tests for RCNarrowbandSpectralDropout.

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
from sar_rc_augmentation import RCNarrowbandSpectralDropout, EPSILON


# =====================================================
# Helper Functions
# =====================================================

def fft2c(z: torch.Tensor) -> torch.Tensor:
    """Centered 2D FFT (replicates module behavior)."""
    return torch.fft.fftshift(torch.fft.fft2(z, norm="backward"), dim=(-2, -1))

def ifft2c(Z: torch.Tensor) -> torch.Tensor:
    """Centered 2D IFFT (replicates module behavior)."""
    return torch.fft.ifft2(torch.fft.ifftshift(Z, dim=(-2, -1)), norm="backward")

def iq_to_complex(iq: torch.Tensor) -> torch.Tensor:
    """(H,W,2) or (N,H,W,2) -> complex (...,H,W)."""
    return torch.complex(iq[..., 0], iq[..., 1])

def complex_to_iq(z: torch.Tensor) -> torch.Tensor:
    """complex (...,H,W) -> (H,W,2) or (N,H,W,2)."""
    return torch.stack([z.real, z.imag], dim=-1)

def make_rc(H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    Create non-degenerate synthetic I/Q with broad spectral content.
    Uses sinusoidal structure + noise to ensure energy across frequencies.
    """
    g = torch.Generator(device=device).manual_seed(123)
    u = torch.linspace(0, 1, steps=H, device=device)
    v = torch.linspace(0, 1, steps=W, device=device)
    U, V = torch.meshgrid(u, v, indexing="ij")
    phase = 2 * math.pi * (2.5 * U + 5.0 * V)
    amp = 1.0 + 0.3 * torch.sin(2 * math.pi * 4.0 * V) + 0.2 * torch.sin(2 * math.pi * 3.0 * U)
    z = amp * torch.exp(1j * phase)
    z = z + 0.15 * (
        torch.randn(H, W, generator=g, device=device)
        + 1j * torch.randn(H, W, generator=g, device=device)
    )
    return complex_to_iq(z.to(torch.complex64)).to(torch.float32)

def est_mask_from_ratio(rc_before: torch.Tensor, rc_after: torch.Tensor) -> torch.Tensor:
    """
    Estimate frequency mask by ratio of magnitudes in FFT domain: |Z_after|/|Z_before|.
    Returns (H,W) real tensor ~ [0..1], robust to small numerical issues.
    """
    z0 = iq_to_complex(rc_before)
    z1 = iq_to_complex(rc_after)
    Z0 = fft2c(z0)
    Z1 = fft2c(z1)
    mag0 = torch.abs(Z0).clamp_min(1e-9)
    mag1 = torch.abs(Z1)
    ratio = (mag1 / mag0).real
    return ratio

def band_to_idx(length: int, frac0: float, frac1: float) -> tuple[int, int]:
    a = int(round(frac0 * length))
    b = int(round(frac1 * length))
    a = max(0, min(length, a))
    b = max(0, min(length, b))
    if b < a:
        a, b = b, a
    return a, b


# =====================================================
# Test Cases
# =====================================================
def test_missing_key_raises_keyerror():
    aug = RCNarrowbandSpectralDropout()
    with pytest.raises(KeyError):
        aug({"not_rc": torch.zeros((8, 8, 2))})

def test_wrong_shape_raises_valueerror():
    aug = RCNarrowbandSpectralDropout()
    with pytest.raises(ValueError):
        aug({"rc": torch.zeros((8, 8, 3))})
    with pytest.raises(ValueError):
        aug({"rc": torch.zeros((8, 8))})


def test_device_and_passthrough_cpu():
    rc = make_rc(64, 128, device=torch.device("cpu"))
    sample = {"rc": rc, "meta": "keep", "id": 7}
    out = RCNarrowbandSpectralDropout(bands_range=[(0.48, 0.52)], soft=True, edge_taper=0.1)(sample)
    assert out["rc"].device.type == "cpu"
    assert out["meta"] == "keep" and out["id"] == 7
    assert out["rc"].shape == rc.shape and out["rc"].dtype == torch.float32

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_cuda():
    rc = make_rc(64, 128, device=torch.device("cuda"))
    out = RCNarrowbandSpectralDropout(bands_range=[(0.48, 0.52)], soft=True, edge_taper=0.1)({"rc": rc})
    assert out["rc"].device.type == "cuda"


@pytest.mark.parametrize("band", [(0.48, 0.52), (0.20, 0.22)])
def test_hard_range_notch_reduces_band_energy(band: Tuple[float, float]):
    H, W = 96, 192
    rc = make_rc(H, W, device=torch.device("cpu"))
    aug = RCNarrowbandSpectralDropout(bands_range=[band], bands_azimuth=[], soft=False)
    out = aug({"rc": rc})["rc"]

    mask_est = est_mask_from_ratio(rc, out)  # (H,W)
    c0, c1 = band_to_idx(W, *band)
    inside = mask_est[:, c0:c1]
    outside_left = mask_est[:, max(0, c0 - 10):c0] if c0 > 0 else mask_est[:, :1]
    outside_right = mask_est[:, c1:min(W, c1 + 10)] if c1 < W else mask_est[:, -1:]

    assert inside.mean().item() < 0.05
    assert outside_left.mean().item() > 0.80
    assert outside_right.mean().item() > 0.80


@pytest.mark.parametrize("band", [(0.45, 0.55), (0.30, 0.32)])
def test_hard_azimuth_notch_reduces_band_energy(band: Tuple[float, float]):
    H, W = 96, 192
    rc = make_rc(H, W, device=torch.device("cpu"))
    aug = RCNarrowbandSpectralDropout(bands_range=[], bands_azimuth=[band], soft=False)
    out = aug({"rc": rc})["rc"]

    mask_est = est_mask_from_ratio(rc, out)  # (H,W)
    r0, r1 = band_to_idx(H, *band)
    inside = mask_est[r0:r1, :]
    above = mask_est[max(0, r0 - 10):r0, :] if r0 > 0 else mask_est[:1, :]
    below = mask_est[r1:min(H, r1 + 10), :] if r1 < H else mask_est[-1:, :]

    assert inside.mean().item() < 0.05
    assert above.mean().item() > 0.80
    assert below.mean().item() > 0.80


def test_soft_range_notch_monotone_ramps_and_core_zero():
    H, W = 96, 192
    band = (0.40, 0.60)
    edge_taper = 0.2  # 20% of band width per edge
    rc = make_rc(H, W, device=torch.device("cpu"))

    aug = RCNarrowbandSpectralDropout(bands_range=[band], bands_azimuth=[], soft=True, edge_taper=edge_taper)
    out = aug({"rc": rc})["rc"]
    mask_est = est_mask_from_ratio(rc, out).mean(dim=0)  # average across rows -> 1D over W

    c0, c1 = band_to_idx(W, *band)
    bw = c1 - c0
    taper = max(1, round(edge_taper * bw))
    core0, core1 = c0 + taper, c1 - taper

    # Compare small window means for robustness
    def window_mean(x: torch.Tensor, a: int, b: int) -> float:
        if b <= a:
            return float("nan")
        k = max(1, (b - a) // 6)
        head = x[a:a + k].mean().item()
        tail = x[b - k:b].mean().item()
        return head, tail 

    # Left ramp 1 -> 0
    if core0 > c0:
        head, tail = window_mean(mask_est, c0, core0)
        assert head > tail, "Left ramp should decrease from ~1 to ~0"

    # Core ~ 0
    if core1 > core0:
        core = mask_est[core0:core1]
        assert core.mean().item() < 0.1

    # Right ramp 0 -> 1
    if c1 > core1:
        head, tail = window_mean(mask_est, core1, c1)
        assert tail > head, "Right ramp should increase from ~0 to ~1"

    # Outside ~1
    outside = torch.cat([mask_est[max(0, c0 - 10):c0], mask_est[c1:min(W, c1 + 10)]], dim=0)
    if outside.numel() > 0:
        assert outside.mean().item() > 0.8


def test_soft_azimuth_notch_monotone_ramps_and_core_zero():
    H, W = 128, 128
    band = (0.30, 0.70)
    edge_taper = 0.15
    rc = make_rc(H, W, device=torch.device("cpu"))

    aug = RCNarrowbandSpectralDropout(bands_range=[], bands_azimuth=[band], soft=True, edge_taper=edge_taper)
    out = aug({"rc": rc})["rc"]
    mask_est = est_mask_from_ratio(rc, out).mean(dim=1)  # average across cols -> 1D over H

    r0, r1 = band_to_idx(H, *band)
    bw = r1 - r0
    taper = max(1, round(edge_taper * bw))
    core0, core1 = r0 + taper, r1 - taper

    def window_mean(x: torch.Tensor, a: int, b: int) -> Tuple[float, float]:
        k = max(1, (b - a) // 6)
        head = x[a:a + k].mean().item()
        tail = x[b - k:b].mean().item()
        return head, tail

    # Top ramp 1 -> 0
    if core0 > r0:
        head, tail = window_mean(mask_est, r0, core0)
        assert head > tail, "Top ramp should decrease 1 -> 0"

    # Core ~ 0
    if core1 > core0:
        core = mask_est[core0:core1]
        assert core.mean().item() < 0.1

    # Bottom ramp 0 -> 1
    if r1 > core1:
        head, tail = window_mean(mask_est, core1, r1)
        assert tail > head, "Bottom ramp should increase 0 -> 1"

    # Outside ~1
    outside = torch.cat([mask_est[max(0, r0 - 10):r0], mask_est[r1:min(H, r1 + 10)]], dim=0)
    if outside.numel() > 0:
        assert outside.mean().item() > 0.8


def test_multiple_range_bands_and_batch():
    H, W, N = 96, 192, 3
    rc_single = make_rc(H, W, device=torch.device("cpu"))
    rc_batch = torch.stack([rc_single for _ in range(N)], dim=0)  # (N,H,W,2)

    bands = [(0.20, 0.22), (0.48, 0.52), (0.78, 0.80)]
    aug = RCNarrowbandSpectralDropout(bands_range=bands, bands_azimuth=[], soft=True, edge_taper=0.15)
    out = aug({"rc": rc_batch})["rc"]

    assert out.shape == rc_batch.shape

    # Estimate mask from first item (mask is broadcast equal across N)
    mask_est = est_mask_from_ratio(rc_batch[0], out[0])
    for (f0, f1) in bands:
        c0, c1 = band_to_idx(W, f0, f1)
        band_mean = mask_est[:, c0:c1].mean().item()
        assert band_mean < 0.5, f"Band {(f0,f1)} should be attenuated; got mean {band_mean:.3f}"


def test_no_nan_inf_in_output():
    rc = make_rc(64, 128, device=torch.device("cpu"))
    aug = RCNarrowbandSpectralDropout(bands_range=[(0.49, 0.51)], soft=True, edge_taper=0.1)
    out = aug({"rc": rc})["rc"]
    assert torch.isfinite(out).all()
