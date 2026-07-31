"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: test_rc_bandwidth_trim.py

Description:
    Unit tests for RCBandwidthTrim. These tests validate:
    • Input validation (shape, missing keys, bad args)
    • Preservation of shape, dtype, device, and extra metadata
    • Identity behavior when trim fractions keep full bandwidth
    • Energy non-increasing in the frequency domain after trimming
    • Extreme trimming (k == 1) produces finite outputs
    • Batched input handling
    • 'cosine' window aliasing to 'hann'
    • Determinism for identical inputs

Run:
    pytest -q test_rc_bandwidth_trim.py
---------------------------------------------------------------------
History:
    - 2025-09-19:
        First version of the core unit tests for RCBandwidthTrim.

---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-09-19

© Copyright INDRA DEIMOS, 2025. All rights reserved.

"""

from __future__ import annotations
import math
from typing import Dict, Tuple
import pytest
import torch
from sar_rc_augmentation import RCBandwidthTrim  


# =====================================================
# Helper Functions
# =====================================================

def make_rc(
    shape: Tuple[int, int, int] = (32, 48, 2),
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Generate a random synthetic RC tensor for testing.

    Args:
        shape: Shape of RC data, default (32, 48, 2).
        device: Device to create tensor on.

    Returns:
        Random RC tensor with both I and Q components (float32).
    """
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(123)
    rc = torch.randn(shape, device=device, dtype=torch.float32)
    rc += 0.05 * torch.sign(rc)  # avoid exact zeros
    return rc


def iq_to_complex(rc: torch.Tensor) -> torch.Tensor:
    """
    Convert (H, W, 2) or (N, H, W, 2) float32 IQ → complex64 (..., H, W).
    """
    if rc.ndim == 3 and rc.shape[-1] == 2:
        real, imag = rc[..., 0], rc[..., 1]
        return real.to(torch.complex64) + 1j * imag.to(torch.complex64)
    if rc.ndim == 4 and rc.shape[-1] == 2:
        real, imag = rc[..., 0], rc[..., 1]
        return real.to(torch.complex64) + 1j * imag.to(torch.complex64)
    raise ValueError("rc must have shape (H, W, 2) or (N, H, W, 2).")


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """
    Centered 2D FFT (like torch.fft.fft2 with fftshift on both in/out).
    Expects x shape (..., H, W) complex.
    """
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    X = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    X = torch.fft.fftshift(X, dim=(-2, -1))
    return X


# =====================================================
# Test Cases
# =====================================================

def test_missing_key_raises_keyerror() -> None:
    """
    Verify that missing 'rc' key in the sample raises KeyError.
    """
    aug = RCBandwidthTrim()
    with pytest.raises(KeyError):
        aug({"not_rc": torch.zeros((8, 8, 2))})


def test_wrong_shape_raises_valueerror() -> None:
    """
    Verify that invalid 'rc' shapes raise ValueError.
    """
    aug = RCBandwidthTrim()
    # Wrong last dim
    with pytest.raises(ValueError):
        aug({"rc": torch.zeros((8, 8, 3))})
    # Not 3D/4D
    with pytest.raises(ValueError):
        aug({"rc": torch.zeros((8, 8))})


@pytest.mark.parametrize("window", ["cosine", "hann", "hamming"])
@pytest.mark.parametrize("frac_r, frac_a", [(0.75, 0.5), (0.5, 0.9)])
def test_frequency_energy_nonincreasing(window: str, frac_r: float, frac_a: float) -> None:
    """
    L2 energy in frequency domain must not increase after windowing (|W|<=1).
    """
    sample = {"rc": make_rc((32, 48, 2))}
    rc0 = sample["rc"].clone()
    Z0 = fft2c(iq_to_complex(rc0))

    aug = RCBandwidthTrim(trim_frac_range=frac_r, trim_frac_azimuth=frac_a, window=window)
    out = aug(sample)
    rc1 = out["rc"]
    Z1 = fft2c(iq_to_complex(rc1))

    e0 = (Z0.abs() ** 2).sum()
    e1 = (Z1.abs() ** 2).sum()
    assert e1 <= e0 + 1e-5, f"Frequency energy increased: {e1.item()} > {e0.item()}"


def test_k_equals_one_no_nan() -> None:
    """
    Choose fractions that force k==1 in both axes; output must be finite.
    """
    H, W = 64, 96
    frac_r = 1.0 / W
    frac_a = 1.0 / H
    aug = RCBandwidthTrim(trim_frac_range=frac_r, trim_frac_azimuth=frac_a, window="hann")
    out = aug({"rc": make_rc((H, W, 2))})
    rc = out["rc"]
    assert torch.isfinite(rc).all(), "Output contains NaN/Inf when k==1."


def test_identity_no_trim_returns_input_exact() -> None:
    """
    When both trims are 1.0, transform must be an exact identity (values equal).
    """
    rc_in = make_rc((33, 41, 2))  # odd sizes to hit centering paths
    aug = RCBandwidthTrim(trim_frac_range=1.0, trim_frac_azimuth=1.0, window="hann")
    out = aug({"rc": rc_in})
    rc_out = out["rc"]
    assert torch.equal(rc_out, rc_in), "Identity trim should return input unmodified."


def test_shape_dtype_and_metadata_passthrough() -> None:
    """
    Verify that shape, dtype, and extra metadata are preserved after augmentation.
    """
    rc = make_rc((16, 20, 2))
    sample: Dict[str, object] = {"rc": rc.clone(), "meta": "keep_me", "id": 42}

    aug = RCBandwidthTrim(trim_frac_range=0.8, trim_frac_azimuth=0.7, window="hann")
    out = aug(sample)  # type: ignore[arg-type]

    # Metadata keys must remain untouched
    assert "rc" in out and "meta" in out and "id" in out
    assert out["meta"] == "keep_me"
    assert out["id"] == 42

    # Shape and dtype preserved
    assert isinstance(out["rc"], torch.Tensor)
    assert out["rc"].dtype == torch.float32
    assert out["rc"].shape == rc.shape


def test_device_is_preserved_cpu() -> None:
    """
    Ensure augmentation preserves CPU device.
    """
    rc = make_rc((12, 18, 2), device=torch.device("cpu"))
    aug = RCBandwidthTrim()
    out = aug({"rc": rc})
    assert out["rc"].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_is_preserved_cuda() -> None:
    """
    Ensure augmentation preserves CUDA device when available.
    """
    rc = make_rc((12, 18, 2), device=torch.device("cuda"))
    aug = RCBandwidthTrim()
    out = aug({"rc": rc})
    assert out["rc"].device.type == "cuda"


def test_batched_input_supported_and_energy_nonincreasing() -> None:
    """
    Verify that batched inputs (N, H, W, 2) are processed correctly
    and frequency-domain energy does not increase.
    """
    N, H, W = 3, 24, 24
    rc = torch.stack([make_rc((H, W, 2)) for _ in range(N)], dim=0)  # (N,H,W,2)

    Z0 = fft2c(iq_to_complex(rc))
    e0 = (Z0.abs() ** 2).sum()

    out = RCBandwidthTrim(0.8, 0.6, "hamming")({"rc": rc})["rc"]

    assert out.shape == rc.shape
    assert out.dtype == torch.float32

    Z1 = fft2c(iq_to_complex(out))
    e1 = (Z1.abs() ** 2).sum()
    assert e1 <= e0 + 1e-5


def test_invalid_trim_values_raise() -> None:
    """
    Invalid trim fractions must raise ValueError.
    """
    with pytest.raises(ValueError):
        _ = RCBandwidthTrim(-0.1, 0.5, "hann")
    with pytest.raises(ValueError):
        _ = RCBandwidthTrim(1.1, 1.0, "hann")
    with pytest.raises(ValueError):
        _ = RCBandwidthTrim(0.5, 0.0, "hann")


def test_invalid_window_raises() -> None:
    """
    Unsupported window names must raise ValueError.
    """
    with pytest.raises(ValueError):
        _ = RCBandwidthTrim(0.9, 0.9, "blackman")  # not supported


def test_cosine_aliases_hann_numerically() -> None:
    """
    'cosine' is treated as 'hann' → outputs should match exactly.
    """
    sample = {"rc": make_rc((36, 28, 2))}
    aug_cos = RCBandwidthTrim(0.7, 0.6, "cosine")
    aug_han = RCBandwidthTrim(0.7, 0.6, "hann")
    out_cos = aug_cos(sample)["rc"]
    out_han = aug_han(sample)["rc"]
    assert torch.allclose(out_cos, out_han, atol=0.0, rtol=0.0), \
        "cosine must alias to hann exactly."


def test_deterministic_for_same_input() -> None:
    """
    Two calls with same input produce identical outputs.
    """
    rc = make_rc((24, 24, 2))
    aug = RCBandwidthTrim(0.65, 0.8, "hamming")
    out1 = aug({"rc": rc.clone()})["rc"]
    out2 = aug({"rc": rc.clone()})["rc"]
    assert torch.equal(out1, out2), "Transform should be deterministic."
