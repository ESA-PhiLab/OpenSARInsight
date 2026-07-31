"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: test_rc_azimuth_defocus.py

Description:
    Unit tests for RCAzimuthDefocus. These tests validate:
    • Input validation (shape, missing keys)
    • Preservation of shape, dtype, device, and extra metadata
    • Identity behavior for kappa=0
    • Phase-only behavior: |Z| (FFT magnitude) is invariant
    • L2 energy invariance (Parseval) within numerical tolerance
    • Inverse-pair: applying +kappa then -kappa recovers input
    • Batched input handling
    • Determinism for identical inputs
    • Finiteness for extreme kappa

Run:
    pytest -q test_rc_azimuth_defocus.py
---------------------------------------------------------------------
History:
    - 2025-09-22:
        First version of unit tests for RCAzimuthDefocus.

---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-09-22

© Copyright INDRA DEIMOS, 2025. All rights reserved.
"""

from __future__ import annotations
from typing import Dict, Tuple
import pytest
import torch
from sar_rc_augmentation import RCAzimuthDefocus


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
    aug = RCAzimuthDefocus()
    with pytest.raises(KeyError):
        aug({"not_rc": torch.zeros((8, 8, 2))})


def test_wrong_shape_raises_valueerror() -> None:
    """
    Verify that invalid 'rc' shapes raise ValueError.
    """
    aug = RCAzimuthDefocus()
    # Wrong last dim
    with pytest.raises(ValueError):
        aug({"rc": torch.zeros((8, 8, 3))})
    # Not 3D/4D
    with pytest.raises(ValueError):
        aug({"rc": torch.zeros((8, 8))})


@pytest.mark.parametrize("kappa", [0.0, 1e-4, -5e-4])
def test_fft_magnitude_invariant(kappa: float) -> None:
    """
    |FFT| must be invariant (phase-only transform).
    """
    rc = make_rc((40, 52, 2))
    z0 = iq_to_complex(rc)
    Z0 = fft2c(z0)

    out = RCAzimuthDefocus(kappa)({"rc": rc})["rc"]
    z1 = iq_to_complex(out)
    Z1 = fft2c(z1)

    assert torch.allclose(Z1.abs(), Z0.abs(), rtol=1e-6, atol=1e-6), \
        "FFT magnitude changed, but azimuth defocus should be phase-only."


@pytest.mark.parametrize("kappa", [0.0, 2e-3, -2e-3])
def test_parseval_energy_invariant(kappa: float) -> None:
    """
    L2 energy in spatial domain should be preserved (unitary FFT + |phase|=1).
    """
    rc = make_rc((31, 45, 2))
    e0 = (iq_to_complex(rc).abs() ** 2).sum()

    out = RCAzimuthDefocus(kappa)({"rc": rc})["rc"]
    e1 = (iq_to_complex(out).abs() ** 2).sum()

    # Numerical tolerance due to float32/FFT round-off
    assert torch.allclose(e1, e0, rtol=1e-5, atol=1e-5), \
        f"Energy changed: {e1.item()} vs {e0.item()}"


def test_identity_kappa_zero_near_exact() -> None:
    """
    kappa=0 should behave as identity (up to numerical precision).
    """
    rc = make_rc((33, 41, 2))  # odd sizes to hit centering paths
    aug = RCAzimuthDefocus(kappa=0.0)
    out = aug({"rc": rc})["rc"]
    # Use tight tolerance rather than exact equality (FFT/IFFT round-off).
    assert torch.allclose(out, rc, rtol=1e-6, atol=1e-6), \
        "kappa=0 should produce (nearly) identical output."


def test_shape_dtype_and_metadata_passthrough() -> None:
    """
    Verify that shape, dtype, and extra metadata are preserved after augmentation.
    """
    rc = make_rc((16, 20, 2))
    sample: Dict[str, object] = {"rc": rc.clone(), "meta": "keep_me", "id": 42}

    out = RCAzimuthDefocus(kappa=7.5e-4)(sample) 

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
    out = RCAzimuthDefocus()({"rc": rc})
    assert out["rc"].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_is_preserved_cuda() -> None:
    """
    Ensure augmentation preserves CUDA device when available.
    """
    rc = make_rc((12, 18, 2), device=torch.device("cuda"))
    out = RCAzimuthDefocus()({"rc": rc})
    assert out["rc"].device.type == "cuda"


def test_batched_input_supported_and_magnitude_invariant() -> None:
    """
    Verify that batched inputs (N, H, W, 2) are processed correctly
    and FFT magnitude is invariant for each batch item.
    """
    N, H, W = 3, 24, 24
    rc = torch.stack([make_rc((H, W, 2)) for _ in range(N)], dim=0)  # (N,H,W,2)

    Z0 = fft2c(iq_to_complex(rc))
    out = RCAzimuthDefocus(6e-4)({"rc": rc})["rc"]
    Z1 = fft2c(iq_to_complex(out))

    assert out.shape == rc.shape
    assert out.dtype == torch.float32
    assert torch.allclose(Z1.abs(), Z0.abs(), rtol=1e-6, atol=1e-6)


def test_inverse_pair_recovers_original() -> None:
    """
    Applying +kappa followed by -kappa should recover the input (within tol).
    """
    rc = make_rc((28, 36, 2))
    kappa = 8e-4
    aug_fwd = RCAzimuthDefocus(kappa)
    aug_inv = RCAzimuthDefocus(-kappa)

    rc_fwd = aug_fwd({"rc": rc.clone()})["rc"]
    rc_back = aug_inv({"rc": rc_fwd})["rc"]

    assert torch.allclose(rc_back, rc, rtol=1e-5, atol=1e-5), \
        "Inverse pair (+kappa then -kappa) did not recover input."


@pytest.mark.parametrize("kappa", [1e-8, 1e-4, 1e-2])
def test_deterministic_for_same_input(kappa: float) -> None:
    """
    Two calls with same input and kappa produce identical outputs.
    """
    rc = make_rc((24, 24, 2))
    aug = RCAzimuthDefocus(kappa)
    out1 = aug({"rc": rc.clone()})["rc"]
    out2 = aug({"rc": rc.clone()})["rc"]
    assert torch.equal(out1, out2), "Transform should be deterministic."


@pytest.mark.parametrize("kappa", [0.0, 1e-1, -1e-1, 1.0])
def test_outputs_are_finite_even_for_large_kappa(kappa: float) -> None:
    """
    Large |kappa| should still yield finite outputs (unit-modulus phase).
    """
    rc = make_rc((32, 32, 2))
    out = RCAzimuthDefocus(kappa)({"rc": rc})["rc"]
    assert torch.isfinite(out).all(), "Output contains NaN/Inf for large kappa."
