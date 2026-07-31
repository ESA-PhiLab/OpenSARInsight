"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: test_rc_normalization_core.py

Description:
    Core unit tests for Range-Compressed (RC) SAR normalization
    using NormalizeRCTransform with RMS-shared variants.

    Checks:
      • Output shape/dtype
      • Mean power normalization (global or per-column)
      • Basic error handling (missing 'rc' key)

Run:
    pytest -q test_rc_normalization_core.py

History:
    - 2025-09-24: First version was added.
     
---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-09-05

© Copyright INDRA DEIMOS, 2025. All rights reserved.
---------------------------------------------------------------------
"""

from __future__ import annotations
import numpy as np
import torch
import logging
from sar_normalization import NormalizeRCTransform
from logging_setup import init_logging

init_logging(level=logging.DEBUG)   # configure once
LOGGER = logging.getLogger(__name__)


# ---------------------------
# Helpers
# ---------------------------
def _dummy_rc_complex(h: int = 48, w: int = 64, seed: int = 1234) -> np.ndarray:
    """Create dummy complex RC data (H, W) with circular complex Gaussian stats."""
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((h, w), dtype=np.float32)
    imag = rng.standard_normal((h, w), dtype=np.float32)
    return (real + 1j * imag).astype(np.complex64)


def _to_hw2_iq(a: np.ndarray) -> np.ndarray:
    """complex (H,W) -> float32 (H,W,2) [I,Q]."""
    return np.stack([a.real, a.imag], axis=-1).astype(np.float32, copy=False)


# ---------------------------
# Core tests
# ---------------------------
def test_rms_shared_basic() -> None:
    """Global RMS-shared: output is float32 (H,W,2) and E[|y|^2] ≈ 1."""
    a = _dummy_rc_complex(64, 96)
    tf = NormalizeRCTransform(method="rms_shared")
    rc = tf({"rc": a})["rc"]  # torch.Tensor (H,W,2) float32

    assert isinstance(rc, torch.Tensor)
    assert rc.dtype == torch.float32 and rc.ndim == 3 and rc.shape[-1] == 2

    power = (rc ** 2).sum(dim=-1)  # (H,W)
    assert abs(power.mean().item() - 1.0) < 1e-4


def test_rms_shared_per_column_basic() -> None:
    """Per-column RMS-shared: each column mean power ≈ 1 (within small tolerance)."""
    a = _dummy_rc_complex(40, 70)
    a_hw2 = _to_hw2_iq(a)
    tf = NormalizeRCTransform(method="rms_shared_per_column")
    rc = tf({"rc": a_hw2})["rc"]

    power = (rc ** 2).sum(dim=-1) # (H,W)
    max_dev = float((power.mean(dim=0) - 1.0).abs().max())
    assert max_dev < 1e-3


def test_missing_key_raises() -> None:
    """Calling without 'rc' key should raise KeyError (basic error handling)."""
    tf = NormalizeRCTransform()
    try:
        _ = tf({"iq": _to_hw2_iq(_dummy_rc_complex(16, 16))})
        assert False, "Expected KeyError for missing 'rc' key"
    except KeyError:
        pass
