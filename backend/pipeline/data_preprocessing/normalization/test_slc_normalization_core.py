"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: test_normalization_core.py

Description:
    This script contains the core unit test for the SAR SLC normalization, including:
        - NormalizeSLCAmpTransform

    These tests validate the essential functionality of the normalization code, ensuring correct behavior for valid inputs and proper error
    handling for invalid cases.


Run:
    pytest -q test_slc_normalization_core.py


History:
    - 2025-08-07:
        This is the first version of the core unit test for the SLC normalization script.
    - 2025-08-18:
        Added the required comments.

---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-08-04

© Copyright INDRA DEIMOS, 2025. All rights reserved.

"""

from __future__ import annotations
import numpy as np
import torch
from torchvision.transforms import Compose
from sar_normalization import NormalizeSLCAmpTransform
from logging_setup import init_logging
import logging
init_logging(level=logging.DEBUG)   # configure once
LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _dummy_slc() -> np.ndarray:
    """
        Generate a dummy SLC amplitude sample for testing.

        Args:
            None
    """
    return (np.abs(np.random.randn(32, 32)) * 50).astype("float32")


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_slc_range_basic():
    """
        Validate that SLC normalization maps values to the [0, 1] range after percentile clipping.

        Args:
            None
    """
    tf  = NormalizeSLCAmpTransform(lower_pct=2, upper_pct=98)
    out = tf({"slc": _dummy_slc()})["slc"]
    
    # allow numerical epsilon
    assert 0.0 <= out.min() <= 1e-6            
    assert 1.0 - 1e-6 <= out.max() <= 1.0



