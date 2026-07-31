"""
Project: OpenSAR Insight
Customer: ESA

File: processing.py

Description:
    Core processing functions for Level-0 SAR.

Functions:
    get_amplitude_phase: Extract amplitude and phase from complex SAR data

Author: Abdulhameed Yunusa (ABHY)
E-mail: ayunusa@indracompany.com
Date: 2025-07-22

© Copyright INDRA DEIMOS, 2025. All rights reserved.
"""

from typing import Tuple
import numpy as np
import numpy.typing as npt


def get_amplitude_phase(
    sar_patch: npt.NDArray[np.complex64],
    return_degrees: bool = False
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    """Compute the amplitude and phase of a Level-0 SAR patch.

    Calculates the amplitude (magnitude) and phase of the complex-valued SAR patch data.
    The input is typically obtained from `read_l0_patch`.

    Args:
        sar_patch (numpy.ndarray): Complex-valued SAR patch data of shape (rows, cols), where each element
            is a complex number (real + j*imag).
        return_degrees (bool, optional): If True, return phase in degrees; otherwise, in radians. Default is False.

    Returns:
        amplitude (numpy.ndarray): Amplitude (magnitude) of the SAR patch, same shape as input.
            Computed as sqrt(real^2 + imag^2).
        phase (numpy.ndarray): Phase of the SAR patch, same shape as input, in radians (or degrees if
            return_degrees is True). Computed as arctan2(imag, real).

    Raises:
        ValueError: If sar_patch is not a complex-valued NumPy array.
    """
    if not np.iscomplexobj(sar_patch):
        raise ValueError("sar_patch must be a complex-valued NumPy array.")

    amplitude = np.abs(sar_patch)
    phase = np.angle(sar_patch, deg=return_degrees)
    return amplitude, phase
