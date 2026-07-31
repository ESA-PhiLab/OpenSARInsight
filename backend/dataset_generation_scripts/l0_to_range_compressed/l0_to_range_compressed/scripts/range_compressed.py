# type: ignore
"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: range_compression.py

Description:
    Perform range compression of Sentinel 1 decoded L0 data using a nominal image replica
    generated from L0 header parameters.

History:
    - 2025-06-09:
        Orginal version written in matlab.
    - 2025-08-18:
        Added the required comments.
    - 2026-01-15
        Modified to include Internal time delay estimation

---------------------------------------------------------------------
Author: Abdulhameed Yunusa (ABHY)
E-mail: ayunusa@indracompany.com
Creation Date: 2026-02-06

Â© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""
import logging
from typing import Dict, Any, Tuple

import numpy as np
import numpy.typing as npt




def range_compression(
    header: Dict[str, Any],
    raw_data: npt.NDArray[np.complex64],
    cal_pulse: npt.NDArray[np.complex64],
    cal_header: Dict[str, Any]
) -> Tuple[npt.NDArray[np.complex64], int]:
    """
    Perform range compression of Sentinel 1 decoded L0 data using a nominal image replica
    generated from L0 header parameters with internal delay estimation using calibration pulses.

    Args:
        header (dict or object): Sentinel-1 burst header, as loaded from .mat or .npi file. Must contain 'RadarConfiguration'
            with keys/attributes: 'TxPulseLength', 'SamplingFreq', 'TxPulseRampRate', 'TxPulseStartFreq'.
        raw_data (np.ndarray): Raw burst data, shape (azimuth_lines, range_samples), complex-valued.
        cal_pulse (np.ndarray): Calibration pulse data, shape (n_pulses, n_samples) or (n_samples,), complex-valued.
            Used to estimate internal time delay by correlating with nominal pulse replica.
        cal_header (dict or object): Calibration pulse header containing 'RadarConfiguration' with keys/attributes:
            'TxPulseLength', 'SamplingFreq', 'TxPulseRampRate', 'TxPulseStartFreq'.

    Returns:
        range_compressed_data (np.ndarray): Range compressed data, shape (azimuth_lines, range_samples), complex-valued.
        range_ref_funct_samples (int): Length of range reference function (partially focused samples at end of range dimension).
            These should be removed from the generated range_compressed_data matrix if needed.

    Notes:
        Reference: S1-TN-MDA-52-7445. Sentinel-1 Level 1 Detailed Algorithm Definition.
    """

    logging.info("Starting Range Compression")

    # ────────────────────────────────────────────────
    # 1. Imaging radar parameters
    # ────────────────────────────────────────────────
    radar_cfg = header['RadarConfiguration']

    if isinstance(radar_cfg, list):
        radar_cfg = radar_cfg[0]

    txpl  = radar_cfg['TxPulseLength']
    fs    = radar_cfg['SamplingFreq']
    txprr = radar_cfg['TxPulseRampRate']
    txpsf = radar_cfg['TxPulseStartFreq']

    # Chirp phase coefficients
    phi1 = txpsf + txprr * (txpl / 2)
    phi2 = txprr / 2

    range_ref_funct_samples = int(np.round(txpl * fs))

    # ────────────────────────────────────────────────
    # 2. Calibration radar parameters
    # ────────────────────────────────────────────────
    cal_cfg = cal_header['RadarConfiguration']

    txpl_cal  = cal_cfg['TxPulseLength']
    fs_cal    = cal_cfg['SamplingFreq']
    txprr_cal = cal_cfg['TxPulseRampRate']
    txpsf_cal = cal_cfg['TxPulseStartFreq']

    N_cal = int(np.round(txpl_cal * fs_cal))

    # Ensure calibration pulse is 2-D
    if cal_pulse.ndim == 1:
        cal_pulse = cal_pulse[np.newaxis, :]

    mean_cal_pulse = np.mean(cal_pulse, axis=0)

    # ────────────────────────────────────────────────
    # 3. Internal delay estimation
    # ────────────────────────────────────────────────
    extracted_replica = mean_cal_pulse / np.linalg.norm(mean_cal_pulse)

    samples_pg = np.arange(N_cal)
    tn_pg = samples_pg / fs_cal

    nom_pg_replica = np.exp(
        2j * np.pi * (
            txpsf_cal * tn_pg +
            (txprr_cal / 2) * tn_pg**2
        )
    )
    nom_pg_replica /= np.linalg.norm(nom_pg_replica)

    corr = np.correlate(extracted_replica, nom_pg_replica, mode="full")
    time_lag_samples = np.argmax(np.abs(corr)) - (N_cal - 1)
    time_delay_est = time_lag_samples / fs_cal

    logging.info(
        f"Estimated internal delay: "
        f"{time_delay_est * 1e9:.2f} ns "
        f"({time_lag_samples} samples)"
    )

    # ────────────────────────────────────────────────
    # 4. Delay-corrected IMAGE replica
    # ────────────────────────────────────────────────
    samples_img = np.arange(range_ref_funct_samples)
    tn_img = (samples_img - range_ref_funct_samples / 2) / fs
    tn_img -= time_delay_est

    ref_funct = (
        1 / range_ref_funct_samples
    ) * np.exp(
        2j * np.pi * (
            phi1 * tn_img +
            phi2 * tn_img**2
        )
    )

    ref_length = len(ref_funct)

    # ────────────────────────────────────────────────
    # 5. Range compression (matched filtering)
    # ────────────────────────────────────────────────
    a_size, r_size = raw_data.shape
    n_fft = 2 ** int(np.ceil(np.log2(r_size + ref_length)))

    chirp_spectrum = np.fft.fft(ref_funct, n_fft)
    conj_chirp_spectrum = np.conj(chirp_spectrum)

    range_compressed = np.empty((a_size, n_fft), dtype=np.complex64)

    for i in range(a_size):
        spectrum = np.fft.fft(raw_data[i], n_fft)
        range_compressed[i] = np.fft.ifft(spectrum * conj_chirp_spectrum)

    return range_compressed[:, :r_size], ref_length
