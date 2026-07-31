import numpy as np
from typing import Tuple # type: ignore

def range_compression(header, raw_data): # type: ignore
    """
    Perform range compression of Sentinel 1 decoded L0 data using a nominal image replica
    generated from L0 header parameters.

    Args:
        header (dict or object): Sentinel-1 burst header, as loaded from .mat or .npi file. Must contain 'RadarConfiguration'
            with keys/attributes: 'TxPulseLength', 'SamplingFreq', 'TxPulseRampRate', 'TxPulseStartFreq'.
        raw_data (np.ndarray): Raw burst data, shape (azimuth_lines, range_samples), complex-valued.

    Returns:
        range_compressed_data (np.ndarray): Range compressed data, shape (azimuth_lines, range_samples), complex-valued.
        range_ref_funct_samples (int): Length of range reference function (partially focused samples at end of range dimension).
            These should be removed from the generated range_compressed_data matrix if needed.

    Notes:
        Reference: S1-TN-MDA-52-7445. Sentinel-1 Level 1 Detailed Algorithm Definition.

        Author: Juan M Cuerda (INTA)
        Converted to Python by GitHub Copilot
    """
    # Read Burst Info
    radar_config = header['RadarConfiguration'] if isinstance(header, dict) else header.RadarConfiguration # type: ignore

    # Extract unique values (assuming they're the same across all configurations)
    if isinstance(radar_config, list):
        txpl = np.unique([config['TxPulseLength'] if isinstance(config, dict) else config.TxPulseLength for config in radar_config])[0]    # Tx pulse length (s) # type: ignore
        fs = np.unique([config['SamplingFreq'] if isinstance(config, dict) else config.SamplingFreq for config in radar_config])[0]      # Sampling frequency (Hz) # type: ignore
        txprr = np.unique([config['TxPulseRampRate'] if isinstance(config, dict) else config.TxPulseRampRate for config in radar_config])[0]  # TX pulse ramp rate (Hz/s) # type: ignore
        txpsf = np.unique([config['TxPulseStartFreq'] if isinstance(config, dict) else config.TxPulseStartFreq for config in radar_config])[0] # Tx pulse start freq (Hz) # type: ignore
    else:
        # If it's a single configuration object
        txpl = radar_config['TxPulseLength'] if isinstance(radar_config, dict) else radar_config.TxPulseLength
        fs = radar_config['SamplingFreq'] if isinstance(radar_config, dict) else radar_config.SamplingFreq
        txprr = radar_config['TxPulseRampRate'] if isinstance(radar_config, dict) else radar_config.TxPulseRampRate
        txpsf = radar_config['TxPulseStartFreq'] if isinstance(radar_config, dict) else radar_config.TxPulseStartFreq

    range_ref_funct_samples = int(np.round(txpl * fs))  # (s)
    samples = np.arange(0, range_ref_funct_samples) - range_ref_funct_samples / 2  # vector of sample number
    tn = samples / fs  # range time for samples (s)

    # Replica construction
    phi1 = txpsf - txprr * (-txpl / 2)
    phi2 = txprr / 2

    nom_chirp_image = (1 / range_ref_funct_samples) * np.exp(2j * np.pi * (phi1 * tn + phi2 * tn**2))

    # Compression
    a_size = raw_data.shape[0]
    r_size = raw_data.shape[1]
    n_fft = 2 ** int(np.ceil(np.log2(r_size + range_ref_funct_samples + 1)))
    compressed_range_line = np.zeros((a_size, n_fft), dtype=complex)

    for i in range(a_size):
        range_line = raw_data[i, :]
        range_spectrum = np.fft.fft(range_line, n_fft)
        chirp_spectrum = np.fft.fft(nom_chirp_image, n_fft)
        compressed_spectrum = range_spectrum * np.conj(chirp_spectrum)
        compressed_range_line[i, :] = np.fft.ifft(compressed_spectrum)

    # Crop to original size
    range_compressed_data = compressed_range_line[:, :r_size]  # Remove zero padding

    return range_compressed_data, range_ref_funct_samples
