import math


def get_l0_block_size_for_l1b_block_size(slc_block_size: int,
                                         swath_number: int,
                                         pri: float,
                                         vs: float,
                                         ka: float,
                                         txpl: float):
    """
    Calculates the equivalent L0 block size for a fixed-size L1B block based on L0 header parameters and configuration.

    Parameters:
    - SLC_block_size: desired square size in SLC image (typically 512)
    - swath_number: IW swath number [1, 2, 3] of SLC image
    - pri: pulse repetition interval (s)
    - vs: satellite velocity (m/s)
    - ka: azimuth (Doppler) FM rate (rad/s)
    - txpl: tx pulse length (s)

    Returns:
    - n_rg: range dimension size of the L0 block
    - n_az: azimuth dimension size of the L0 block
    """

    # Constants
    fc = 5.405000454334350e9  # S1 radar frequency [Hz]
    c = 299792458  # speed of light [m/s]
    az_time_interval = 0.002055556  # azimuth pixel spacing for SLC

    # Azimuth processing bandwidth for IW1, IW2, IW3 [Hz]
    az_proc_bw = [327, 313, 314]

    # Antenna steering rate in degrees (for IW1, IW2, IW3)
    k_psi_deg = [1.590368784, 0.979863325, 1.397440818]
    k_psi_rad = [math.radians(deg) for deg in k_psi_deg]  # Convert to radians

    # Sampling frequency [Hz] for IW1, IW2, IW3
    fs = [1.0e7 * 6.434523812571428, 1.0e7 * 5.459595962181818, 1.0e7 * 4.69184028]

    # Adjust index for swath_number (1-based to 0-based)
    idx = swath_number - 1

    # Doppler centroid rate induced by antenna scanning
    ks = 2 * vs / c * fc * k_psi_rad[idx]

    # Conversion Factor between focused and raw time
    alpha = 1.0 - (ks / ka)

    # Zero Doppler sampling interval
    dt_zd = pri * alpha

    # Number of data L0 azimuth lines matching the desired SLC_block_size
    n_az_data = (slc_block_size - 1) * az_time_interval / dt_zd + 1

    # Azimuth reference function length
    az_ref_length = az_proc_bw[idx] / abs(ka) / pri

    # Number of L0 block size in azimuth dimension
    n_az = n_az_data + az_ref_length

    # Calculate range size
    rg_ref_length = int(round(txpl * fs[idx]))

    # Range resampling: ratio between raw and SLC sampling frequencies
    SLCRangeSamplingRate = fs[0]
    rawRangeSamplingRate = fs[idx]

    n_rg_data = (slc_block_size - 1) * rawRangeSamplingRate / SLCRangeSamplingRate + 1
    n_rg = n_rg_data + rg_ref_length

    print(f"Range size (n_rg): {n_rg}")
    print(f"Azimuth size (n_az): {n_az}")

    return n_rg, n_az
