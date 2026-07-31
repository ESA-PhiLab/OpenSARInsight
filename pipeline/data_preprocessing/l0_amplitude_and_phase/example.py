"""
Project: OpenSAR Insight
Customer: ESA

File: example.py

Description:
    Example demonstration of Level-0 SAR patch processing.

Functions:
    main: Demonstrate reading and processing an L0 SAR patch
    
Author : Abdulhameed Yunusa
Email: ayunusa@indracompany.com
Date: 2025-07-22

© Copyright INDRA DEIMOS, 2025. All rights reserved.
"""

import os
import matplotlib
# Check if we're in a headless environment
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt

from l0_processing.read_l0_patch import read_l0_patch
from l0_processing.processing import get_amplitude_phase
from l0_processing.utils import read_config
import numpy as np


def main():
    """Demonstrate reading and processing an L0 SAR patch."""
    print("L0 patch processing example")
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found.")

    config = read_config(config_path)
    sample_file = config.get('L0_patch', {}).get('sample_data', None)
    if not sample_file:
        raise ValueError("Sample data path not found in config.yaml.")

    sample_file = os.path.join(os.path.dirname(__file__), sample_file)
    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"Sample file {sample_file} not found.")

    print(f"Reading L0 patch from: {sample_file}")
    
    # Read L0 patch
    l0_patch = read_l0_patch(sample_file)
    print(f"Successfully read SAR data with shape: {l0_patch.shape}")
    print(f"Data type: {l0_patch.dtype}")
    print(f'Slice of L0 Patch: {l0_patch[10:11]}')
    # Compute amplitude and phase
    print("Computing amplitude and phase")
    amplitude, phase = get_amplitude_phase(l0_patch, return_degrees=True)
    print(f"Amplitude range: [{amplitude.min():.2f}, {amplitude.max():.2f}]")
    print(f"Phase range: [{phase.min():.2f}, {phase.max():.2f}] degrees")

    # Visualize results
    print("Creating visualization")
    plt.figure(figsize=(12, 10))  # type: ignore  # 4 plots, 2 per line

    # Amplitude plot
    plt.subplot(2, 2, 1)  # type: ignore
    plt.imshow(amplitude)  # type: ignore
    plt.title('Amplitude')  # type: ignore
    plt.xlabel('Samples')  # type: ignore
    plt.ylabel('Pulses')  # type: ignore

    # Phase plot
    plt.subplot(2, 2, 2)  # type: ignore
    plt.imshow(phase)  # type: ignore
    plt.title('Phase (degrees)')  # type: ignore

    plt.xlabel('Samples')  # type: ignore
    plt.ylabel('Pulses')  # type: ignore

    # Intensity plot
    intensity = amplitude ** 2
    plt.subplot(2, 2, 3)  # type: ignore
    plt.imshow(intensity)  # type: ignore
    plt.title('Intensity')  # type: ignore
    plt.xlabel('Samples')  # type: ignore
    plt.ylabel('Pulses')  # type: ignore

    # Combined amplitude and phase plot
    amp_norm = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min())
    phase_norm = (phase - phase.min()) / (phase.max() - phase.min())

    rgb_img = np.zeros(amplitude.shape + (3,), dtype=np.float32)
    rgb_img[..., 0] = amp_norm  # Red channel: normalized amplitude
    rgb_img[..., 1] = phase_norm  # Green channel: normalized phase
    rgb_img[..., 2] = 0  # Blue channel: unused

    plt.subplot(2, 2, 4)  # type: ignore
    plt.imshow(rgb_img, cmap='gray')  # type: ignore
    plt.title('Amplitude + Phase')  # type: ignore
    plt.xlabel('Samples')  # type: ignore
    plt.ylabel('Pulses')  # type: ignore

    plt.tight_layout()
    # Save the plot
    output_file = 'sar_amplitude_phase.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')  # type: ignore
    print(f"Plot saved as: {output_file}")
    print("\nDone")


if __name__ == '__main__':
    main()
