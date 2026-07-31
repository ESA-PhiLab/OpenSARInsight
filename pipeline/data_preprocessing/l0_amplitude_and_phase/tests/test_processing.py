
"""
Project: OpenSAR Insight
Customer: ESA

File: test_processing.py

Description:
    Unit tests for the SAR processing module.

Functions:
    TestProcessing: Test class for amplitude and phase computation functions
    
Author : Abdulhameed Yunusa
Email: ayunusa@indracompany.com
Date: 2025-07-22

© Copyright INDRA DEIMOS, 2025. All rights reserved.
"""

import os
import unittest

import numpy as np
import numpy.typing as npt

from l0_processing.processing import get_amplitude_phase
from l0_processing.read_l0_patch import read_l0_patch
from l0_processing.utils import read_config


class TestProcessing(unittest.TestCase):
    """Test suite for the processing module."""

    def setUp(self):
        """Set up test fixtures by loading config and sample data path.

        Loads the configuration from config.yaml and constructs the path to the
        sample L0 patch file.
        """
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        if not os.path.exists(config_path):
            self.skipTest(f"Config file {config_path} not found.")
        self.config = read_config(config_path)
        self.sample_file = self.config.get('L0_patch', {}).get('sample_data', None)
        if self.sample_file:
            self.sample_file = os.path.join(
                os.path.dirname(__file__), '..', self.sample_file
            )

    def test_get_amplitude_phase_known_data(self):
        """Test amplitude and phase computation for known complex data.

        Verifies that amplitude and phase are correctly computed for a known
        complex array in both radians and degrees.
        """
        # Create a 2x2 complex array with known values
        data: npt.NDArray[np.complex64] = np.array(
            [[1 + 1j, 0 + 1j], [1 + 0j, -1 + 0j]], dtype=np.complex64
        )

        # Test in radians (default)
        amp, phase = get_amplitude_phase(data, return_degrees=False)
        expected_amp: npt.NDArray[np.float64] = np.array(
            [[np.sqrt(2.0), 1.0], [1.0, 1.0]], dtype=np.float64
        )
        expected_phase: npt.NDArray[np.float64] = np.array(
            [[np.pi / 4, np.pi / 2], [0.0, np.pi]], dtype=np.float64
        )
        np.testing.assert_almost_equal(amp, expected_amp, decimal=5)
        np.testing.assert_almost_equal(phase, expected_phase, decimal=5)

        # Test in degrees
        amp, phase = get_amplitude_phase(data, return_degrees=True)
        expected_phase_deg: npt.NDArray[np.float64] = np.array(
            [[45.0, 90.0], [0.0, 180.0]], dtype=np.float64
        )
        np.testing.assert_almost_equal(amp, expected_amp, decimal=5)
        np.testing.assert_almost_equal(phase, expected_phase_deg, decimal=5)

    def test_get_amplitude_phase_sample_file(self):
        """Test amplitude and phase computation using sample L0 patch file.

        Reads the sample L0 patch file specified in config.yaml and verifies that
        the output arrays have the correct shape and data type.
        """
        if not self.sample_file or not os.path.exists(self.sample_file):
            self.skipTest(f"Sample file {self.sample_file} not found.")

        # Read sample L0 patch
        data: npt.NDArray[np.complex64] = read_l0_patch(self.sample_file)

        # Compute amplitude and phase
        amp, phase = get_amplitude_phase(data, return_degrees=False)

        # Verify shape matches the input data and types
        self.assertEqual(amp.shape, data.shape)
        self.assertEqual(phase.shape, data.shape)
        self.assertEqual(amp.dtype, np.float32)
        self.assertEqual(phase.dtype, np.float32)

        # Verify amplitude is non-negative
        self.assertTrue(np.all(amp >= 0))

        # Verify phase is within valid range [-pi, pi] for radians
        self.assertTrue(np.all(phase >= -np.pi) & np.all(phase <= np.pi))

    def test_get_amplitude_phase_zero_input(self):
        """Test amplitude and phase computation for zero-valued input.

        Verifies behavior for a zero-valued complex array.
        """
        data: npt.NDArray[np.complex64] = np.zeros((2, 2), dtype=np.complex64)
        amp, phase = get_amplitude_phase(data, return_degrees=False)

        # Amplitude should be zero
        expected_amp: npt.NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
        np.testing.assert_almost_equal(amp, expected_amp, decimal=5)

        # Phase for zero input is undefined but typically 0 in NumPy
        expected_phase: npt.NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
        np.testing.assert_almost_equal(phase, expected_phase, decimal=5)

    def test_invalid_input(self):
        """Test that non-complex input raises ValueError.

        Verifies that the function raises an error for invalid input types.
        """
        # Non-complex array
        with self.assertRaises(ValueError):
            get_amplitude_phase(np.array([1, 2, 3]))  # type: ignore

        # Another non-complex array (real-valued)
        with self.assertRaises(ValueError):
            get_amplitude_phase(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))  # type: ignore


if __name__ == '__main__':
    unittest.main()