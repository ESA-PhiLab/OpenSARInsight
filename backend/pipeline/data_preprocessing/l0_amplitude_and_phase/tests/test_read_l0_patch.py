
"""
Project: OpenSAR Insight
Customer: ESA

File: test_read_l0_patch.py

Description:
    Unit tests for the Level-0 SAR patch reader module.

Functions:
    TestReadL0Patch: Test class for L0 patch reading functionality
    
Author: Abdulhameed Yunusa
Email: ayunusa@indracompany.com
Date: 2025-07-22

© Copyright INDRA DEIMOS, 2025. All rights reserved.
"""

import os
import unittest

import numpy as np
from l0_processing.read_l0_patch import read_l0_patch
from l0_processing.utils import read_config


class TestReadL0Patch(unittest.TestCase):
    """Test suite for the read_l0_patch module."""

    def setUp(self):
        """Set up test fixtures by loading config and sample data path."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        if not os.path.exists(config_path):
            self.skipTest(f"Config file {config_path} not found.")
        self.config = read_config(config_path)
        self.sample_file = self.config.get('L0_patch', {}).get('sample_data', None)
        if self.sample_file:
            self.sample_file = os.path.join(
                os.path.dirname(__file__), '..', self.sample_file
            )

    def test_read_l0_patch_sample_file(self):
        """Test reading L0 patch from the sample file specified in config.

        Reads the sample L0 patch file specified in config.yaml and verifies that
        the output array has the correct shape, data type, and properties.
        """
        if not self.sample_file or not os.path.exists(self.sample_file):
            self.skipTest(f"Sample file {self.sample_file} not found.")

        # Read sample L0 patch
        data = read_l0_patch(self.sample_file)

        # Verify data type is complex64
        self.assertEqual(data.dtype, np.complex64)

        # Verify data is 2D array
        self.assertEqual(len(data.shape), 2)

        # Verify shape has reasonable dimensions (both > 0)
        self.assertGreater(data.shape[0], 0)
        self.assertGreater(data.shape[1], 0)

        # Verify that data contains complex values
        self.assertTrue(np.iscomplexobj(data))

        # Verify that the data is not empty
        self.assertGreater(data.size, 0)

    def test_read_l0_patch_sample_file_specific_dimensions(self):
        """Test that the sample file has the expected specific dimensions."""
        if not self.sample_file or not os.path.exists(self.sample_file):
            self.skipTest(f"Sample file {self.sample_file} not found.")

        # Read sample L0 patch
        data = read_l0_patch(self.sample_file)

        # Verify shape matches expected dimensions from the actual sample file
        # The actual sample file has dimensions (674, 2879)
        self.assertEqual(data.shape, (674, 2879))

    def test_read_l0_patch_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        non_existent_file = "/path/to/non/existent/file.dat"
        with self.assertRaises(FileNotFoundError):
            read_l0_patch(non_existent_file)

    def test_read_l0_patch_sample_file_properties(self):
        """Test additional properties of the L0 patch data from sample file.

        Verifies specific properties of the data read from the sample file.
        """
        if not self.sample_file or not os.path.exists(self.sample_file):
            self.skipTest(f"Sample file {self.sample_file} not found.")

        # Read sample L0 patch
        data = read_l0_patch(self.sample_file)

        # Check that real and imaginary parts exist
        real_part = np.real(data)
        imag_part = np.imag(data)

        # Verify real and imaginary parts have same shape as original data
        self.assertEqual(real_part.shape, data.shape)
        self.assertEqual(imag_part.shape, data.shape)

        # Verify real and imaginary parts are float32
        self.assertEqual(real_part.dtype, np.float32)
        self.assertEqual(imag_part.dtype, np.float32)

        # Verify data is finite (no NaN or inf values)
        self.assertTrue(np.all(np.isfinite(real_part)))
        self.assertTrue(np.all(np.isfinite(imag_part)))


if __name__ == '__main__':
    unittest.main()
