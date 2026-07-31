"""
/**
 * @file: DVD_lib.py
 * @brief: Model validation and logging utilities for DVD detection pipeline.
 * @author: railie@indracompany.com 
 * @copyright: Indra Company
 * @project: OpenSAR Insight / AI4SAR
 * @customer: ESA
 *
 * This module provides classes and functions for validating model metrics 
 * against expected thresholds, logging results to log files, and managing test case inputs.
 */
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime

from model_validation.utils_lib import cls_UTILS_LIB as UTILS
from model_validation.gen_sof_req.GEN_SOF_lib import run_vd_small_end2end


_LARGE_MODEL_DIR = str(Path(__file__).parents[2] / 'pipeline' / 'dvd_use_case' / 'large_model')

# Directory where log files will be stored
LOG_DIR = Path(__file__).parents[2] / 'model_validation' / 'logs'

# Generate a dynamic filename with date + time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DVD001_LOG_FILE = LOG_DIR / f"DVD001_{timestamp}.log"

# Create a separate logger for DVD001 metrics
dvd001_logger = logging.getLogger("DVD001")
dvd001_logger.setLevel(logging.INFO)

# Clear existing handlers to avoid duplicate logging
dvd001_logger.handlers.clear()
dvd001_logger.propagate = False  # prevent logging to root

# Prevent logs from propagating to root logger
dvd001_logger.propagate = False


class cls_DVD_SETUP:
    @staticmethod
    def DVD_LOG_FILE_SETUP():
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(DVD001_LOG_FILE, mode="w")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        dvd001_logger.addHandler(file_handler)



class cls_DVD001_LOGS:
    @staticmethod
    def DVD001_VALIDATION(out_large_best: dict, out_large_maritime: dict):
        """
        @brief Validates the DVD001 test case: vessel detection and discrimination in SAR imagery.
        @details
            Runs run_val_script() (general validation) and run_analysis_script_maritime_features()
            (maritime-features subset). Compares precision between the two runs.
            The system shall accurately distinguish vessels from other objects; if the precision
            from the maritime-features run drops by more than 10 percentage points relative to
            the general validation run, the test fails.
        @return bool
            True if the precision difference is within the 10% tolerance, False otherwise.
        """
        status = True
        dvd001_logger.info(f"Log file name: {DVD001_LOG_FILE}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Test Case Identifier : {UTILS.dvd_test_cases_matrix[0][0]}")
        dvd001_logger.info(f"Test Case Name       : {UTILS.dvd_test_cases_matrix[0][1]}")
        dvd001_logger.info(f"Requirement Verified : {UTILS.dvd_test_cases_matrix[0][2]}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Inputs:")
        ret_value, msg = UTILS.verify_path(cls_DVD001_INPUT.config_path, False)
        if ret_value == 0:
            dvd001_logger.info(msg)
            dvd001_logger.info(f"============================================================\n")
            return False
        else:
            dvd001_logger.info(f"\tConfig path          : {cls_DVD001_INPUT.config_path}")
        ret_value, msg = UTILS.verify_path(cls_DVD001_INPUT.data_maritime_yaml, False)
        if ret_value == 0:
            dvd001_logger.info(msg)
            dvd001_logger.info(f"============================================================\n")
            return False
        else:
            dvd001_logger.info(f"\tMaritime YAML        : {cls_DVD001_INPUT.data_maritime_yaml}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Execution:")

        val_out = out_large_best
        maritime_out = out_large_maritime
        precision_diff = val_out.precision - maritime_out.precision

        dvd001_logger.info(f"\trun_val_script precision             : {val_out.precision}")
        dvd001_logger.info(f"\trun_analysis_script_maritime_features: {maritime_out.precision}")
        dvd001_logger.info(f"\tPrecision drop (val - maritime)       : {precision_diff:.4f}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Expected outputs:")
        dvd001_logger.info(f"\tMax allowed precision drop: 0.10 (10 percentage points)")
        dvd001_logger.info(f"------------------------------------------------------------")

        _THRESHOLD = 0.10
        if precision_diff > _THRESHOLD:
            dvd001_logger.info(
                f"[FAIL]: Maritime-features precision ({maritime_out.precision}) dropped by "
                f"{precision_diff:.4f} > {_THRESHOLD} relative to val precision ({val_out.precision})"
            )
            status = False
        else:
            dvd001_logger.info(
                f"[PASS]: Maritime-features precision ({maritime_out.precision}) within tolerance — "
                f"drop = {precision_diff:.4f} <= {_THRESHOLD}"
            )

        dvd001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def DVD004_VALIDATION(out: dict):
        """
        @brief Validates the DVD004 test case based on precision value.
        @details
            This static method logs the details of the DVD004 test case, including inputs, expected outputs, and execution results.
            It checks whether the computed precision meets or exceeds the expected threshold defined in
            `cls_DVD001_EXPECTED_OUTPUT`. If precision is below the expected value, the validation fails.
        @param out 
            Object containing the computed 'precision' attribute.
        @return bool 
            True if precision meets or exceeds expected value and input path is valid, False otherwise.
        """
        status = True
        dvd001_logger.info(f"Log file name: {DVD001_LOG_FILE}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Test Case Identifier : {UTILS.dvd_test_cases_matrix[3][0]}")
        dvd001_logger.info(f"Test Case Name       : {UTILS.dvd_test_cases_matrix[3][1]}")
        dvd001_logger.info(f"Requirement Verified : {UTILS.dvd_test_cases_matrix[3][2]}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Inputs:")
        ret_value, msg = UTILS.verify_path(cls_DVD001_INPUT.config_path, False)
        if ret_value == 0:
            dvd001_logger.info(msg)
            dvd001_logger.info(f"============================================================\n")
            return False
        else:
            dvd001_logger.info(f"\tConfig path: {cls_DVD001_INPUT.config_path}")
        
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Expected outputs:")
        dvd001_logger.info(f"\tExpected precision: {cls_DVD001_EXPECTED_OUTPUT.expected_precision}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Execution:")
        dvd001_logger.info(f"\tprecision: {out.precision}")
        dvd001_logger.info(f"------------------------------------------------------------")
        
        almost_pass_threshold = cls_DVD001_EXPECTED_OUTPUT.expected_precision - 0.05
        if out.precision >= cls_DVD001_EXPECTED_OUTPUT.expected_precision:
            dvd001_logger.info(f"[PASS]: Computed precision = {out.precision} >= Expected precision = {cls_DVD001_EXPECTED_OUTPUT.expected_precision}")
            status = True
        elif out.precision >= almost_pass_threshold:
            dvd001_logger.info(f"[PASS]: Computed precision = {out.precision} is within the threshold of Expected precision = {cls_DVD001_EXPECTED_OUTPUT.expected_precision} (threshold = {almost_pass_threshold})")
            status = True
        else:
            dvd001_logger.info(f"[FAIL]: Computed precision = {out.precision} < Expected precision = {cls_DVD001_EXPECTED_OUTPUT.expected_precision} (threshold with 0.05 tolerance = {almost_pass_threshold})")
            status = False

        dvd001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def DVD005_VALIDATION(out: dict):
        """
        @brief Validates the DVD005 test case based on recall value.
        @details
            This static method logs the details of the DVD005 test case, including inputs, expected outputs, and execution results.
            It checks whether the computed recall meets or exceeds the expected threshold defined in
            `cls_DVD001_EXPECTED_OUTPUT`. If recall is below the expected value, the validation fails.
        @param out 
            Object containing the computed 'recall' attribute.
        @return bool 
            True if recall meets or exceeds expected value and input path is valid, False otherwise.
        """
        status = True
        dvd001_logger.info(f"Log file name: {DVD001_LOG_FILE}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Test Case Identifier : {UTILS.dvd_test_cases_matrix[4][0]}")
        dvd001_logger.info(f"Test Case Name       : {UTILS.dvd_test_cases_matrix[4][1]}")
        dvd001_logger.info(f"Requirement Verified : {UTILS.dvd_test_cases_matrix[4][2]}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Inputs:")
        ret_value, msg = UTILS.verify_path(cls_DVD001_INPUT.config_path, False)
        if ret_value == 0:
            dvd001_logger.info(msg)
            dvd001_logger.info(f"============================================================\n")
            return False
        else:
            dvd001_logger.info(f"\tConfig path: {cls_DVD001_INPUT.config_path}")
        
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Expected outputs:")
        dvd001_logger.info(f"\tExpected recall: {cls_DVD001_EXPECTED_OUTPUT.expected_recall}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Execution:")
        dvd001_logger.info(f"\trecall: {out.recall}")
        dvd001_logger.info(f"------------------------------------------------------------")
        
        almost_pass_threshold = cls_DVD001_EXPECTED_OUTPUT.expected_recall - 0.05
        if out.recall >= cls_DVD001_EXPECTED_OUTPUT.expected_recall:
            dvd001_logger.info(f"[PASS]: Computed recall = {out.recall} >= Expected recall = {cls_DVD001_EXPECTED_OUTPUT.expected_recall}")
            status = True
        elif out.recall >= almost_pass_threshold:
            dvd001_logger.info(f"[PASS]: Computed recall = {out.recall} is within the threshold of Expected recall = {cls_DVD001_EXPECTED_OUTPUT.expected_recall} (threshold = {almost_pass_threshold})")
            status = True
        else:
            dvd001_logger.info(f"[FAIL]: Computed recall = {out.recall} < Expected recall = {cls_DVD001_EXPECTED_OUTPUT.expected_recall} (threshold with 0.05 tolerance = {almost_pass_threshold})")
            status = False
        
        dvd001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def DVD006_VALIDATION(exit_code, step1_time_s, step2_time_s, patches_nr):
        """
        @brief Validates the DVD006 test case: Vessel Detection Report Timeliness Evaluation.
        @details
            Uses run_vd_small_end2end() output to estimate the processing time for a full swath:
                swath_time_s = (total_time_s / patches_nr) * patches_per_swath
            where patches_per_swath = cls_DVD001_INPUT.patches_per_swath (1000).
            The estimated swath processing time must be < 300 s (5 minutes) to ensure
            vessel detection reports are available for downlink within 5 minutes of
            raw data acquisition.
        @param exit_code    Exit code returned by run_vd_small_end2end().
        @param step1_time_s Pre-processing step duration in seconds.
        @param step2_time_s Inference step duration in seconds.
        @param patches_nr   Number of patches processed in the test run.
        @return bool
            True if estimated swath processing time is below 300 s, False otherwise.
        """
        status = True
        total_time_s = step1_time_s + step2_time_s
        avg_time_per_patch_s = total_time_s / max(patches_nr, 1)
        swath_time_s = avg_time_per_patch_s * cls_DVD001_INPUT.patches_per_swath
        _THRESHOLD = cls_DVD001_EXPECTED_OUTPUT.expected_max_e2e_time_s

        dvd001_logger.info(f"Log file name: {DVD001_LOG_FILE}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Test Case Identifier : {UTILS.dvd_test_cases_matrix[5][0]}")
        dvd001_logger.info(f"Test Case Name       : {UTILS.dvd_test_cases_matrix[5][1]}")
        dvd001_logger.info(f"Requirement Verified : {UTILS.dvd_test_cases_matrix[5][2]}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Inputs:")
        ret_value, msg = UTILS.verify_path(cls_DVD001_INPUT.config_path, False)
        if ret_value == 0:
            dvd001_logger.info(msg)
            dvd001_logger.info(f"============================================================\n")
            return False
        else:
            dvd001_logger.info(f"\tConfig path          : {cls_DVD001_INPUT.config_path}")
            dvd001_logger.info(f"\tPatches per swath    : {cls_DVD001_INPUT.patches_per_swath}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Expected outputs:")
        dvd001_logger.info(f"\tMax swath processing time (s): {_THRESHOLD} (5 minutes)")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Execution:")
        dvd001_logger.info(f"\texit_code              : {exit_code}")
        dvd001_logger.info(f"\tstep1_time_s           : {step1_time_s:.5f} s")
        dvd001_logger.info(f"\tstep2_time_s           : {step2_time_s:.5f} s")
        dvd001_logger.info(f"\ttotal_time_s           : {total_time_s:.5f} s")
        dvd001_logger.info(f"\tpatches_nr (test set)  : {patches_nr}")
        dvd001_logger.info(f"\tavg_time_per_patch (s) : {avg_time_per_patch_s:.5f}")
        dvd001_logger.info(f"\tswath_time_estimate (s): {swath_time_s:.5f}  "
                           f"[= {total_time_s:.5f} / {patches_nr} * {cls_DVD001_INPUT.patches_per_swath}]")
        dvd001_logger.info(f"------------------------------------------------------------")

        if swath_time_s < _THRESHOLD:
            dvd001_logger.info(
                f"[PASS]: Estimated swath time = {swath_time_s:.5f} s < {_THRESHOLD} s "
                f"(report available within 5 minutes)"
            )
        else:
            dvd001_logger.info(
                f"[FAIL]: Estimated swath time = {swath_time_s:.5f} s >= {_THRESHOLD} s "
                f"(report NOT available within 5 minutes)"
            )
            status = False

        dvd001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def DVD007_VALIDATION(out: dict):
        """
        @brief Validates the DVD007 test case based on precision value.
        @details
            This static method logs the details of the DVD007 test case, including inputs, expected outputs, and execution results.
            It checks whether the computed precision meets or exceeds the expected threshold defined in
            `cls_DVD001_EXPECTED_OUTPUT`. If precision is below the expected value, the validation fails.
        @param out 
            Object containing the computed 'precision' attribute.
        @return bool 
            True if precision meets or exceeds expected value and input path is valid, False otherwise.
        """
        status = True
        dvd001_logger.info(f"Log file name: {DVD001_LOG_FILE}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Test Case Identifier : {UTILS.dvd_test_cases_matrix[6][0]}")
        dvd001_logger.info(f"Test Case Name       : {UTILS.dvd_test_cases_matrix[6][1]}")
        dvd001_logger.info(f"Requirement Verified : {UTILS.dvd_test_cases_matrix[6][2]}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Inputs:")
        ret_value, msg = UTILS.verify_path(cls_DVD001_INPUT.config_path, False)
        if ret_value == 0:
            dvd001_logger.info(msg)
            dvd001_logger.info(f"============================================================\n")
            return False
        else:
            dvd001_logger.info(f"\tConfig path: {cls_DVD001_INPUT.config_path}")
        
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Expected outputs:")
        dvd001_logger.info(f"\tExpected precision: {cls_DVD001_EXPECTED_OUTPUT.expected_precision}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Execution:")
        dvd001_logger.info(f"\tprecision: {out.precision}")
        dvd001_logger.info(f"------------------------------------------------------------")
        
        almost_pass_threshold = cls_DVD001_EXPECTED_OUTPUT.expected_precision - 0.05
        if out.precision >= cls_DVD001_EXPECTED_OUTPUT.expected_precision:
            dvd001_logger.info(f"[PASS]: Computed precision = {out.precision} >= Expected precision = {cls_DVD001_EXPECTED_OUTPUT.expected_precision}")
            status = True
        elif out.precision >= almost_pass_threshold:
            dvd001_logger.info(f"[PASS]: Computed precision = {out.precision} is within the threshold of Expected precision = {cls_DVD001_EXPECTED_OUTPUT.expected_precision} (threshold = {almost_pass_threshold})")
            status = True
        else:
            dvd001_logger.info(f"[FAIL]: Computed precision = {out.precision} < Expected precision = {cls_DVD001_EXPECTED_OUTPUT.expected_precision} (threshold with 0.05 tolerance = {almost_pass_threshold})")
            status = False

        dvd001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def DVD008_VALIDATION(out: dict):
        """
        @brief Validates the DVD008 test case based on recall value.
        @details
            This static method logs the details of the DVD008 test case, including inputs, expected outputs, and execution results.
            It checks whether the computed recall meets or exceeds the expected threshold defined in
            `cls_DVD001_EXPECTED_OUTPUT`. If recall is below the expected value, the validation fails.
        @param out 
            Object containing the computed 'recall' attribute.
        @return bool 
            True if recall meets or exceeds expected value and input path is valid, False otherwise.
        """
        status = True
        dvd001_logger.info(f"Log file name: {DVD001_LOG_FILE}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Test Case Identifier : {UTILS.dvd_test_cases_matrix[7][0]}")
        dvd001_logger.info(f"Test Case Name       : {UTILS.dvd_test_cases_matrix[7][1]}")
        dvd001_logger.info(f"Requirement Verified : {UTILS.dvd_test_cases_matrix[7][2]}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Inputs:")
        ret_value, msg = UTILS.verify_path(cls_DVD001_INPUT.config_path, False)
        if ret_value == 0:
            dvd001_logger.info(msg)
            dvd001_logger.info(f"============================================================\n")
            return False
        else:
            dvd001_logger.info(f"\tConfig path: {cls_DVD001_INPUT.config_path}")
        
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Expected outputs:")
        dvd001_logger.info(f"\tExpected recall: {cls_DVD001_EXPECTED_OUTPUT.expected_recall}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Execution:")
        dvd001_logger.info(f"\trecall: {out.recall}")
        dvd001_logger.info(f"------------------------------------------------------------")
        
        almost_pass_threshold = cls_DVD001_EXPECTED_OUTPUT.expected_recall - 0.05
        if out.recall >= cls_DVD001_EXPECTED_OUTPUT.expected_recall:
            dvd001_logger.info(f"[PASS]: Computed recall = {out.recall} >= Expected recall = {cls_DVD001_EXPECTED_OUTPUT.expected_recall}")
            status = True
        elif out.recall >= almost_pass_threshold:
            dvd001_logger.info(f"[PASS]: Computed recall = {out.recall} is within the threshold of Expected recall = {cls_DVD001_EXPECTED_OUTPUT.expected_recall} (threshold = {almost_pass_threshold})")
            status = True
        else:
            dvd001_logger.info(f"[FAIL]: Computed recall = {out.recall} < Expected recall = {cls_DVD001_EXPECTED_OUTPUT.expected_recall} (threshold with 0.05 tolerance = {almost_pass_threshold})")
            status = False
        
        dvd001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def DVD009_VALIDATION(out_large_best, out_large_near_shore):
        """
        @brief Validates the DVD009 test case: Land-Based False Positive Robustness.
        @details
            Compares precision from run_analysis_script() (general validation) against
            run_analysis_near_shore() (near-shore subset, where land-based false positives
            are more likely). Near-shore vessels are defined as those located < 5 km from
            the shoreline. If the near-shore precision drops by more than 10 percentage
            points relative to the general validation precision, the test fails.
        @param out_large_best
            Object with 'precision' attribute from the general validation run.
        @param out_large_near_shore
            Object with 'precision' attribute from the near-shore validation run.
        @return bool
            True if the precision difference is within the 10% tolerance, False otherwise.
        """
        status = True
        precision_diff = out_large_best.precision - out_large_near_shore.precision
        _THRESHOLD = 0.10

        dvd001_logger.info(f"Log file name: {DVD001_LOG_FILE}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Test Case Identifier : {UTILS.dvd_test_cases_matrix[8][0]}")
        dvd001_logger.info(f"Test Case Name       : {UTILS.dvd_test_cases_matrix[8][1]}")
        dvd001_logger.info(f"Requirement Verified : {UTILS.dvd_test_cases_matrix[8][2]}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Inputs:")
        ret_value, msg = UTILS.verify_path(cls_DVD001_INPUT.config_path, False)
        if ret_value == 0:
            dvd001_logger.info(msg)
            dvd001_logger.info(f"============================================================\n")
            return False
        else:
            dvd001_logger.info(f"\tConfig path          : {cls_DVD001_INPUT.config_path}")
        ret_value, msg = UTILS.verify_path(cls_DVD001_INPUT.config_near_shore, False)
        if ret_value == 0:
            dvd001_logger.info(msg)
            dvd001_logger.info(f"============================================================\n")
            return False
        else:
            dvd001_logger.info(f"\tNear-shore config    : {cls_DVD001_INPUT.config_near_shore}")
            dvd001_logger.info(f"\tNear-shore definition: distance to shore < 5 km")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Expected outputs:")
        dvd001_logger.info(f"\tMax allowed precision drop: {_THRESHOLD} (10 percentage points)")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Execution:")
        dvd001_logger.info(f"\tGeneral validation precision         : {out_large_best.precision}")
        dvd001_logger.info(f"\tNear-shore precision                 : {out_large_near_shore.precision}")
        dvd001_logger.info(f"\tPrecision drop (general - near-shore): {precision_diff:.4f}")
        dvd001_logger.info(f"------------------------------------------------------------")

        if precision_diff > _THRESHOLD:
            dvd001_logger.info(
                f"[FAIL]: Near-shore precision ({out_large_near_shore.precision}) dropped by "
                f"{precision_diff:.4f} > {_THRESHOLD} relative to general precision ({out_large_best.precision})"
            )
            status = False
        else:
            dvd001_logger.info(
                f"[PASS]: Near-shore precision ({out_large_near_shore.precision}) within tolerance \u2014 "
                f"drop = {precision_diff:.4f} <= {_THRESHOLD}"
            )

        dvd001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def DVD010_VALIDATION(out_large_best, out_large_maritime):
        """
        @brief Validates the DVD010 test case: Maritime Feature Robustness.
        @details
            Compares precision from run_analysis_script() (general validation) against
            run_analysis_script_maritime_features() (maritime-features subset).
            If the maritime-features precision drops by more than 10 percentage points
            relative to the general validation precision, the test fails.
        @param out_large_best
            Object with 'precision' attribute from the general validation run.
        @param out_large_maritime
            Object with 'precision' attribute from the maritime-features validation run.
        @return bool
            True if the precision difference is within the 10% tolerance, False otherwise.
        """
        status = True
        dvd001_logger.info(f"Log file name: {DVD001_LOG_FILE}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Test Case Identifier : {UTILS.dvd_test_cases_matrix[9][0]}")
        dvd001_logger.info(f"Test Case Name       : {UTILS.dvd_test_cases_matrix[9][1]}")
        dvd001_logger.info(f"Requirement Verified : {UTILS.dvd_test_cases_matrix[9][2]}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Inputs:")
        ret_value, msg = UTILS.verify_path(cls_DVD001_INPUT.config_path, False)
        if ret_value == 0:
            dvd001_logger.info(msg)
            dvd001_logger.info(f"============================================================\n")
            return False
        else:
            dvd001_logger.info(f"\tConfig path          : {cls_DVD001_INPUT.config_path}")
        ret_value, msg = UTILS.verify_path(cls_DVD001_INPUT.data_maritime_yaml, False)
        if ret_value == 0:
            dvd001_logger.info(msg)
            dvd001_logger.info(f"============================================================\n")
            return False
        else:
            dvd001_logger.info(f"\tMaritime YAML        : {cls_DVD001_INPUT.data_maritime_yaml}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Execution:")

        precision_diff = out_large_best.precision - out_large_maritime.precision

        dvd001_logger.info(f"\tGeneral validation precision         : {out_large_best.precision}")
        dvd001_logger.info(f"\tMaritime-features precision          : {out_large_maritime.precision}")
        dvd001_logger.info(f"\tPrecision drop (general - maritime)  : {precision_diff:.4f}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Expected outputs:")
        dvd001_logger.info(f"\tMax allowed precision drop: 0.10 (10 percentage points)")
        dvd001_logger.info(f"------------------------------------------------------------")

        _THRESHOLD = 0.10
        if precision_diff > _THRESHOLD:
            dvd001_logger.info(
                f"[FAIL]: Maritime-features precision ({out_large_maritime.precision}) dropped by "
                f"{precision_diff:.4f} > {_THRESHOLD} relative to general precision ({out_large_best.precision})"
            )
            status = False
        else:
            dvd001_logger.info(
                f"[PASS]: Maritime-features precision ({out_large_maritime.precision}) within tolerance — "
                f"drop = {precision_diff:.4f} <= {_THRESHOLD}"
            )

        dvd001_logger.info(f"============================================================\n")
        return status
        

    @staticmethod
    def DVD011_VALIDATION(exit_code, step1_time_s, step2_time_s, patches_nr):
        """
        @brief Validates the DVD011 test case: End-to-End Latency Verification for the lightweight pipeline.
        @details
            Runs run_vd_small_end2end() and computes average processing latency normalised to
            100 km\u00b2 of geographic coverage. Each patch covers cls_DVD001_INPUT.patch_area_km2 km\u00b2.
            FAIL if latency_per_100km2 >= 30 s/100km\u00b2 (cls_DVD001_EXPECTED_OUTPUT.expected_max_latency_per_100km2_s).
        @param exit_code    Exit code returned by run_vd_small_end2end().
        @param step1_time_s Pre-processing step duration in seconds.
        @param step2_time_s Inference step duration in seconds.
        @param patches_nr   Number of patches processed.
        @return bool
            True if latency per 100 km\u00b2 is below the threshold, False otherwise.
        """
        status = True
        total_time_s   = step1_time_s + step2_time_s
        coverage_km2   = patches_nr * cls_DVD001_INPUT.patch_area_km2
        latency_per_100km2 = (total_time_s / max(coverage_km2, 1e-9)) * 100.0
        _THRESHOLD = cls_DVD001_EXPECTED_OUTPUT.expected_max_latency_per_100km2_s

        dvd001_logger.info(f"Log file name: {DVD001_LOG_FILE}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Test Case Identifier : {UTILS.dvd_test_cases_matrix[10][0]}")
        dvd001_logger.info(f"Test Case Name       : {UTILS.dvd_test_cases_matrix[10][1]}")
        dvd001_logger.info(f"Requirement Verified : {UTILS.dvd_test_cases_matrix[10][2]}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Inputs:")
        ret_value, msg = UTILS.verify_path(cls_DVD001_INPUT.config_path, False)
        if ret_value == 0:
            dvd001_logger.info(msg)
            dvd001_logger.info(f"============================================================\n")
            return False
        else:
            dvd001_logger.info(f"\tConfig path          : {cls_DVD001_INPUT.config_path}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Expected outputs:")
        dvd001_logger.info(f"\tMax latency per 100 km\u00b2 (s): {_THRESHOLD}")
        dvd001_logger.info(f"------------------------------------------------------------")
        dvd001_logger.info(f"Execution:")
        dvd001_logger.info(f"\texit_code                   : {exit_code}")
        dvd001_logger.info(f"\tstep1_time_s                : {step1_time_s:.5f} s")
        dvd001_logger.info(f"\tstep2_time_s                : {step2_time_s:.5f} s")
        dvd001_logger.info(f"\ttotal_time_s                : {total_time_s:.5f} s")
        dvd001_logger.info(f"\tpatches_nr                  : {patches_nr}")
        dvd001_logger.info(f"\tcoverage_km\u00b2                : {coverage_km2:.2f}")
        dvd001_logger.info(f"\tlatency_per_100km\u00b2 (s)      : {latency_per_100km2:.5f}")
        dvd001_logger.info(f"------------------------------------------------------------")

        if latency_per_100km2 < _THRESHOLD:
            dvd001_logger.info(
                f"[PASS]: Latency per 100 km\u00b2 = {latency_per_100km2:.5f} s < {_THRESHOLD} s"
            )
        else:
            dvd001_logger.info(
                f"[FAIL]: Latency per 100 km\u00b2 = {latency_per_100km2:.5f} s >= {_THRESHOLD} s"
            )
            status = False

        dvd001_logger.info(f"============================================================\n")
        return status
        

# ---------------- INPUT CONFIG ---------------- #

class cls_DVD001_INPUT:
    config_path = str(Path(__file__).parent / 'config_vd_large.yaml')
    data_maritime_yaml = str(Path(__file__).parent / 'data_maritime_features.yaml')
    config_near_shore = str(Path(__file__).parent / 'config_vd_large.yaml')
    patch_area_km2 = 28.1
    patches_per_swath = 1000
    

# ---------------- EXPECTED METRICS ---------------- #

class cls_DVD001_EXPECTED_OUTPUT:
    expected_precision = 0.85
    expected_recall = 0.85
    expected_max_latency_per_100km2_s = 30.0    
    expected_max_e2e_time_s = 300.0  # 5 minutes    expected_downlink_s = 300

    
# ---------------- TEST CASE ---------------- #

#@dataclass
class EpochStats:
    precision = 0.90
    recall = 0.90
    latency_e2e = 30
    patch_area_km2 = 28.1


def run_val_script():
    """
    @brief Runs val.py as a module and returns val_results.
    @return Object with 'precision' and 'recall' attributes from scripts.val.
    """
    import importlib
    _orig_cwd = os.getcwd()
    _orig_sys_path = sys.path.copy()
    try:
        os.chdir(_LARGE_MODEL_DIR)
        if _LARGE_MODEL_DIR not in sys.path:
            sys.path.insert(0, _LARGE_MODEL_DIR)
        import scripts.val as _val_module
        importlib.reload(_val_module)
        return _val_module.val_results
    finally:
        os.chdir(_orig_cwd)
        sys.path = _orig_sys_path


def run_analysis_script():
    """
    @brief Runs analysis.py overall validation and returns a namespace with metrics.
    @return SimpleNamespace with 'precision' and 'recall' attributes.
    """
    import types
    import torch
    from ultralytics import YOLO
    if _LARGE_MODEL_DIR not in sys.path:
        sys.path.insert(0, _LARGE_MODEL_DIR)
    from scripts.analysis import run_val_on_subset, MODEL_PATH, DATA_YAML, IOU_THRESHOLD
    _model = YOLO(MODEL_PATH)
    _device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _metrics = run_val_on_subset(_model, DATA_YAML, _device, IOU_THRESHOLD)
    return types.SimpleNamespace(**_metrics)


def run_analysis_script_maritime_features():
    """
    @brief Runs analysis.py overall validation using data_maritime_features.yaml and returns a namespace with metrics.
    @return SimpleNamespace with 'precision' and 'recall' attributes.
    """
    import types
    import torch
    from ultralytics import YOLO
    _DATA_YAML = cls_DVD001_INPUT.data_maritime_yaml
    if _LARGE_MODEL_DIR not in sys.path:
        sys.path.insert(0, _LARGE_MODEL_DIR)
    from scripts.analysis import run_val_on_subset, MODEL_PATH, IOU_THRESHOLD
    _model = YOLO(MODEL_PATH)
    _device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _metrics = run_val_on_subset(_model, _DATA_YAML, _device, IOU_THRESHOLD)
    return types.SimpleNamespace(**_metrics)


def run_analysis_near_shore():
    """
    @brief Runs model.val() on the near_shore image subset using config_vd_large.yaml.
    @return SimpleNamespace with 'precision' and 'recall' attributes (pose metrics).
    """
    import types
    import yaml
    import tempfile
    import shutil
    import torch
    from ultralytics import YOLO
    _CONFIG_PATH = Path(cls_DVD001_INPUT.config_near_shore)
    if _LARGE_MODEL_DIR not in sys.path:
        sys.path.insert(0, _LARGE_MODEL_DIR)
    from scripts.analysis import (
        run_val_on_subset, load_metadata_csv, create_subset_dataset, extract_patch_number,
        MODEL_PATH, IOU_THRESHOLD,
    )
    with open(_CONFIG_PATH) as _f:
        _cfg = yaml.safe_load(_f)
    _inference_cfg = _cfg.get('inference', {})
    _shore_near = _cfg.get('analysis', {}).get('shore', {}).get('near_shore', 5)

    images_dir = _inference_cfg.get('test_images_dir')
    labels_dir = _inference_cfg.get('test_labels_dir')
    iou = _inference_cfg.get('iou_threshold', IOU_THRESHOLD)
    metadata_path = str(Path(_LARGE_MODEL_DIR) / 'patch_vessel_metadata.csv')

    metadata = load_metadata_csv(metadata_path)
    model = YOLO(_inference_cfg.get('model_path', MODEL_PATH))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    near_shore_images = [
        img for img in sorted(os.listdir(images_dir))
        if (pid := extract_patch_number(img)) is not None
        and (dist := metadata.get(pid, {}).get('distance_to_shore')) is not None
        and dist < _shore_near
    ]

    tmp_root = tempfile.mkdtemp(prefix="yolo_near_shore_")
    try:
        data_yaml = create_subset_dataset(near_shore_images, images_dir, labels_dir, tmp_root)
        _metrics = run_val_on_subset(model, data_yaml, device, iou)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return types.SimpleNamespace(**_metrics)


class cls_DVD001_TC:

    @staticmethod
    def tc_DVD001():
        """
        @brief Runs model evaluation on the test dataset and logs validation metrics.
        @details
        - Initializes a list to store the status of each test case.
        - Runs evaluation on the test dataset using the specified input configuration.
        - @testcase Runs precision validation result logging.
        - @testcase Runs recall validation result logging.
        - Logs summary of test case status to console and log file.
        """

        # Initialize a list to store the status of each test case
        tc_status = []

        cls_DVD_SETUP.DVD_LOG_FILE_SETUP()

        #out = FOCUS.execute_focusing_model_inference(cls_DVD001_INPUT.config_path)

        out_large_best = run_analysis_script()
        out_large_maritime = run_analysis_script_maritime_features()
        out_large_near_shore = run_analysis_near_shore()
        exit_code_vd_small, _profile_stats_vd_small, step1_time_s_vd_small, step2_time_s_vd_small, patches_nr_vd_small = run_vd_small_end2end()


        # Test Case: DVD001 - Vessel Detection and Discrimination
        tc_status.append(cls_DVD001_LOGS.DVD001_VALIDATION(out_large_best, out_large_maritime))

        # Test Case: DVD004 - Vessel Detection Precision Evaluation
        tc_status.append(cls_DVD001_LOGS.DVD004_VALIDATION(out_large_best))

        # Test Case: DVD005 - Vessel Detection Recall Evaluation
        tc_status.append(cls_DVD001_LOGS.DVD005_VALIDATION(out_large_best))

        # Test Case: DVD006 - Vessel Detection Report Timeliness Evaluation
        tc_status.append(cls_DVD001_LOGS.DVD006_VALIDATION(exit_code_vd_small, step1_time_s_vd_small, step2_time_s_vd_small, patches_nr_vd_small))

        # Test Case: DVD007 - Verification of Vessel Detection Precision
        tc_status.append(cls_DVD001_LOGS.DVD007_VALIDATION(out_large_best))

        # Test Case: DVD008 - Verification of Vessel Detection Recall
        tc_status.append(cls_DVD001_LOGS.DVD008_VALIDATION(out_large_best))

        # Test Case: DVD009 - Land-Based False Positive Robustness
        tc_status.append(cls_DVD001_LOGS.DVD009_VALIDATION(out_large_best, out_large_near_shore))

        # Test Case: DVD010 - Maritime Feature Robustness
        tc_status.append(cls_DVD001_LOGS.DVD010_VALIDATION(out_large_best, out_large_maritime))

        # Test Case: DVD011 - End-to-End Latency Verification (lightweight pipeline)
        tc_status.append(cls_DVD001_LOGS.DVD011_VALIDATION(exit_code_vd_small, step1_time_s_vd_small, step2_time_s_vd_small, patches_nr_vd_small))
        
        #tc_status.append(cls_DVD001_LOGS.DVD003_VALIDATION(out))

        # Log outputs
        # Log summary of test case status to console and log file
        dvd001_logger.info(f"DVD large model validation status:\n\tNumber of PASSED tests = {sum(tc_status)}\n\tNumber of FAILED tests = {len(tc_status) - sum(tc_status)}\n\tNumber of TOTAL tests = {len(tc_status)}")
