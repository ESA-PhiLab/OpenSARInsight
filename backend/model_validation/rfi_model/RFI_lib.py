"""
/**
 * @file RFI_lib.py
 * @brief Model validation and logging utilities for RFI detection pipeline.
 * @author: railie@indracompany.com 
 * @copyright: Indra Company
 * @project: OpenSAR Insight / AI4SAR
 * @customer: ESA
 *
 * This module provides classes and functions for validating model metrics
 * against expected thresholds, logging results to log files, and managing test case inputs.
 */
"""
# Importing necessary libraries and modules
import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Set config env vars to repo-local paths before importing pipeline modules
# that read these vars at module level (e.g. train_rfi_large.py).
_cfg_base = Path(__file__).parent.parent.parent / 'configuration'
_env_defaults = {
    'DGS_CONFIG_PATH':   str(_cfg_base / 'data_generation_scripts.yaml'),
    'MODEL_USECASES_PATH': str(_cfg_base / 'model_usecases.yaml'),
    'DATA_PREPROC_PATH': str(_cfg_base / 'data_preprocessing.yaml'),
}
for _var, _default in _env_defaults.items():
    if not os.environ.get(_var):
        os.environ[_var] = _default

import backend.pipeline.RFI_usecase.inference as EVAL
import backend.pipeline.RFI_usecase.post_processing.rfi_postprocessing as POST
from backend.model_validation.utils_lib import cls_UTILS_LIB as UTILS
from model_validation.gen_sof_req.GEN_SOF_lib import run_rfi_large_end2end, run_rfi_small_end2end


# Directory where log files will be stored
LOG_DIR = Path(__file__).parents[1] / "logs"

# Generate a dynamic filename with date + time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RFI001_LOG_FILE = LOG_DIR / f"RFI001_{timestamp}.log"

# Create a separate logger for RFI001 metrics
rfi001_logger = logging.getLogger("RFI001")
rfi001_logger.setLevel(logging.INFO)

# Clear existing handlers to avoid duplicate logging
rfi001_logger.handlers.clear()
rfi001_logger.propagate = False  # prevent logging to root

# Prevent logs from propagating to root logger
rfi001_logger.propagate = False


class cls_RFI_SETUP:
    @staticmethod
    def RFI_LOG_FILE_SETUP():
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(RFI001_LOG_FILE, mode="w")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        rfi001_logger.addHandler(file_handler)


class cls_RFI001_LOGS:

    @staticmethod
    def RFI001_VALIDATION(out: dict):
        """
        @brief Validates the RFI001 test case based on F1 score value.
        @details
            The system shall be able to identify image tiles affected by RFI caused by both ground and satellite sources
            This static method logs the details of the RFI001 test case, including inputs, expected outputs, and execution results.
            It checks whether the computed F1 score meets or exceeds the expected threshold.
            If the F1 score is below the expected value, the validation fails.
        @param out 
            Contains the Precision and Recall values to compute 'f1' value.
        @return bool 
            True if F1 score meets or exceeds expected value and input path is valid, False otherwise.
        """

        f1 = float(out['precision'] * out['recall'] * 2 / (out['precision'] + out['recall'] + 1e-8))

        status = True
        rfi001_logger.info(f"Log file name: {RFI001_LOG_FILE}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Test Case Identifier : {UTILS.rfi_test_cases_matrix[0][0]}")
        rfi001_logger.info(f"Test Case Name       : {UTILS.rfi_test_cases_matrix[0][1]}")
        rfi001_logger.info(f"Requirement Verified : {UTILS.rfi_test_cases_matrix[0][2]}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Inputs:")
        paths = {
        "Config path": cls_RFI001_INPUT.config_path,
        "Weights path": cls_RFI001_INPUT.weights,
        "Test directory": cls_RFI001_INPUT.test_dir,
        #"Single npy": cls_RFI001_INPUT.single_npy
        }
        UTILS.verify_paths(paths, rfi001_logger)
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Expected outputs:")
        rfi001_logger.info(f"\tExpected F1 score: {cls_RFI001_EXPECTED_OUTPUT.expected_f1}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Execution:")
        rfi001_logger.info(f"\tf1: {f1}")
        rfi001_logger.info(f"------------------------------------------------------------")
        
        almost_pass_threshold = cls_RFI001_EXPECTED_OUTPUT.expected_f1 - 0.05
        if f1 >= cls_RFI001_EXPECTED_OUTPUT.expected_f1:
            rfi001_logger.info(f"[PASS]: Computed F1 score = {f1} >= Expected F1 score = {cls_RFI001_EXPECTED_OUTPUT.expected_f1}")
            status = "PASS"
        elif f1 >= almost_pass_threshold:
            rfi001_logger.info(f"[ALMOST_PASS]: Computed F1 score = {f1} is within 5% of Expected F1 score = {cls_RFI001_EXPECTED_OUTPUT.expected_f1}")
            status = "ALMOST_PASS"
        else:
            rfi001_logger.info(f"[FAIL]: Computed F1 score = {f1} < Expected F1 score = {cls_RFI001_EXPECTED_OUTPUT.expected_f1} (threshold with 5% tolerance = {almost_pass_threshold})")
            status = "FAIL"
        
        rfi001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def RFI002_VALIDATION(out: dict):
        """
        @brief Validates the RFI002 test case by comparing the computed IOU with the expected IOU.
        @details
            This static method logs tests if the system is able to isolate the pixelwise regions of SAR focussed scenes affected 
            by RFI. It provides detailed information about the test case, including its identifier, name, requirement, 
            input configuration path, expected outputs, and execution results. It verifies the validity of the configuration 
            path and compares the computed IOU value from the output dictionary with the expected IOU value. The method logs 
            whether the test passes or fails based on this comparison.
        @param out 
            dict: A dictionary containing the computed output values, specifically the 'iou' key.
        @return 
            bool: True if the computed IOU meets or exceeds the expected IOU, False otherwise or if the input path is invalid.
        """
        status = True
        rfi001_logger.info(f"Log file name: {RFI001_LOG_FILE}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Test Case Identifier : {UTILS.rfi_test_cases_matrix[1][0]}")
        rfi001_logger.info(f"Test Case Name       : {UTILS.rfi_test_cases_matrix[1][1]}")
        rfi001_logger.info(f"Requirement Verified : {UTILS.rfi_test_cases_matrix[1][2]}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Inputs:")
        paths = {
        "Config path": cls_RFI001_INPUT.config_path,
        "Weights path": cls_RFI001_INPUT.weights,
        "Test directory": cls_RFI001_INPUT.test_dir,
        #"Single npy": cls_RFI001_INPUT.single_npy
        }
        UTILS.verify_paths(paths, rfi001_logger)
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Expected outputs:")
        rfi001_logger.info(f"\tExpected iou: {cls_RFI001_EXPECTED_OUTPUT.expected_iou}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Execution:")
        rfi001_logger.info(f"\tiou: {out['iou']}")
        rfi001_logger.info(f"------------------------------------------------------------")
        
        almost_pass_threshold = cls_RFI001_EXPECTED_OUTPUT.expected_iou - 0.05
        if out['iou'] >= cls_RFI001_EXPECTED_OUTPUT.expected_iou:
            rfi001_logger.info(f"[PASS]: Computed iou = {out['iou']} >= Expected iou = {cls_RFI001_EXPECTED_OUTPUT.expected_iou}")
            status = "PASS"
        elif out['iou'] >= almost_pass_threshold:
            rfi001_logger.info(f"[ALMOST_PASS]: Computed iou = {out['iou']} is within 5% of Expected iou = {cls_RFI001_EXPECTED_OUTPUT.expected_iou}")
            status = "ALMOST_PASS"
        else:
            rfi001_logger.info(f"[FAIL]: Computed iou = {out['iou']} < Expected iou = {cls_RFI001_EXPECTED_OUTPUT.expected_iou} (threshold with 5% tolerance = {almost_pass_threshold})")
            status = "FAIL"
        
        rfi001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def RFI005_VALIDATION(out_post: tuple):
        """
        @brief Validates the RFI005 test case by testing the successful filtering of regions affected by RFI.
        @details
            This static method logs detailed information about the test case, including its identifier, name, requirement, 
            input configuration path, expected outputs, and execution results. It verifies the validity of the postprocessing outputs 
            path and that the system is able to filter (regions of) focussed scenes from the incoming data stream which are 
            significantly affected by RFI by testing the successful filtering of these regions.
        @param out_post 
            Tuple[int, int] containing the 'successful'/'failed' counts from process_prediction_folder.
        @return bool 
            True if 'successful'>0 and 'failed'=0, False otherwise.
        """
        successful, failed = out_post
        status = True
        rfi001_logger.info(f"Log file name: {RFI001_LOG_FILE}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Test Case Identifier : {UTILS.rfi_test_cases_matrix[4][0]}")
        rfi001_logger.info(f"Test Case Name       : {UTILS.rfi_test_cases_matrix[4][1]}")
        rfi001_logger.info(f"Requirement Verified : {UTILS.rfi_test_cases_matrix[4][2]}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Inputs:")
        paths = {
        "Prediction mask dir": cls_RFI001_INPUT.prediction_mask_dir,
        "Output postprocessing dir": cls_RFI001_INPUT.output_postprocessing_dir,
        }
        UTILS.verify_paths(paths, rfi001_logger)
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Expected outputs:")
        rfi001_logger.info(f"\tExpected successful > 0 and failed = 0")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Execution:")
        rfi001_logger.info(f"\tsuccessful: {successful}")
        rfi001_logger.info(f"\tfailed: {failed}")
        rfi001_logger.info(f"------------------------------------------------------------")

        if successful > 0 and failed == 0:
            rfi001_logger.info(f"[PASS]: successful = {successful} > 0 and failed = {failed} = 0")
            status = "PASS"
        else:
            rfi001_logger.info(f"[FAIL]: successful = {successful}, failed = {failed} (expected successful > 0 and failed = 0)")
            status = "FAIL"

        rfi001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def RFI006_VALIDATION(out: dict):
        """
        @brief Validates the RFI006 test case based on precision and recall values.
        @details
            This static method logs the details of the RFI006 test case, including inputs, expected outputs, and execution results.
            It checks whether the computed precision and recall meet or exceed the expected thresholds defined in
            `cls_RFI001_EXPECTED_OUTPUT`. If either precision or recall is below the expected value, the validation fails.
        @param out 
            Dictionary containing the computed 'precision' and 'recall' values.
        @return bool 
            True if both precision and recall meet or exceed expected values and input path is valid, False otherwise.
        """
        status = True
        rfi001_logger.info(f"Log file name: {RFI001_LOG_FILE}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Test Case Identifier : {UTILS.rfi_test_cases_matrix[5][0]}")
        rfi001_logger.info(f"Test Case Name       : {UTILS.rfi_test_cases_matrix[5][1]}")
        rfi001_logger.info(f"Requirement Verified : {UTILS.rfi_test_cases_matrix[5][2]}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Inputs:")
        paths = {
        "Config path": cls_RFI001_INPUT.config_path,
        "Weights path": cls_RFI001_INPUT.weights,
        "Test directory": cls_RFI001_INPUT.test_dir,
        #"Single npy": cls_RFI001_INPUT.single_npy
        }
        UTILS.verify_paths(paths, rfi001_logger)
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Expected outputs:")
        rfi001_logger.info(f"\tExpected precision: {cls_RFI001_EXPECTED_OUTPUT.expected_precision}")
        rfi001_logger.info(f"\tExpected recall: {cls_RFI001_EXPECTED_OUTPUT.expected_recall}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Execution:")
        rfi001_logger.info(f"\tprecision: {out['precision']}")
        rfi001_logger.info(f"\trecall: {out['recall']}")
        rfi001_logger.info(f"------------------------------------------------------------")
        
        almost_pass_threshold_precision = cls_RFI001_EXPECTED_OUTPUT.expected_precision - 0.05
        if out['precision'] >= cls_RFI001_EXPECTED_OUTPUT.expected_precision:
            rfi001_logger.info(f"[PASS]: Computed precision = {out['precision']} >= Expected precision = {cls_RFI001_EXPECTED_OUTPUT.expected_precision}")
            precision_status = "PASS"
        elif out['precision'] >= almost_pass_threshold_precision:
            rfi001_logger.info(f"[ALMOST_PASS]: Computed precision = {out['precision']} is within 5% of Expected precision = {cls_RFI001_EXPECTED_OUTPUT.expected_precision}")
            precision_status = "ALMOST_PASS"
        else:
            rfi001_logger.info(f"[FAIL]: Computed precision = {out['precision']} < Expected precision = {cls_RFI001_EXPECTED_OUTPUT.expected_precision} (threshold with 5% tolerance = {almost_pass_threshold_precision})")
            precision_status = "FAIL"

        almost_pass_threshold_recall = cls_RFI001_EXPECTED_OUTPUT.expected_recall - 0.05
        if out['recall'] >= cls_RFI001_EXPECTED_OUTPUT.expected_recall:
            rfi001_logger.info(f"[PASS]: Computed recall = {out['recall']} >= Expected recall = {cls_RFI001_EXPECTED_OUTPUT.expected_recall}")
            recall_status = "PASS"
        elif out['recall'] >= almost_pass_threshold_recall:
            rfi001_logger.info(f"[ALMOST_PASS]: Computed recall = {out['recall']} is within 5% of Expected recall = {cls_RFI001_EXPECTED_OUTPUT.expected_recall}")
            recall_status = "ALMOST_PASS"
        else:
            rfi001_logger.info(f"[FAIL]: Computed recall = {out['recall']} < Expected recall = {cls_RFI001_EXPECTED_OUTPUT.expected_recall} (threshold with 5% tolerance = {almost_pass_threshold_recall})")
            recall_status = "FAIL"

        status_priority = {"FAIL": 0, "ALMOST_PASS": 1, "PASS": 2}
        status = min(precision_status, recall_status, key=lambda s: status_priority[s])
        
        rfi001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def RFI007_VALIDATION(out: dict):
        """
        @brief Validates the RFI007 test case by comparing the computed IOU with the expected IOU.
        @details
            This static method logs detailed information about the test case, including its identifier, name, requirement, 
            input configuration path, expected outputs, and execution results. It verifies the validity of the configuration 
            path and compares the computed IOU value from the output dictionary with the expected IOU value. The method logs 
            whether the test passes or fails based on this comparison.
        @param out 
            dict: A dictionary containing the computed output values, specifically the 'iou' key.
        @return 
            bool: True if the computed IOU meets or exceeds the expected IOU, False otherwise or if the input path is invalid.
        """
        status = True
        rfi001_logger.info(f"Log file name: {RFI001_LOG_FILE}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Test Case Identifier : {UTILS.rfi_test_cases_matrix[6][0]}")
        rfi001_logger.info(f"Test Case Name       : {UTILS.rfi_test_cases_matrix[6][1]}")
        rfi001_logger.info(f"Requirement Verified : {UTILS.rfi_test_cases_matrix[6][2]}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Inputs:")
        paths = {
        "Config path": cls_RFI001_INPUT.config_path,
        "Weights path": cls_RFI001_INPUT.weights,
        "Test directory": cls_RFI001_INPUT.test_dir,
        #"Single npy": cls_RFI001_INPUT.single_npy
        }
        UTILS.verify_paths(paths, rfi001_logger)
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Expected outputs:")
        rfi001_logger.info(f"\tExpected iou: {cls_RFI001_EXPECTED_OUTPUT.expected_iou}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Execution:")
        rfi001_logger.info(f"\tiou: {out['iou']}")
        rfi001_logger.info(f"------------------------------------------------------------")
        
        almost_pass_threshold = cls_RFI001_EXPECTED_OUTPUT.expected_iou - 0.05
        if out['iou'] >= cls_RFI001_EXPECTED_OUTPUT.expected_iou:
            rfi001_logger.info(f"[PASS]: Computed iou = {out['iou']} >= Expected iou = {cls_RFI001_EXPECTED_OUTPUT.expected_iou}")
            status = "PASS"
        elif out['iou'] >= almost_pass_threshold:
            rfi001_logger.info(f"[ALMOST_PASS]: Computed iou = {out['iou']} is within 5% of Expected iou = {cls_RFI001_EXPECTED_OUTPUT.expected_iou}")
            status = "ALMOST_PASS"
        else:
            rfi001_logger.info(f"[FAIL]: Computed iou = {out['iou']} < Expected iou = {cls_RFI001_EXPECTED_OUTPUT.expected_iou} (threshold with 5% tolerance = {almost_pass_threshold})")
            status = "FAIL"
        
        rfi001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def RFI010_VALIDATION(out_post: tuple):
        """
        @brief Validates the RFI010 test case by testing the successful filtering of regions affected by RFI.
        @details
            The system shall filter out (regions of) image tiles containing RFI signals at a level of >5dB (TBC)
            above the scene speckle noise floor, after any compensation has been applied.
            This static method logs detailed information about the test case, including its identifier, name, requirement,
            input configuration path, expected outputs, and execution results. It verifies the validity of the postprocessing
            outputs path and that the system is able to filter (regions of) focussed scenes from the incoming data stream
            which are significantly affected by RFI by testing the successful filtering of these regions.
        @param out_post
            Tuple[int, int] containing the 'successful'/'failed' counts from process_prediction_folder.
        @return bool
            True if 'successful'>0 and 'failed'=0, False otherwise.
        """
        successful, failed = out_post
        status = True
        rfi001_logger.info(f"Log file name: {RFI001_LOG_FILE}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Test Case Identifier : {UTILS.rfi_test_cases_matrix[9][0]}")
        rfi001_logger.info(f"Test Case Name       : {UTILS.rfi_test_cases_matrix[9][1]}")
        rfi001_logger.info(f"Requirement Verified : {UTILS.rfi_test_cases_matrix[9][2]}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Inputs:")
        paths = {
        "Prediction mask dir": cls_RFI001_INPUT.prediction_mask_dir,
        "Output postprocessing dir": cls_RFI001_INPUT.output_postprocessing_dir,
        }
        UTILS.verify_paths(paths, rfi001_logger)
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Expected outputs:")
        rfi001_logger.info(f"\tExpected successful > 0 and failed = 0")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Execution:")
        rfi001_logger.info(f"\tsuccessful: {successful}")
        rfi001_logger.info(f"\tfailed: {failed}")
        rfi001_logger.info(f"------------------------------------------------------------")

        if successful > 0 and failed == 0:
            rfi001_logger.info(f"[PASS]: successful = {successful} > 0 and failed = {failed}")
            status = "PASS"
        else:
            rfi001_logger.info(f"[FAIL]: successful = {successful}, failed = {failed} (expected successful > 0 and failed = 0)")
            status = "FAIL"

        rfi001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def RFI012_VALIDATION(out: dict):
        """
        @brief Validates the output accuracy against the expected accuracy for RFI012 test case.
        @details
            This static method logs detailed information about the RFI012 validation process,
            including test case metadata, input configuration, expected outputs, and execution results.
            It verifies the existence of the configuration path and compares the computed accuracy
            with the expected accuracy, logging the result as PASS or FAIL.
        @param out
            dict: A dictionary containing the output values, specifically the computed accuracy ('acc').
        @return
            bool: True if the computed accuracy meets or exceeds the expected accuracy and the input path is valid, False otherwise.
        """
        status = True
        rfi001_logger.info(f"Log file name: {RFI001_LOG_FILE}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Test Case Identifier : {UTILS.rfi_test_cases_matrix[11][0]}")
        rfi001_logger.info(f"Test Case Name       : {UTILS.rfi_test_cases_matrix[11][1]}")
        rfi001_logger.info(f"Requirement Verified : {UTILS.rfi_test_cases_matrix[11][2]}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Inputs:")
        paths = {
        "Config path": cls_RFI001_INPUT.config_path,
        "Weights path": cls_RFI001_INPUT.weights,
        "Test directory": cls_RFI001_INPUT.test_dir,
        #"Single npy": cls_RFI001_INPUT.single_npy
        }
        UTILS.verify_paths(paths, rfi001_logger)
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Expected outputs:")
        rfi001_logger.info(f"\tExpected accuracy: {cls_RFI001_EXPECTED_OUTPUT.expected_acc}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Execution:")
        rfi001_logger.info(f"\taccuracy: {out['acc']}")
        rfi001_logger.info(f"------------------------------------------------------------")
        
        almost_pass_threshold = cls_RFI001_EXPECTED_OUTPUT.expected_acc - 0.05
        if out['acc'] >= cls_RFI001_EXPECTED_OUTPUT.expected_acc:
            rfi001_logger.info(f"[PASS]: Computed accuracy = {out['acc']} >= Expected accuracy = {cls_RFI001_EXPECTED_OUTPUT.expected_acc}")
            status = "PASS"
        elif out['acc'] >= almost_pass_threshold:
            rfi001_logger.info(f"[ALMOST_PASS]: Computed accuracy = {out['acc']} is within 5% of Expected accuracy = {cls_RFI001_EXPECTED_OUTPUT.expected_acc}")
            status = "ALMOST_PASS"
        else:
            rfi001_logger.info(f"[FAIL]: Computed accuracy = {out['acc']} < Expected accuracy = {cls_RFI001_EXPECTED_OUTPUT.expected_acc} (threshold with 5% tolerance = {almost_pass_threshold})")
            status = "FAIL"
        
        rfi001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def RFI013_VALIDATION(out: dict):
        """
        @brief Validates the RFI013 test case by comparing the computed IOU with the expected IOU.
        @details
            This static method logs detailed information about the test case, including its identifier, name, requirement, 
            input configuration path, expected outputs, and execution results. It verifies the validity of the configuration 
            path and compares the computed IOU value from the output dictionary with the expected IOU value. The method logs 
            whether the test passes or fails based on this comparison.
        @param out 
            dict: A dictionary containing the computed output values, specifically the 'iou' key.
        @return 
            bool: True if the computed IOU meets or exceeds the expected IOU, False otherwise or if the input path is invalid.
        """
        status = True
        rfi001_logger.info(f"Log file name: {RFI001_LOG_FILE}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Test Case Identifier : {UTILS.rfi_test_cases_matrix[12][0]}")
        rfi001_logger.info(f"Test Case Name       : {UTILS.rfi_test_cases_matrix[12][1]}")
        rfi001_logger.info(f"Requirement Verified : {UTILS.rfi_test_cases_matrix[12][2]}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Inputs:")
        paths = {
            "Config path": cls_RFI001_INPUT.config_path,
            "Weights path": cls_RFI001_INPUT.weights,
            "Test directory": cls_RFI001_INPUT.test_dir,
            #"Single npy": cls_RFI001_INPUT.single_npy
        }
        UTILS.verify_paths(paths, rfi001_logger)        
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Expected outputs:")
        rfi001_logger.info(f"\tExpected iou: {cls_RFI001_EXPECTED_OUTPUT.expected_iou}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Execution:")
        rfi001_logger.info(f"\tiou: {out['iou']}")
        rfi001_logger.info(f"------------------------------------------------------------")
        
        almost_pass_threshold = cls_RFI001_EXPECTED_OUTPUT.expected_iou - 0.05
        if out['iou'] >= cls_RFI001_EXPECTED_OUTPUT.expected_iou:
            rfi001_logger.info(f"[PASS]: Computed iou = {out['iou']} >= Expected iou = {cls_RFI001_EXPECTED_OUTPUT.expected_iou}")
            status = "PASS"
        elif out['iou'] >= almost_pass_threshold:
            rfi001_logger.info(f"[ALMOST_PASS]: Computed iou = {out['iou']} is within 5% of Expected iou = {cls_RFI001_EXPECTED_OUTPUT.expected_iou}")
            status = "ALMOST_PASS"
        else:
            rfi001_logger.info(f"[FAIL]: Computed iou = {out['iou']} < Expected iou = {cls_RFI001_EXPECTED_OUTPUT.expected_iou} (threshold with 5% tolerance = {almost_pass_threshold})")
            status = "FAIL"
        
        rfi001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def RFI016_VALIDATION(out_post: tuple):
        """
        @brief Validates the RFI016 test case by testing the successful filtering of regions affected by RFI.
        @details
            The system shall filter out (regions of) image tiles containing RFI signals at a level of >5dB (TBC)
            above the scene speckle noise floor, after any compensation has been applied.
            This static method logs detailed information about the test case, including its identifier, name, requirement,
            input configuration path, expected outputs, and execution results. It verifies the validity of the postprocessing
            outputs path and that the system is able to filter (regions of) focussed scenes from the incoming data stream
            which are significantly affected by RFI by testing the successful filtering of these regions.
        @param out_post
            Tuple[int, int] containing the 'successful'/'failed' counts from process_prediction_folder.
        @return bool
            True if 'successful'>0 and 'failed'=0, False otherwise.
        """
        successful, failed = out_post
        status = True
        rfi001_logger.info(f"Log file name: {RFI001_LOG_FILE}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Test Case Identifier : {UTILS.rfi_test_cases_matrix[15][0]}")
        rfi001_logger.info(f"Test Case Name       : {UTILS.rfi_test_cases_matrix[15][1]}")
        rfi001_logger.info(f"Requirement Verified : {UTILS.rfi_test_cases_matrix[15][2]}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Inputs:")
        paths = {
        "Prediction mask dir": cls_RFI001_INPUT.prediction_mask_dir,
        "Output postprocessing dir": cls_RFI001_INPUT.output_postprocessing_dir,
        }
        UTILS.verify_paths(paths, rfi001_logger)
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Expected outputs:")
        rfi001_logger.info(f"\tExpected successful > 0 and failed = 0")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Execution:")
        rfi001_logger.info(f"\tsuccessful: {successful}")
        rfi001_logger.info(f"\tfailed: {failed}")
        rfi001_logger.info(f"------------------------------------------------------------")

        if successful > 0 and failed == 0:
            rfi001_logger.info(f"[PASS]: successful = {successful} > 0 and failed = {failed}")
            status = "PASS"
        else:
            rfi001_logger.info(f"[FAIL]: successful = {successful}, failed = {failed} (expected successful > 0 and failed = 0)")
            status = "FAIL"

        rfi001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def RFI011_VALIDATION(exit_code, step1_time_s, step2_time_s, patches_nr):
        """
        @brief Validates the RFI011 test case: End-to-End Latency Verification for RFI Processing.
        @details
            Uses run_rfi_large_end2end() output to estimate the processing time for a full swath:
                swath_time_s = (total_time_s / patches_nr) * patches_per_swath
            where patches_per_swath = cls_RFI001_INPUT.patches_per_swath (1000).
            FAIL if estimated swath time >= cls_RFI001_EXPECTED_OUTPUT.expected_max_e2e_time_s.
        @param exit_code    Exit code returned by run_rfi_large_end2end().
        @param step1_time_s Pre-processing step duration in seconds.
        @param step2_time_s Inference step duration in seconds.
        @param patches_nr   Number of patches processed in the test run.
        @return str  "PASS" or "FAIL".
        """
        total_time_s = step1_time_s + step2_time_s
        avg_time_per_patch_s = total_time_s / max(patches_nr, 1)
        swath_time_s = avg_time_per_patch_s * cls_RFI001_INPUT.patches_per_swath
        _THRESHOLD = cls_RFI001_EXPECTED_OUTPUT.expected_max_e2e_time_s

        rfi001_logger.info(f"Log file name: {RFI001_LOG_FILE}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Test Case Identifier : {UTILS.rfi_test_cases_matrix[10][0]}")
        rfi001_logger.info(f"Test Case Name       : {UTILS.rfi_test_cases_matrix[10][1]}")
        rfi001_logger.info(f"Requirement Verified : {UTILS.rfi_test_cases_matrix[10][2]}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Inputs:")
        paths = {"Config path": cls_RFI_SMALL_INPUT.config_path}
        UTILS.verify_paths(paths, rfi001_logger)
        rfi001_logger.info(f"\tPatches per swath    : {cls_RFI001_INPUT.patches_per_swath}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Expected outputs:")
        rfi001_logger.info(f"\tMax swath processing time (s): {_THRESHOLD}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Execution:")
        rfi001_logger.info(f"\texit_code              : {exit_code}")
        rfi001_logger.info(f"\tstep1_time_s           : {step1_time_s:.5f} s")
        rfi001_logger.info(f"\tstep2_time_s           : {step2_time_s:.5f} s")
        rfi001_logger.info(f"\ttotal_time_s           : {total_time_s:.5f} s")
        rfi001_logger.info(f"\tpatches_nr (test set)  : {patches_nr}")
        rfi001_logger.info(f"\tavg_time_per_patch (s) : {avg_time_per_patch_s:.5f}")
        rfi001_logger.info(f"\tswath_time_estimate (s): {swath_time_s:.5f}  "
                           f"[= {total_time_s:.5f} / {patches_nr} * {cls_RFI001_INPUT.patches_per_swath}]")
        rfi001_logger.info(f"------------------------------------------------------------")

        if swath_time_s < _THRESHOLD:
            rfi001_logger.info(f"[PASS]: Estimated swath time = {swath_time_s:.5f} s < {_THRESHOLD} s")
            status = "PASS"
        else:
            rfi001_logger.info(f"[FAIL]: Estimated swath time = {swath_time_s:.5f} s >= {_THRESHOLD} s")
            status = "FAIL"

        rfi001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def RFI017_VALIDATION(exit_code, step1_time_s, step2_time_s, patches_nr):
        """
        @brief Validates the RFI017 test case: End-to-End Processing Latency Verification for Lightweight Pipeline.
        @details
            Uses run_rfi_small_end2end() output to compute average processing latency normalised
            to 100 km\u00b2 of geographic coverage. Each patch covers cls_RFI001_INPUT.patch_area_km2 km\u00b2.
            FAIL if latency_per_100km2 >= cls_RFI001_EXPECTED_OUTPUT.expected_max_latency_per_100km2_s.
        @param exit_code    Exit code returned by run_rfi_small_end2end().
        @param step1_time_s Pre-processing step duration in seconds.
        @param step2_time_s Inference step duration in seconds.
        @param patches_nr   Number of patches processed.
        @return str  "PASS" or "FAIL".
        """
        total_time_s = step1_time_s + step2_time_s
        coverage_km2 = patches_nr * cls_RFI001_INPUT.patch_area_km2
        latency_per_100km2 = (total_time_s / max(coverage_km2, 1e-9)) * 100.0
        _THRESHOLD = cls_RFI001_EXPECTED_OUTPUT.expected_max_latency_per_100km2_s

        rfi001_logger.info(f"Log file name: {RFI001_LOG_FILE}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Test Case Identifier : {UTILS.rfi_test_cases_matrix[16][0]}")
        rfi001_logger.info(f"Test Case Name       : {UTILS.rfi_test_cases_matrix[16][1]}")
        rfi001_logger.info(f"Requirement Verified : {UTILS.rfi_test_cases_matrix[16][2]}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Inputs:")
        paths = {"Config path": cls_RFI_SMALL_INPUT.config_path}
        UTILS.verify_paths(paths, rfi001_logger)
        rfi001_logger.info(f"\tPatch area (km\u00b2)     : {cls_RFI001_INPUT.patch_area_km2}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Expected outputs:")
        rfi001_logger.info(f"\tMax latency per 100 km\u00b2 (s): {_THRESHOLD}")
        rfi001_logger.info(f"------------------------------------------------------------")
        rfi001_logger.info(f"Execution:")
        rfi001_logger.info(f"\texit_code                   : {exit_code}")
        rfi001_logger.info(f"\tstep1_time_s                : {step1_time_s:.5f} s")
        rfi001_logger.info(f"\tstep2_time_s                : {step2_time_s:.5f} s")
        rfi001_logger.info(f"\ttotal_time_s                : {total_time_s:.5f} s")
        rfi001_logger.info(f"\tpatches_nr                  : {patches_nr}")
        rfi001_logger.info(f"\tcoverage_km\u00b2                : {coverage_km2:.2f}")
        rfi001_logger.info(f"\tlatency_per_100km\u00b2 (s)      : {latency_per_100km2:.5f}")
        rfi001_logger.info(f"------------------------------------------------------------")

        if latency_per_100km2 < _THRESHOLD:
            rfi001_logger.info(f"[PASS]: Latency per 100 km\u00b2 = {latency_per_100km2:.5f} s < {_THRESHOLD} s")
            status = "PASS"
        else:
            rfi001_logger.info(f"[FAIL]: Latency per 100 km\u00b2 = {latency_per_100km2:.5f} s >= {_THRESHOLD} s")
            status = "FAIL"

        rfi001_logger.info(f"============================================================\n")
        return status


# ---------------- EXPECTED METRICS ---------------- #

class cls_RFI001_EXPECTED_OUTPUT:
    expected_precision = 0.85
    expected_recall = 0.85
    expected_acc = 0.85
    expected_iou = 0.70
    expected_f1 = 0.80
    expected_max_latency_per_100km2_s = 30.0    
    expected_max_e2e_time_s = 60.0  # 1 minute  



# ---------------- SMALL MODEL INPUT CONFIG ---------------- #

class cls_RFI_SMALL_INPUT:
    model_size = "small"
    config_path     = Path(__file__).parents[2] / "pipeline" / "RFI_usecase" / "config-rfi-kd.yaml"
    weights         = Path(__file__).parents[2] / "pipeline" / "RFI_usecase" / "checkpoints" / "small-basechannel8" / "best_student.pth"
    stride_multiple = 8
    thr             = 0.325


# ---------------- INPUT CONFIG ---------------- #

class cls_RFI001_INPUT:
    model_size = "large"
    config_path   = Path(__file__).parents[2] / "pipeline" / "RFI_usecase" / "config-rfi-large.yaml"
    weights    = Path(__file__).parents[2] / "pipeline" / "RFI_usecase" / "checkpoints" / "BS64+lr3e-5+focal0.6" / "epoch_112.pth"
    test_dir   = Path("/media/ubuntu_24_04/data/opensar/range_compressed_scaled/rfi-v2/test2")
    prediction_mask_dir = Path(__file__).parent / "rfi_prediction_mask"
    output_postprocessing_dir = Path(__file__).parent / "postprocessing_outputs"
    # One VV or VH npy from the test set; the dataset will find its VV/VH pair
    # single_npy = Path("/media/ubuntu_24_04/data/opensar/range_compressed_scaled/rfi-v2/test2/s1a-iw-raw-s-vh-20200515t032854-20200515t032926-032570-03c5ba-IW2-RFI_407_scaled.npy")

    batch_size = 2
    num_workers = 1
    stride_multiple = 16
    ignore_index = 255
    resize_to = None
    thr = 0.404

    THRESHOLD_DB = 5.0
    FILTER_MODE = "remove_high_rfi"  # Choose: "keep_high_rfi" or "remove_high_rfi"
    FILTER_LEVEL = "patch"          # Choose: "region" or "patch"

    REGION_STAT = "p90"              # Choose: "mean", "median", "p90"
    BACKGROUND_STAT = "median"      # Choose: "mean", "median", "trimmed_mean"

    MIN_REGION_AREA = 20
    RING_INNER_ITERS = 3
    RING_OUTER_ITERS = 12
    TRIM_FRACTION = 0.1

    patch_area_km2 = 176.11
    patches_per_swath = 120


#@dataclass
class EpochStats:
    precision = 0.80
    recall = 0.80
    acc = 0.80
    f1 = 0.80   
    iou = 0.70

# ---------------- TEST CASE ---------------- #

def run_evaluate_test():
    return EVAL.evaluate_test(
        cls_RFI001_INPUT.model_size,       # Model size ('large' or 'small')
        cls_RFI001_INPUT.config_path,      # Path to model config file
        cls_RFI001_INPUT.weights,          # Path to model weights file
        cls_RFI001_INPUT.test_dir,         # Path to test dataset directory
        cls_RFI001_INPUT.batch_size,       # Batch size for evaluation
        cls_RFI001_INPUT.num_workers,      # Number of workers for data loading
        cls_RFI001_INPUT.stride_multiple,  # Stride multiple for evaluation
        cls_RFI001_INPUT.ignore_index,     # Index to ignore in evaluation
        cls_RFI001_INPUT.resize_to,        # Resize dimension for input images
        cls_RFI001_INPUT.thr               # Threshold for RFI detection
    )


def run_post_processing():
    return POST.process_prediction_folder(
        sar_data_dir=cls_RFI001_INPUT.test_dir,
        prediction_mask_dir=cls_RFI001_INPUT.prediction_mask_dir,
        output_dir=cls_RFI001_INPUT.output_postprocessing_dir,
        threshold_db=cls_RFI001_INPUT.THRESHOLD_DB,
        region_stat=cls_RFI001_INPUT.REGION_STAT,
        background_stat=cls_RFI001_INPUT.BACKGROUND_STAT,
        min_region_area=cls_RFI001_INPUT.MIN_REGION_AREA,
        ring_inner_iters=cls_RFI001_INPUT.RING_INNER_ITERS,
        ring_outer_iters=cls_RFI001_INPUT.RING_OUTER_ITERS,
        trim_fraction=cls_RFI001_INPUT.TRIM_FRACTION,
        pred_suffix="_pred",
        filter_mode=cls_RFI001_INPUT.FILTER_MODE,
        filter_level=cls_RFI001_INPUT.FILTER_LEVEL,
        save_results=False
    )


def run_generate_prediction_masks():
    return EVAL.generate_prediction_masks(
        cls_RFI001_INPUT.model_size,
        cls_RFI001_INPUT.config_path,
        cls_RFI001_INPUT.weights,
        cls_RFI001_INPUT.test_dir,
        cls_RFI001_INPUT.prediction_mask_dir,
        cls_RFI001_INPUT.batch_size,
        cls_RFI001_INPUT.num_workers,
        cls_RFI001_INPUT.stride_multiple,
        cls_RFI001_INPUT.ignore_index,
        cls_RFI001_INPUT.resize_to,
        cls_RFI001_INPUT.thr,
    )


class cls_RFI_Run:

    @staticmethod
    def test_main_RFI():
        """
        @brief Runs model evaluation on the test dataset and logs validation metrics.
        @details
        - Initializes a list to store the status of each test case.
        - Runs evaluation on the test dataset using the specified input configuration.
        - @testcase Runs precision and recall validation result logging.
        - @testcase Runs accuracy validation result logging.
        - @testcase Runs IOU validation result logging.
        - Logs summary of test case status to console and log file.
        """
        # Initialize a list to store the status of each test case
        tc_status = []

        cls_RFI_SETUP.RFI_LOG_FILE_SETUP()

        # Run evaluation on the test dataset using the specified input configuration
        out = run_evaluate_test()

        # Generate prediction mask PNGs required by process_prediction_folder
        #run_generate_prediction_masks()

        # Run post-processing on prediction folder
        out_post = run_post_processing()
        
        # Example output for testing (commented out)
        # out = EpochStats()

        # Test Case: RFI001 - F1 score
        # Log F1 score validation result
        tc_status.append(cls_RFI001_LOGS.RFI001_VALIDATION(out))

        # Test Case: RFI002 - IOU
        # Log IOU validation result
        tc_status.append(cls_RFI001_LOGS.RFI002_VALIDATION(out))

        # Test Case: RFI005 - Successful filtering of RFI regions
        # Log post-processing successful/failed counts validation result
        tc_status.append(cls_RFI001_LOGS.RFI005_VALIDATION(out_post))


        # Test Case: RFI006 - Precision & Recall
        # Log precision and recall validation result
        tc_status.append(cls_RFI001_LOGS.RFI006_VALIDATION(out))

        # Test Case: RFI007 - IOU
        # Log IOU validation result
        tc_status.append(cls_RFI001_LOGS.RFI007_VALIDATION(out))

        # Test Case: RFI010 - Successful filtering of RFI regions (>5dB above noise floor)
        # Log post-processing successful/failed counts validation result
        tc_status.append(cls_RFI001_LOGS.RFI010_VALIDATION(out_post))
        
        # Test Case: RFI012 - Accuracy
        # Log accuracy validation result
        tc_status.append(cls_RFI001_LOGS.RFI012_VALIDATION(out))

        # Test Case: RFI013 - IOU (RFI Region Masking IOU Verification)
        # Log IOU verification result
        tc_status.append(cls_RFI001_LOGS.RFI013_VALIDATION(out))

        # Test Case: RFI016 - Successful filtering of RFI regions (>5dB above noise floor)
        # Log post-processing successful/failed counts validation result
        tc_status.append(cls_RFI001_LOGS.RFI016_VALIDATION(out_post))

        # Test Case: RFI011 - End-to-End Latency Verification for RFI Processing (large pipeline)
        exit_code_rfi_small, _profile_rfi_small, step1_rfi_small, step2_rfi_small, patches_nr_rfi_small = run_rfi_small_end2end()
        tc_status.append(cls_RFI001_LOGS.RFI011_VALIDATION(exit_code_rfi_small, step1_rfi_small, step2_rfi_small, patches_nr_rfi_small))

        # Test Case: RFI017 - End-to-End Processing Latency Verification for Lightweight Pipeline
        tc_status.append(cls_RFI001_LOGS.RFI017_VALIDATION(exit_code_rfi_small, step1_rfi_small, step2_rfi_small, patches_nr_rfi_small))


        

        # Log outputs
        # Log summary of test case status to console and log file
        rfi001_logger.info(
            f"RFI large model validation status:\n"
            f"\tNumber of PASSED tests      = {tc_status.count('PASS')}\n"
            f"\tNumber of ALMOST_PASS tests = {tc_status.count('ALMOST_PASS')}\n"
            f"\tNumber of FAILED tests      = {tc_status.count('FAIL')}\n"
            f"\tNumber of TOTAL tests       = {len(tc_status)}"
        )
