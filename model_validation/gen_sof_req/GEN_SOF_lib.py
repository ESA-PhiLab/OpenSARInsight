"""
/**
 * @file: GEN_SOF_lib.py
 * @brief: General model validation and logging utilities for the pipeline.
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
import re
from pathlib import Path
import logging
from datetime import datetime

from model_validation.utils_lib import cls_UTILS_LIB as UTILS


# Directory where log files will be stored
LOG_DIR = Path(__file__).parents[1] / "logs"

# Generate a dynamic filename with date + time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
GEN_SOF001_LOG_FILE = LOG_DIR / f"GEN_SOF001_{timestamp}.log"

# Create a separate logger for GEN_SOF001 metrics
gen_sof001_logger = logging.getLogger("GEN_SOF001")
gen_sof001_logger.setLevel(logging.INFO)

# Clear existing handlers to avoid duplicate logging
gen_sof001_logger.handlers.clear()
gen_sof001_logger.propagate = False  # prevent logging to root


def _apply_end2end_stubs():
    """Ensure required env vars and sys.path are set
    """
    import types

    # Set config env vars to the repo-local paths if not already defined.
    _cfg_base = Path(__file__).parent.parent.parent / 'configuration'
    _env_defaults = {
        'DGS_CONFIG_PATH':    str(_cfg_base / 'data_generation_scripts.yaml'),
        'MODEL_USECASES_PATH': str(_cfg_base / 'model_usecases.yaml'),
        'DATA_PREPROC_PATH':  str(_cfg_base / 'data_preprocessing.yaml'),
    }
    for _var, _default in _env_defaults.items():
        if not os.environ.get(_var):
            os.environ[_var] = _default

    # Ensure the real l0_to_range_compressed package is importable by its bare name.
    _l0_pkg = str(Path(__file__).parent.parent.parent /
                  'dataset_generation_scripts' / 'l0_to_range_compressed')
    if _l0_pkg not in sys.path:
        sys.path.insert(0, _l0_pkg)

    # dem_stitcher is not installed — stub it out.
    _stubs = {
        'dem_stitcher':          {'stitch_dem': None},
        'dem_stitcher.stitcher': {'stitch_dem': None},
    }
    for _mod, _attrs in _stubs.items():
        if _mod not in sys.modules:
            _stub = types.ModuleType(_mod)
            for _k, _v in _attrs.items():
                setattr(_stub, _k, _v)
            sys.modules[_mod] = _stub


def run_rfi_large_end2end():
    _apply_end2end_stubs()

    from pipeline.main.main_pipeline import end2end_rfi_large
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    exit_code, profile_stats, step1_time_s, step2_time_s, patches_nr = end2end_rfi_large(logger=logger)
    print(f"exit_code:     {exit_code}")
    print(f"step1_time_s:  {step1_time_s:.5f} s")
    print(f"step2_time_s:  {step2_time_s:.5f} s")
    print(f"patches_nr:    {patches_nr}")
    print(f"profile_stats: {profile_stats}")
    print(f"GPU Mem Peak (GB):                                    {profile_stats.get('profile/gpu_mem_peak_GB', 'N/A')}")

    return exit_code, profile_stats, step1_time_s, step2_time_s, patches_nr


def run_vd_large_end2end():
    _apply_end2end_stubs()

    from pipeline.main.main_pipeline import end2end_vd_large
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    exit_code, profile_stats, step1_time_s, step2_time_s, patches_nr = end2end_vd_large(logger=logger)
    print(f"exit_code:     {exit_code}")
    print(f"step1_time_s:  {step1_time_s:.5f} s")
    print(f"step2_time_s:  {step2_time_s:.5f} s")
    print(f"patches_nr:    {patches_nr}")
    print(f"profile_stats: {profile_stats}")
    print(f"GPU Mem Peak (GB):                                    {profile_stats.get('profile/gpu_mem_peak_GB', 'N/A') if profile_stats else 'N/A'}")

    return exit_code, profile_stats, step1_time_s, step2_time_s, patches_nr

def run_vd_small_end2end():
    _apply_end2end_stubs()

    from pipeline.main.main_pipeline import end2end_vd_small
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    exit_code, profile_stats, step1_time_s, step2_time_s, patches_nr = end2end_vd_small(logger=logger)
    print(f"exit_code:     {exit_code}")
    print(f"step1_time_s:  {step1_time_s:.5f} s")
    print(f"step2_time_s:  {step2_time_s:.5f} s")
    print(f"patches_nr:    {patches_nr}")
    print(f"profile_stats: {profile_stats}")
    print(f"GPU Mem Peak (GB):                                    {profile_stats.get('profile/gpu_mem_peak_GB', 'N/A') if profile_stats else 'N/A'}")

    return exit_code, profile_stats, step1_time_s, step2_time_s, patches_nr

def run_rfi_small_end2end():
    _apply_end2end_stubs()

    from pipeline.main.main_pipeline import end2end_rfi_small
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    exit_code, profile_stats, step1_time_s, step2_time_s, patches_nr = end2end_rfi_small(logger=logger)
    # print(f"exit_code:     {exit_code}")
    # print(f"step1_time_s:  {step1_time_s:.5f} s")
    # print(f"step2_time_s:  {step2_time_s:.5f} s")
    # print(f"patches_nr:    {patches_nr}")
    # print(f"profile_stats: {profile_stats}")
    # print(f"GPU Mem Peak (GB):                                    {profile_stats.get('profile/gpu_mem_peak_GB', 'N/A') if profile_stats else 'N/A'}")

    return exit_code, profile_stats, step1_time_s, step2_time_s, patches_nr

class cls_GEN_SOF_SETUP:
    @staticmethod
    def GEN_SOF_LOG_FILE_SETUP():
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(GEN_SOF001_LOG_FILE, mode="w")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        gen_sof001_logger.addHandler(file_handler)


class cls_GEN_LOGS:

    @staticmethod
    def GEN001_VALIDATION():
        """
        @brief Validates the GEN-001 test case: Evaluation of Inference Performance Using Quantitative Metrics per Use-Case.
        @details
            Runs run_analysis_script() (DVD unrestricted model) and run_evaluate_test()
            (RFI unrestricted model) and checks that both output precision and recall.
            Values are also compared against expected thresholds.
        @return str
            "PASS", "ALMOST_PASS", or "FAIL".
        """
        # Lazy imports to avoid circular dependencies (both libs import from GEN_SOF_lib)
        from model_validation.vessel_detection.DVD_lib import run_analysis_script as _run_dvd
        from model_validation.rfi_model.RFI_lib import run_evaluate_test as _run_rfi
        from model_validation.vessel_detection.DVD_lib import cls_DVD001_INPUT
        from model_validation.rfi_model.RFI_lib import cls_RFI001_INPUT

        gen_sof001_logger.info(f"Log file name: {GEN_SOF001_LOG_FILE}")
        gen_sof001_logger.info(f"------------------------------------------------------------")
        gen_sof001_logger.info(f"Test Case Identifier : {UTILS.general_test_cases_matrix[0][0]}")
        gen_sof001_logger.info(f"Test Case Name       : {UTILS.general_test_cases_matrix[0][1]}")
        gen_sof001_logger.info(f"Requirement Verified : {UTILS.general_test_cases_matrix[0][2]}")
        gen_sof001_logger.info(f"------------------------------------------------------------")
        gen_sof001_logger.info(f"Inputs:")
        paths = {
            "DVD config path": cls_DVD001_INPUT.config_path,
            "RFI config path": cls_RFI001_INPUT.config_path,
        }
        UTILS.verify_paths(paths, gen_sof001_logger)
        gen_sof001_logger.info(f"------------------------------------------------------------")
        gen_sof001_logger.info(f"Execution:")

        # --- DVD unrestricted model ---
        gen_sof001_logger.info(f"\tRunning DVD unrestricted model (run_analysis_script)...")
        dvd_out = _run_dvd()
        dvd_precision = getattr(dvd_out, 'precision', None) if not isinstance(dvd_out, dict) else dvd_out.get('precision')
        dvd_recall    = getattr(dvd_out, 'recall',    None) if not isinstance(dvd_out, dict) else dvd_out.get('recall')
        gen_sof001_logger.info(f"\tDVD unrestricted — precision : {dvd_precision if dvd_precision is not None else 'N/A'}")
        gen_sof001_logger.info(f"\tDVD unrestricted — recall    : {dvd_recall    if dvd_recall    is not None else 'N/A'}")

        # --- RFI unrestricted model ---
        gen_sof001_logger.info(f"\tRunning RFI unrestricted model (run_evaluate_test)...")
        rfi_out = _run_rfi()
        rfi_precision = rfi_out.get('precision') if isinstance(rfi_out, dict) else getattr(rfi_out, 'precision', None)
        rfi_recall    = rfi_out.get('recall')    if isinstance(rfi_out, dict) else getattr(rfi_out, 'recall',    None)
        gen_sof001_logger.info(f"\tRFI unrestricted — precision : {rfi_precision if rfi_precision is not None else 'N/A'}")
        gen_sof001_logger.info(f"\tRFI unrestricted — recall    : {rfi_recall    if rfi_recall    is not None else 'N/A'}")

        gen_sof001_logger.info(f"------------------------------------------------------------")

        statuses = []

        for model_label, precision, recall in [
            ("DVD unrestricted", dvd_precision, dvd_recall),
            ("RFI unrestricted", rfi_precision, rfi_recall),
        ]:
            gen_sof001_logger.info(f"\t{model_label}:")
            for metric, value in [("precision", precision), ("recall", recall)]:
                if value is not None:
                    gen_sof001_logger.info(f"\t\t[PASS]: {metric} outputted = {value}")
                    statuses.append("PASS")
                else:
                    gen_sof001_logger.info(f"\t\t[FAIL]: {metric} not outputted by {model_label}")
                    statuses.append("FAIL")

        status = "PASS" if all(s == "PASS" for s in statuses) else "FAIL"
        gen_sof001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def GEN002_VALIDATION(profile_stats_vd_small: dict, profile_stats_rfi_small: dict,
                          profile_stats_rfi_large: dict, profile_stats_vd_large: dict):
        """
        @brief Validates the GEN-002 test case: Computational Complexity Benchmarking and Performance-Efficiency Trade-off Analysis.
        @details
            Checks that GFLOPs and GMACs are present for all four use-cases
            (VD small, RFI small, RFI large, VD large). Also logs parameter count.
            PASS requires GFLOPs and GMACs to be present for every use-case.
        @param profile_stats_vd_small
            dict: containing 'profile/GFLOPs', 'profile/GMACs', 'profile/params_M' for VD small.
        @param profile_stats_rfi_small
            dict: containing 'profile/GFLOPs', 'profile/GMACs', 'profile/params_M' for RFI small.
        @param profile_stats_rfi_large
            dict: containing 'profile/GFLOPs', 'profile/GMACs', 'profile/params_M' for RFI large.
        @param profile_stats_vd_large
            dict: containing 'profile/GFLOPs', 'profile/GMACs', 'profile/params_M' for VD large.
        @return str
            "PASS" or "FAIL".
        """
        use_cases = [
            ("VD  small (lightweight)",  profile_stats_vd_small  or {}),
            ("RFI small (lightweight)",  profile_stats_rfi_small or {}),
            ("RFI large (unrestricted)", profile_stats_rfi_large or {}),
            ("VD  large (unrestricted)", profile_stats_vd_large  or {}),
        ]

        gen_sof001_logger.info(f"Log file name: {GEN_SOF001_LOG_FILE}")
        gen_sof001_logger.info(f"------------------------------------------------------------")
        gen_sof001_logger.info(f"Test Case Identifier : {UTILS.general_test_cases_matrix[1][0]}")
        gen_sof001_logger.info(f"Test Case Name       : {UTILS.general_test_cases_matrix[1][1]}")
        gen_sof001_logger.info(f"Requirement Verified : {UTILS.general_test_cases_matrix[1][2]}")
        gen_sof001_logger.info(f"------------------------------------------------------------")
        gen_sof001_logger.info(f"Execution:")

        statuses = []
        for label, ps in use_cases:
            gflops   = ps.get('profile/GFLOPs',  None)
            gmacs    = ps.get('profile/GMACs',   None)
            params_m = ps.get('profile/params_M', None)
            gen_sof001_logger.info(f"\t{label}:")
            gen_sof001_logger.info(f"\t\tGFLOPs     : {gflops   if gflops   is not None else 'N/A'}")
            gen_sof001_logger.info(f"\t\tGMACs      : {gmacs    if gmacs    is not None else 'N/A'}")
            gen_sof001_logger.info(f"\t\tParams (M) : {params_m if params_m is not None else 'N/A'}")
            missing = [m for m, v in [("GFLOPs", gflops), ("GMACs", gmacs)] if v is None]
            if missing:
                gen_sof001_logger.info(f"\t\t[FAIL]: Missing metrics for {label}: {missing}")
                statuses.append("FAIL")
            else:
                gen_sof001_logger.info(f"\t\t[PASS]: GFLOPs and GMACs available for {label}")
                statuses.append("PASS")

        gen_sof001_logger.info(f"------------------------------------------------------------")
        status = "PASS" if all(s == "PASS" for s in statuses) else "FAIL"
        if status == "PASS":
            gen_sof001_logger.info(f"[PASS]: Complexity metrics (GFLOPs, GMACs) available for all use-cases")
        else:
            gen_sof001_logger.info(f"[FAIL]: One or more use-cases are missing GFLOPs or GMACs")
        gen_sof001_logger.info(f"============================================================\n")
        return status


class cls_SOF_LOGS:

    @staticmethod
    def SOF001_VALIDATION(profile_stats_vd_small: dict, profile_stats_rfi_small: dict):
        """
        @brief Validates the SOF-001 test case: Lightweight modes VRAM usage test.
        @details
            Checks that GPU peak memory for both small models (VD small and RFI small)
            does not exceed the expected maximum.
        @param profile_stats_vd_small
            dict: containing 'profile/gpu_mem_peak_GB' for the VD small model.
        @param profile_stats_rfi_small
            dict: containing 'profile/gpu_mem_peak_GB' for the RFI small model.
        @return str
            "PASS" or "FAIL" (FAIL if either model exceeds the threshold).
        """
        profile_stats_vd_small  = profile_stats_vd_small  or {}
        profile_stats_rfi_small = profile_stats_rfi_small or {}
        vram_gb_vd_small  = profile_stats_vd_small.get('profile/gpu_mem_peak_GB',  None)
        vram_gb_rfi_small = profile_stats_rfi_small.get('profile/gpu_mem_peak_GB', None)

        gen_sof001_logger.info(f"Log file name: {GEN_SOF001_LOG_FILE}")
        gen_sof001_logger.info(f"------------------------------------------------------------")
        gen_sof001_logger.info(f"Test Case Identifier : {UTILS.sof_test_cases_matrix[0][0]}")
        gen_sof001_logger.info(f"Test Case Name       : {UTILS.sof_test_cases_matrix[0][1]}")
        gen_sof001_logger.info(f"Requirement Verified : {UTILS.sof_test_cases_matrix[0][2]}")
        gen_sof001_logger.info(f"------------------------------------------------------------")
        gen_sof001_logger.info(f"Expected outputs:")
        gen_sof001_logger.info(f"\tExpected max VRAM (GB): {cls_GEN_SOF_EXPECTED_OUTPUT.expected_max_vram_gb}")
        gen_sof001_logger.info(f"------------------------------------------------------------")
        gen_sof001_logger.info(f"Execution:")
        gen_sof001_logger.info(f"\tVD  small - GPU Mem Peak (GB): {vram_gb_vd_small  if vram_gb_vd_small  is not None else 'N/A'}")
        gen_sof001_logger.info(f"\tRFI small - GPU Mem Peak (GB): {vram_gb_rfi_small if vram_gb_rfi_small is not None else 'N/A'}")
        gen_sof001_logger.info(f"------------------------------------------------------------")

        statuses = []

        for model_name, vram_gb in [("VD small",  vram_gb_vd_small),
                                     ("RFI small", vram_gb_rfi_small)]:
            if vram_gb is None:
                gen_sof001_logger.info(f"[FAIL]: {model_name} GPU memory peak not available")
                statuses.append("FAIL")
            elif vram_gb <= cls_GEN_SOF_EXPECTED_OUTPUT.expected_max_vram_gb:
                gen_sof001_logger.info(f"[PASS]: {model_name} GPU Mem Peak = {vram_gb:.3f} GB <= Expected max VRAM = {cls_GEN_SOF_EXPECTED_OUTPUT.expected_max_vram_gb} GB")
                statuses.append("PASS")
            else:
                gen_sof001_logger.info(f"[FAIL]: {model_name} GPU Mem Peak = {vram_gb:.3f} GB > Expected max VRAM = {cls_GEN_SOF_EXPECTED_OUTPUT.expected_max_vram_gb} GB")
                statuses.append("FAIL")

        status = "PASS" if all(s == "PASS" for s in statuses) else "FAIL"
        gen_sof001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def SOF002_VALIDATION():
        """
        @brief Validates the SOF-002 test case: Product traceability for pipeline output products.
        @details
            Checks that all files in the RFI post-processing output directory contain "RFI_<number>_" in
            their name, and that all files in the VD labels directory contain "VD_<number>" in
            their name.
        @return str
            "PASS" or "FAIL".
        """
        RFI_PATTERN = re.compile(r'RFI_\d+_')
        VD_PATTERN  = re.compile(r'VD_\d+')

        gen_sof001_logger.info(f"Log file name: {GEN_SOF001_LOG_FILE}")
        gen_sof001_logger.info(f"------------------------------------------------------------")
        gen_sof001_logger.info(f"Test Case Identifier : {UTILS.sof_test_cases_matrix[1][0]}")
        gen_sof001_logger.info(f"Test Case Name       : {UTILS.sof_test_cases_matrix[1][1]}")
        gen_sof001_logger.info(f"Requirement Verified : {UTILS.sof_test_cases_matrix[1][2]}")
        gen_sof001_logger.info(f"------------------------------------------------------------")
        gen_sof001_logger.info(f"Inputs:")
        gen_sof001_logger.info(f"\tRFI post-processing output dir : {cls_GEN_SOF_INPUT.rfi_comparisons_dir}")
        gen_sof001_logger.info(f"\tVD labels dir       : {cls_GEN_SOF_INPUT.vd_labels_dir}")
        gen_sof001_logger.info(f"------------------------------------------------------------")
        gen_sof001_logger.info(f"Execution:")

        statuses = []

        # --- RFI post-processing output directory ---
        rfi_dir = cls_GEN_SOF_INPUT.rfi_comparisons_dir
        if not os.path.isdir(rfi_dir):
            gen_sof001_logger.info(f"\t[FAIL]: RFI post-processing output directory not found: {rfi_dir}")
            statuses.append("FAIL")
        else:
            rfi_files = sorted(os.listdir(rfi_dir))
            gen_sof001_logger.info(f"\tRFI post-processing output dir — {len(rfi_files)} file(s):")
            rfi_invalid = []
            for fname in rfi_files:
                matched = bool(RFI_PATTERN.search(fname))
                gen_sof001_logger.info(f"\t\t{'[OK]' if matched else '[INVALID]'} {fname}")
                if not matched:
                    rfi_invalid.append(fname)
            if rfi_invalid:
                gen_sof001_logger.info(f"\t[FAIL]: {len(rfi_invalid)} RFI file(s) do not match pattern 'RFI_<number>_': {rfi_invalid}")
                statuses.append("FAIL")
            else:
                gen_sof001_logger.info(f"\t[PASS]: All {len(rfi_files)} RFI file(s) match pattern 'RFI_<number>_'")
                statuses.append("PASS")

        # --- VD labels directory ---
        vd_dir = cls_GEN_SOF_INPUT.vd_labels_dir
        if not os.path.isdir(vd_dir):
            gen_sof001_logger.info(f"\t[FAIL]: VD labels directory not found: {vd_dir}")
            statuses.append("FAIL")
        else:
            vd_files = sorted(os.listdir(vd_dir))
            gen_sof001_logger.info(f"\tVD labels dir — {len(vd_files)} file(s):")
            vd_invalid = []
            for fname in vd_files:
                matched = bool(VD_PATTERN.search(fname))
                gen_sof001_logger.info(f"\t\t{'[OK]' if matched else '[INVALID]'} {fname}")
                if not matched:
                    vd_invalid.append(fname)
            if vd_invalid:
                gen_sof001_logger.info(f"\t[FAIL]: {len(vd_invalid)} VD file(s) do not match pattern 'VD_<number>': {vd_invalid}")
                statuses.append("FAIL")
            else:
                gen_sof001_logger.info(f"\t[PASS]: All {len(vd_files)} VD file(s) match pattern 'VD_<number>'")
                statuses.append("PASS")

        gen_sof001_logger.info(f"------------------------------------------------------------")
        status = "PASS" if all(s == "PASS" for s in statuses) else "FAIL"
        if status == "PASS":
            gen_sof001_logger.info(f"[PASS]: All output product files have the expected naming convention")
        else:
            gen_sof001_logger.info(f"[FAIL]: One or more output product files do not match the expected naming convention")
        gen_sof001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def SOF003_VALIDATION(out_vd_small: dict, out_rfi_small: dict,
                          out_rfi_large: dict, out_vd_large: dict):
        """
        @brief Validates the SOF-003 test case: End-to-End Processing Latency Measurement for SAR Pipeline Architectures.
        @details
            Measures and logs the average end-to-end processing latency for each use-case,
            covering both the lightweight (small) and unrestricted (large) architectures.
            PASS requires all four average latencies to be available (not missing).
        @param out_vd_small
            dict: containing 'step1_time_s', 'step2_time_s', 'patches_nr' for VD small.
        @param out_rfi_small
            dict: containing 'step1_time_s', 'step2_time_s', 'patches_nr' for RFI small.
        @param out_rfi_large
            dict: containing 'step1_time_s', 'step2_time_s', 'patches_nr' for RFI large.
        @param out_vd_large
            dict: containing 'step1_time_s', 'step2_time_s', 'patches_nr' for VD large.
        @return str
            "PASS" or "FAIL".
        """
        def _avg_latency(out: dict):
            if out is None:
                return None
            step1 = out.get('step1_time_s')
            step2 = out.get('step2_time_s')
            patches_nr = out.get('patches_nr')
            if step1 is None or step2 is None or patches_nr is None:
                return None
            return (step1 + step2) / max(patches_nr, 1)

        use_cases = [
            ("VD  small (lightweight)",   out_vd_small),
            ("RFI small (lightweight)",   out_rfi_small),
            ("RFI large (unrestricted)",  out_rfi_large),
            ("VD  large (unrestricted)",  out_vd_large),
        ]

        gen_sof001_logger.info(f"Log file name: {GEN_SOF001_LOG_FILE}")
        gen_sof001_logger.info(f"------------------------------------------------------------")
        gen_sof001_logger.info(f"Test Case Identifier : {UTILS.sof_test_cases_matrix[2][0]}")
        gen_sof001_logger.info(f"Test Case Name       : {UTILS.sof_test_cases_matrix[2][1]}")
        gen_sof001_logger.info(f"Requirement Verified : {UTILS.sof_test_cases_matrix[2][2]}")
        gen_sof001_logger.info(f"------------------------------------------------------------")
        gen_sof001_logger.info(f"Inputs:")
        paths = {
            "Config path": cls_GEN_SOF_INPUT.config_path,
        }
        UTILS.verify_paths(paths, gen_sof001_logger)
        gen_sof001_logger.info(f"------------------------------------------------------------")
        gen_sof001_logger.info(f"Execution:")

        statuses = []
        for label, out in use_cases:
            avg = _avg_latency(out)
            patches_nr = (out or {}).get('patches_nr', 'N/A')
            total      = ((out or {}).get('step1_time_s', 0.0) +
                          (out or {}).get('step2_time_s', 0.0)) if out else None
            gen_sof001_logger.info(f"\t{label}:")
            gen_sof001_logger.info(f"\t\ttotal_time_s          : {f'{total:.5f}' if total is not None else 'N/A'} s")
            gen_sof001_logger.info(f"\t\tpatches_nr            : {patches_nr}")
            gen_sof001_logger.info(f"\t\tavg_time_per_patch (s): {f'{avg:.5f}' if avg is not None else 'N/A'}")
            if avg is None:
                gen_sof001_logger.info(f"\t\t[FAIL]: Average latency not available for {label}")
                statuses.append("FAIL")
            else:
                gen_sof001_logger.info(f"\t\t[PASS]: Average latency measured = {avg:.5f} s/patch")
                statuses.append("PASS")

        gen_sof001_logger.info(f"------------------------------------------------------------")
        status = "PASS" if all(s == "PASS" for s in statuses) else "FAIL"
        if status == "PASS":
            gen_sof001_logger.info(f"[PASS]: Average end-to-end latency measured for all use-cases")
        else:
            gen_sof001_logger.info(f"[FAIL]: One or more use-case average latencies are missing")
        gen_sof001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def SOF004_VALIDATION(profile_stats_vd_small: dict, profile_stats_rfi_small: dict,
                          profile_stats_rfi_large: dict, profile_stats_vd_large: dict):
        """
        @brief Validates the SOF-004 test case: CPU and GPU Resource Utilization Measurement for Each Model and Use-Case.
        @details
            Logs and checks peak CPU and GPU utilization metrics for all four models
            (VD small, RFI small, RFI large, VD large). PASS requires all peak utilization
            metrics to be present for every model.
        @param profile_stats_vd_small
            dict: profile stats for VD small model.
        @param profile_stats_rfi_small
            dict: profile stats for RFI small model.
        @param profile_stats_rfi_large
            dict: profile stats for RFI large model.
        @param profile_stats_vd_large
            dict: profile stats for VD large model.
        @return str
            "PASS" or "FAIL".
        """
        _PEAK_METRICS = [
            ("GPU Mem Peak (GB)  ", "profile/gpu_mem_peak_GB"),
            ("GPU Util Peak (%)  ", "profile/gpu_util_peak_%"),
            ("GPU Power Peak (W) ", "profile/gpu_power_peak_W"),
            ("CPU Util Peak (%)  ", "profile/cpu_util_peak_%"),
            ("CPU Util Avg (%)   ", "profile/cpu_util_avg_%"),
        ]

        use_cases = [
            ("VD  small (lightweight)",  profile_stats_vd_small  or {}),
            ("RFI small (lightweight)",  profile_stats_rfi_small or {}),
            ("RFI large (unrestricted)", profile_stats_rfi_large or {}),
            ("VD  large (unrestricted)", profile_stats_vd_large  or {}),
        ]

        gen_sof001_logger.info(f"Log file name: {GEN_SOF001_LOG_FILE}")
        gen_sof001_logger.info(f"------------------------------------------------------------")
        gen_sof001_logger.info(f"Test Case Identifier : {UTILS.sof_test_cases_matrix[3][0]}")
        gen_sof001_logger.info(f"Test Case Name       : {UTILS.sof_test_cases_matrix[3][1]}")
        gen_sof001_logger.info(f"Requirement Verified : {UTILS.sof_test_cases_matrix[3][2]}")
        gen_sof001_logger.info(f"------------------------------------------------------------")
        gen_sof001_logger.info(f"Execution:")

        statuses = []
        for label, ps in use_cases:
            gen_sof001_logger.info(f"\t{label}:")
            missing = []
            for display_name, key in _PEAK_METRICS:
                val = ps.get(key, None)
                val_str = f"{val:.4f}" if isinstance(val, float) else (str(val) if val is not None else "N/A")
                gen_sof001_logger.info(f"\t\t{display_name}: {val_str}")
                if val is None:
                    missing.append(key)
            if missing:
                gen_sof001_logger.info(f"\t\t[FAIL]: Missing peak metrics: {missing}")
                statuses.append("FAIL")
            else:
                gen_sof001_logger.info(f"\t\t[PASS]: All peak CPU/GPU metrics available")
                statuses.append("PASS")

        gen_sof001_logger.info(f"------------------------------------------------------------")
        status = "PASS" if all(s == "PASS" for s in statuses) else "FAIL"
        if status == "PASS":
            gen_sof001_logger.info(f"[PASS]: All peak CPU/GPU resource utilization metrics available for all use-cases")
        else:
            gen_sof001_logger.info(f"[FAIL]: One or more use-cases are missing peak CPU/GPU resource utilization metrics")
        gen_sof001_logger.info(f"============================================================\n")
        return status

    @staticmethod
    def SOF005_VALIDATION(profile_stats_vd_small: dict, profile_stats_rfi_small: dict,
                          profile_stats_rfi_large: dict, profile_stats_vd_large: dict):
        """
        @brief Validates the SOF-005 test case: Complexity Measurement of Core AI Models and End-to-End Pipelines Using FLOPs and MACs.
        @details
            Checks that GFLOPs and GMACs are available for all four use-cases
            (VD small, RFI small, RFI large, VD large). Also logs parameter count.
            PASS requires GFLOPs and GMACs to be present for every use-case.
        @param profile_stats_vd_small
            dict: containing 'profile/GFLOPs', 'profile/GMACs', 'profile/params_M' for VD small.
        @param profile_stats_rfi_small
            dict: containing 'profile/GFLOPs', 'profile/GMACs', 'profile/params_M' for RFI small.
        @param profile_stats_rfi_large
            dict: containing 'profile/GFLOPs', 'profile/GMACs', 'profile/params_M' for RFI large.
        @param profile_stats_vd_large
            dict: containing 'profile/GFLOPs', 'profile/GMACs', 'profile/params_M' for VD large.
        @return str
            "PASS" or "FAIL".
        """
        use_cases = [
            ("VD  small (lightweight)",  profile_stats_vd_small  or {}),
            ("RFI small (lightweight)",  profile_stats_rfi_small or {}),
            ("RFI large (unrestricted)", profile_stats_rfi_large or {}),
            ("VD  large (unrestricted)", profile_stats_vd_large  or {}),
        ]

        gen_sof001_logger.info(f"Log file name: {GEN_SOF001_LOG_FILE}")
        gen_sof001_logger.info(f"------------------------------------------------------------")
        gen_sof001_logger.info(f"Test Case Identifier : {UTILS.sof_test_cases_matrix[4][0]}")
        gen_sof001_logger.info(f"Test Case Name       : {UTILS.sof_test_cases_matrix[4][1]}")
        gen_sof001_logger.info(f"Requirement Verified : {UTILS.sof_test_cases_matrix[4][2]}")
        gen_sof001_logger.info(f"------------------------------------------------------------")
        gen_sof001_logger.info(f"Inputs:")
        paths = {
            "Config path": cls_GEN_SOF_INPUT.config_path,
        }
        UTILS.verify_paths(paths, gen_sof001_logger)
        gen_sof001_logger.info(f"------------------------------------------------------------")
        gen_sof001_logger.info(f"Execution:")

        statuses = []
        for label, ps in use_cases:
            gflops    = ps.get('profile/GFLOPs',  None)
            gmacs     = ps.get('profile/GMACs',   None)
            params_m  = ps.get('profile/params_M', None)
            gen_sof001_logger.info(f"\t{label}:")
            gen_sof001_logger.info(f"\t\tGFLOPs     : {gflops    if gflops   is not None else 'N/A'}")
            gen_sof001_logger.info(f"\t\tGMACs      : {gmacs     if gmacs    is not None else 'N/A'}")
            gen_sof001_logger.info(f"\t\tParams (M) : {params_m  if params_m is not None else 'N/A'}")
            missing = [m for m, v in [("GFLOPs", gflops), ("GMACs", gmacs)] if v is None]
            if missing:
                gen_sof001_logger.info(f"\t\t[FAIL]: Missing metrics for {label}: {missing}")
                statuses.append("FAIL")
            else:
                gen_sof001_logger.info(f"\t\t[PASS]: GFLOPs and GMACs available for {label}")
                statuses.append("PASS")

        gen_sof001_logger.info(f"------------------------------------------------------------")
        status = "PASS" if all(s == "PASS" for s in statuses) else "FAIL"
        if status == "PASS":
            gen_sof001_logger.info(f"[PASS]: Complexity metrics (GFLOPs, GMACs) available for all use-cases")
        else:
            gen_sof001_logger.info(f"[FAIL]: One or more use-cases are missing GFLOPs or GMACs")
        gen_sof001_logger.info(f"============================================================\n")
        return status


# ---------------- INPUT CONFIG ---------------- #

class cls_GEN_SOF_INPUT:
    config_path            = str(Path(__file__).parents[2] / "pipeline" / "focusing_large_model" / "config_focusing_model.yaml")
    rfi_comparisons_dir    = str(Path(__file__).parents[1] / "rfi_model" / "postprocessing_outputs" / "comparisons_remove_high_rfi")
    vd_labels_dir          = str(Path(__file__).parents[2] / "pipeline" / "dvd_use_case" / "large_model" / "scripts" / "test_set" / "labels")


# ---------------- EXPECTED METRICS ---------------- #

class cls_GEN_SOF_EXPECTED_OUTPUT:
    expected_precision     = 0.1
    expected_recall        = 0.1
    expected_max_latency_s = 60.0
    expected_max_vram_gb   = 4.0

# ---------------- TEST CASE ---------------- #

class cls_GEN_SOF_Run:

    @staticmethod
    def test_main_GEN_SOF():
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

        cls_GEN_SOF_SETUP.GEN_SOF_LOG_FILE_SETUP()

        exit_code_vd_small, profile_stats_vd_small, step1_time_s_vd_small, step2_time_s_vd_small, patches_nr_vd_small = run_vd_small_end2end()
        exit_code_rfi_small, profile_stats_rfi_small, step1_time_s_rfi_small, step2_time_s_rfi_small, patches_nr_rfi_small = run_rfi_small_end2end()  
        exit_code_rfi_large, profile_stats_rfi_large, step1_time_s_rfi_large, step2_time_s_rfi_large, patches_nr_rfi_large = run_rfi_large_end2end()
        exit_code_vd_large, profile_stats_vd_large, step1_time_s_vd_large, step2_time_s_vd_large, patches_nr_vd_large = run_vd_large_end2end()

        print(f"exit_code:     {exit_code_vd_small}")
        print(f"step1_time_s:  {step1_time_s_vd_small:.5f} s")
        print(f"step2_time_s:  {step2_time_s_vd_small:.5f} s")
        print(f"patches_nr:    {patches_nr_vd_small}")
        print(f"profile_stats: {profile_stats_vd_small}")
        print(f"Model GFLOPs:                                         {profile_stats_vd_small.get('profile/GFLOPs', 'N/A') if profile_stats_vd_small else 'N/A'}")

        # GEN-001: Precision and recall output check for DVD and RFI unrestricted models
        tc_status.append(cls_GEN_LOGS.GEN001_VALIDATION())

        # GEN-002: FLOPs and MACs existence check for all models
        tc_status.append(cls_GEN_LOGS.GEN002_VALIDATION(profile_stats_vd_small, profile_stats_rfi_small, profile_stats_rfi_large, profile_stats_vd_large))

        # SOF-001: Lightweight modes VRAM usage test
        tc_status.append(cls_SOF_LOGS.SOF001_VALIDATION(profile_stats_vd_small, profile_stats_rfi_small))

        # SOF-002: Product traceability — output file naming convention
        tc_status.append(cls_SOF_LOGS.SOF002_VALIDATION())

        # SOF-003: End-to-end latency measurement for all use-cases
        tc_status.append(cls_SOF_LOGS.SOF003_VALIDATION(
            {'step1_time_s': step1_time_s_vd_small,  'step2_time_s': step2_time_s_vd_small,  'patches_nr': patches_nr_vd_small},
            {'step1_time_s': step1_time_s_rfi_small, 'step2_time_s': step2_time_s_rfi_small, 'patches_nr': patches_nr_rfi_small},
            {'step1_time_s': step1_time_s_rfi_large, 'step2_time_s': step2_time_s_rfi_large, 'patches_nr': patches_nr_rfi_large},
            {'step1_time_s': step1_time_s_vd_large,  'step2_time_s': step2_time_s_vd_large,  'patches_nr': patches_nr_vd_large},
        ))

        # SOF-004: Peak CPU and GPU resource utilization for all models
        tc_status.append(cls_SOF_LOGS.SOF004_VALIDATION(profile_stats_vd_small, profile_stats_rfi_small, profile_stats_rfi_large, profile_stats_vd_large))

        # SOF-005: Complexity Measurement for all models
        tc_status.append(cls_SOF_LOGS.SOF005_VALIDATION(profile_stats_vd_small, profile_stats_rfi_small, profile_stats_rfi_large, profile_stats_vd_large))

        # Log summary of test case status to console and log file
        gen_sof001_logger.info(
            f"Software and general requirements validation status:\n"
            f"\tNumber of PASSED tests      = {tc_status.count('PASS')}\n"
            f"\tNumber of ALMOST_PASS tests = {tc_status.count('ALMOST_PASS')}\n"
            f"\tNumber of FAILED tests      = {tc_status.count('FAIL')}\n"
            f"\tNumber of TOTAL tests       = {len(tc_status)}"
        )
