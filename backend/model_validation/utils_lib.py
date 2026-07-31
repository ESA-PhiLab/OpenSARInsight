import os


class cls_UTILS_LIB:
    general_test_cases_matrix = [
        # USR - requirements
        ["TC-PP-GEN-001", "Evaluation of Inference Performance Using Quantitative Metrics per Use-Case", "UR-GEN-VER-02"],
        ["TC-PP-GEN-002", "Computational Complexity Benchmarking and Performance–Efficiency Trade-off Analysis", "UR-GEN-VER-04"]
    ]
    
    rfi_test_cases_matrix = [
        ["TC-PP-RFI-001", "RFI Detection Test (Tile Classification)", "UR-RFI-FUN-01"],
        ["TC-PP-RFI-002", "RFI Pixelwise Detection and Mask Generation Validation", "UR-RFI-FUN-02"],
        ["TC-PP-RFI-003", "RFI Compensation Test (Image Restoration)", "UR-RFI-FUN-03"],
        ["TC-PP-RFI-004", "RFI Source Parameter Estimation Test", "UR-RFI-FUN-04"],
        ["TC-PP-RFI-005", "RFI Affected Image De-Prioritisation Test", "UR-RFI-FUN-05"],
        ["TC-PP-RFI-006", "Scene-Level RFI Detection Performance Evaluation", "UR-RFI-PER-01"],
        ["TC-PP-RFI-007", "RFI Region Masking IOU Validation", "UR-RFI-PER-02"],
        ["TC-PP-RFI-008", "", "UR-RFI-PER-03"],
        ["TC-PP-RFI-009", "", "UR-RFI-PER-04"],
        ["TC-PP-RFI-010", "RFI De-Prioritisation Threshold", "UR-RFI-PER-05"],
        ["TC-PP-RFI-011", "End-to-End Latency Verification for RFI Processing", "UR-RFI-PER-06"],
        ["TC-PP-RFI-012", "RFI Presence Detection Accuracy Verification", "SR-RFI-PER-01"],
        ["TC-PP-RFI-013", "RFI Region Masking IOU Verification", "SR-RFI-PER-02"],
        ["TC-PP-RFI-014", "", "SR-RFI-PER-03"],
        ["TC-PP-RFI-015", "", "SR-RFI-PER-04"],
        ["TC-PP-RFI-016", "RFI De-Prioritisation Threshold", "SR-RFI-PER-05"],
        ["TC-PP-RFI-017", "End-to-End Processing Latency Verification for Lightweight Pipeline", "SR-RFI-PER-06"]
    ]

    sof_test_cases_matrix = [
        ["TC-PP-SOF-001", "Lightweight modes VRAM usage test", "SR-SOF-PER-05"],
        ["TC-PP-SOF-002", "Product traceability for pipeline output products", "SR-SOF-QUA-03"],
        ["TC-PP-SOF-003", "End-to-End Processing Latency Measurement for SAR Pipeline Architectures", "SR-SOF-VER-04"],
        ["TC-PP-SOF-004", "CPU and GPU Resource Utilization Measurement for Each Model and Use-Case", "SR-SOF-VER-05"],
        ["TC-PP-SOF-005", "Complexity Measurement of Core AI Models and End-to-End Pipelines Using FLOPs and MACs", "SR-SOF-VER-06"]
    ]

    dvd_test_cases_matrix = [
        ["TC-PP-DVD-001", "Vessel Detection and Discrimination in SAR Imagery", "UR-DVD-FUN-01"],
        ["TC-PP-DVD-002", "", "UR-DVD-FUN-02"],
        ["TC-PP-DVD-003", "", "UR-DVD-FUN-05"],
        ["TC-PP-DVD-004", "Vessel Detection Precision Evaluation", "UR-DVD-PER-01"],
        ["TC-PP-DVD-005", "Vessel Detection Recall Evaluation", "UR-DVD-PER-02"],
        ["TC-PP-DVD-006", "Vessel Detection Report Timeliness Evaluation", "UR-DVD-PER-05"],
        ["TC-PP-DVD-007", "Verification of Vessel Detection Precision", "SR-DVD-PER-01"],
        ["TC-PP-DVD-008", "Verification of Vessel Detection Recall", "SR-DVD-PER-02"],
        ["TC-PP-DVD-009", "Land-Based False Positive Robustness", "SR-DVD-PER-06"],
        ["TC-PP-DVD-010", "Maritime Feature Robustness", "SR-DVD-PER-07"],
        ["TC-PP-DVD-011", "End-to-End Latency Verification", "SR-DVD-PER-08"]
    ]


    @staticmethod
    def verify_path(path_file: str, check_permission = True):
        dir_path = os.path.dirname(path_file)

        # 1) Check directory existence
        if not os.path.exists(dir_path):
            return False, f"Failed: Path {path_file} does not exist"

        if check_permission == True: 
            # 2) Check if directory is writable
            if not os.access(dir_path, os.W_OK):
                return False, f"Failed: User does not have permissions to write data to this path {path_file}"
        
        return True, "Success"

    def verify_paths(paths_dict, logger):
        """
        Verify multiple paths.

        Args:
            paths_dict (dict): Dictionary like {"Config path": Path(...), ...}
            logger (logger): Logger used for messages
        """

        for name, path in paths_dict.items():

            ret_value, msg = cls_UTILS_LIB.verify_path(path, False)

            if ret_value == 0:
                logger.info(msg)
                logger.info("============================================================\n")
                return False
            else:
                logger.info(f"\t{name}: {path}")

        return True
