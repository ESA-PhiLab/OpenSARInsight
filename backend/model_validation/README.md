# Model Validation

This module contains the model validation framework for the **OpenSAR Insight / AI4SAR** project (ESA). It verifies that each AI use-case pipeline meets the performance requirements defined in the system and user requirements documents.

---

## Folder Structure

```
model_validation/
├── main.py                  # Master entry point – runs all test cases
├── utils_lib.py             # Shared utilities and test-case matrices
├── vessel_detection/
│   ├── DVD_lib.py           # Test logic for Vessel Detection (DVD) use case
│   ├── config_vd_large.yaml
│   ├── config_vd_maritime_features.yaml
│   ├── data_maritime_features.yaml
│   └── analysis/            # Analysis outputs
├── rfi_model/
│   ├── RFI_lib.py           # Test logic for RFI Detection use case
│   ├── rfi_prediction_mask/ # Prediction mask PNGs
│   └── postprocessing_outputs/
├── gen_sof_req/
│   └── GEN_SOF_lib.py       # General / cross-use-case test logic and end-to-end runners
└── logs/                    # Auto-generated log files (one per run, timestamped)
```

---

## Use Cases and Test Cases

### Vessel Detection (DVD)

Defined in `vessel_detection/DVD_lib.py`. All test cases use the YOLO-based large model validated against the test dataset.

| ID            | Name                                            | Requirement   | Metric / Criterion                                          |
|---------------|-------------------------------------------------|---------------|-------------------------------------------------------------|
| TC-PP-DVD-001 | Vessel Detection and Discrimination in SAR Imagery | UR-DVD-FUN-01 | Precision drop ≤ 10 pp (maritime vs. general)            |
| TC-PP-DVD-004 | Vessel Detection Precision Evaluation           | UR-DVD-PER-01 | Precision ≥ 0.85 (±0.5 threshold)                          |
| TC-PP-DVD-005 | Vessel Detection Recall Evaluation              | UR-DVD-PER-02 | Recall ≥ 0.85 (±0.5 threshold)                             |
| TC-PP-DVD-006 | Vessel Detection Report Timeliness Evaluation   | UR-DVD-PER-05 | Estimated swath processing time < 300 s                     |
| TC-PP-DVD-007 | Verification of Vessel Detection Precision      | SR-DVD-PER-01 | Precision ≥ 0.85 (±0.5 threshold)                          |
| TC-PP-DVD-008 | Verification of Vessel Detection Recall         | SR-DVD-PER-02 | Recall ≥ 0.85 (±0.5 threshold)                             |
| TC-PP-DVD-009 | Land-Based False Positive Robustness            | SR-DVD-PER-06 | Precision drop ≤ 10 pp (near-shore vs. general)            |
| TC-PP-DVD-010 | Maritime Feature Robustness                     | SR-DVD-PER-07 | Precision drop ≤ 10 pp (maritime vs. general)              |
| TC-PP-DVD-011 | End-to-End Latency Verification                 | SR-DVD-PER-08 | Estimated swath processing time < 300 s                     |

**Pass/fail logic for precision and recall tests (DVD-004, DVD-005, DVD-007, DVD-008):**
- `[PASS]` if computed metric ≥ expected value.
- `[PASS]` if computed metric is within the 0.5 tolerance of the expected value, with the threshold reported in the message.
- `[FAIL]` if computed metric < expected − 0.5.

---

### RFI Detection (RFI)

Defined in `rfi_model/RFI_lib.py`. Uses a segmentation model evaluated on a held-out test set.

| ID            | Name                                                        | Requirement   | Metric / Criterion                              |
|---------------|-------------------------------------------------------------|---------------|-------------------------------------------------|
| TC-PP-RFI-001 | RFI Detection Test (Tile Classification)                    | UR-RFI-FUN-01 | F1 ≥ 0.80 (±0.05 tolerance)                    |
| TC-PP-RFI-002 | RFI Pixelwise Detection and Mask Generation Validation      | UR-RFI-FUN-02 | IOU ≥ 0.70 (±0.05 tolerance)                   |
| TC-PP-RFI-005 | RFI Affected Image De-Prioritisation Test                   | UR-RFI-FUN-05 | successful > 0 and failed = 0                   |
| TC-PP-RFI-006 | Scene-Level RFI Detection Performance Evaluation            | UR-RFI-PER-01 | Precision & Recall ≥ 0.85 (±0.05 tolerance)    |
| TC-PP-RFI-007 | RFI Region Masking IOU Validation                           | UR-RFI-PER-02 | IOU ≥ 0.70 (±0.05 tolerance)                   |
| TC-PP-RFI-010 | RFI De-Prioritisation Threshold                             | UR-RFI-PER-05 | successful > 0 and failed = 0                   |
| TC-PP-RFI-011 | End-to-End Latency Verification for RFI Processing          | UR-RFI-PER-06 | Estimated swath processing time < 60 s          |
| TC-PP-RFI-012 | RFI Presence Detection Accuracy Verification                | SR-RFI-PER-01 | Accuracy ≥ 0.85 (±0.05 tolerance)              |
| TC-PP-RFI-013 | RFI Region Masking IOU Verification                         | SR-RFI-PER-02 | IOU ≥ 0.70 (±0.05 tolerance)                   |
| TC-PP-RFI-016 | RFI De-Prioritisation Threshold                             | SR-RFI-PER-05 | successful > 0 and failed = 0                   |
| TC-PP-RFI-017 | End-to-End Processing Latency Verification (Lightweight)    | SR-RFI-PER-06 | Latency per 100 km² < 30 s                      |

**Pass/fail logic (RFI):**
- `[PASS]` if computed metric ≥ expected value.
- `[ALMOST_PASS]` if computed metric is within the tolerance of the expected value.
- `[FAIL]` if computed metric falls below the tolerance threshold.

---

### General Requirements (GEN)

Defined in `gen_sof_req/GEN_SOF_lib.py`. Covers inference performance and computational complexity across all use cases.

| ID            | Name                                                                   | Requirement   | Metric / Criterion                             |
|---------------|------------------------------------------------------------------------|---------------|------------------------------------------------|
| TC-PP-GEN-001 | Evaluation of Inference Performance Using Quantitative Metrics per Use-Case | UR-GEN-VER-02 | Precision and recall outputted for DVD and RFI |
| TC-PP-GEN-002 | Computational Complexity Benchmarking and Performance–Efficiency Trade-off Analysis | UR-GEN-VER-04 | GFLOPs and GMACs available for all 4 models  |

---

### System / SOF Requirements (SOF)

Defined in `gen_sof_req/GEN_SOF_lib.py`. Covers system-level requirements for resource usage, traceability, latency, and complexity.

| ID            | Name                                                                              | Requirement   | Metric / Criterion                                              |
|---------------|-----------------------------------------------------------------------------------|---------------|-----------------------------------------------------------------|
| TC-PP-SOF-001 | Lightweight Modes VRAM Usage Test                                                 | SR-SOF-PER-05 | GPU peak memory ≤ 4.0 GB for VD small and RFI small            |
| TC-PP-SOF-002 | Product Traceability for Pipeline Output Products                                 | SR-SOF-QUA-03 | All RFI output files match `RFI_<n>_` and VD files match `VD_<n>` |
| TC-PP-SOF-003 | End-to-End Processing Latency Measurement for SAR Pipeline Architectures          | SR-SOF-VER-04 | Average latency measured for all 4 models (VD/RFI × small/large) |
| TC-PP-SOF-004 | CPU and GPU Resource Utilization Measurement for Each Model and Use-Case          | SR-SOF-VER-05 | GPU/CPU peak utilization metrics present for all 4 models       |
| TC-PP-SOF-005 | Complexity Measurement of Core AI Models Using FLOPs and MACs                    | SR-SOF-VER-06 | GFLOPs and GMACs available for all 4 models                     |

---

## Configuration

Expected metric thresholds are defined as class attributes and can be adjusted without touching test logic:

| Class                        | File                          | Configurable fields                                                                  |
|------------------------------|-------------------------------|--------------------------------------------------------------------------------------|
| `cls_DVD001_EXPECTED_OUTPUT` | `vessel_detection/DVD_lib.py` | `expected_precision`, `expected_recall`, `expected_max_e2e_time_s`, `expected_max_latency_per_100km2_s` |
| `cls_DVD001_INPUT`           | `vessel_detection/DVD_lib.py` | `config_path`, `data_maritime_yaml`, `config_near_shore`, `patches_per_swath`        |
| `cls_RFI001_EXPECTED_OUTPUT` | `rfi_model/RFI_lib.py`        | `expected_precision`, `expected_recall`, `expected_f1`, `expected_iou`, `expected_acc`, `expected_max_e2e_time_s`, `expected_max_latency_per_100km2_s` |
| `cls_RFI001_INPUT`           | `rfi_model/RFI_lib.py`        | `config_path`, `weights`, `test_dir`, `thr`, `batch_size`, `patch_area_km2`, `patches_per_swath` |
| `cls_GEN_SOF_EXPECTED_OUTPUT`| `gen_sof_req/GEN_SOF_lib.py`  | `expected_max_vram_gb`, `expected_max_latency_s`                                     |

---

## How to Run

From the root folder of the `backend/` directory:

```bash
python3 -m backend.model_validation.main
```

Individual use-case suites can be enabled/disabled by commenting in or out the corresponding calls in `main.py`:

```python
RFI.test_main_RFI()         # RFI detection test suite
DVD001_TC.tc_DVD001()       # Vessel detection test suite
GEN_SOF.test_main_GEN_SOF() # General / system requirements
```

---

## Logs

Each run writes a timestamped log file to `model_validation/logs/`:

| File pattern                      | Use case              |
|-----------------------------------|-----------------------|
| `DVD001_YYYYMMDD_HHMMSS.log`      | Vessel Detection      |
| `RFI001_YYYYMMDD_HHMMSS.log`      | RFI Detection         |
| `GEN_SOF001_YYYYMMDD_HHMMSS.log`  | General / SOF         |

Each log records: test case identifier, requirement verified, input paths, expected vs. computed metrics, and a `[PASS]` / `[FAIL]` (or `[ALMOST_PASS]` for RFI) verdict per test case, followed by a summary line with passed/failed/total counts.
