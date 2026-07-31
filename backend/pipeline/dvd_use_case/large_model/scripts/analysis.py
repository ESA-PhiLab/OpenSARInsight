# type: ignore
"""
---------------------------------------------------------------------
Project: OpenSAR Insight
---------------------------------------------------------------------
analysis.py

Tool: YOLO-Pose inference analysis for metadata-aware performance slicing.

Description:
Uses YOLO's own model.val() to compute metrics. 
For polarization slicing (VH/VV), creates
temporary dataset subsets and runs model.val() on each, then reports
per-group averaged metrics.

Outputs:
- Overall test-set metrics (Precision, Recall, mAP@50, mAP@50–95)
- Per-polarization metrics (VH, VV)
- Per-bucket metrics (wind, vessel size, density, shore, swath)
- Group-level latency metrics
- Performance plots (bar charts)
- Full analysis log

Author: Abdulhameed Yunusa
---------------------------------------------------------------------
"""

import csv
import os
import re
import sys
import time
import shutil
import tempfile
import argparse
from pathlib import Path

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utilities.read_yaml import read_yaml
from scripts.profile_model import profile_model

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
config = read_yaml(Path(__file__).resolve().parent.parent / "config.yaml")
inference_config = config.get("inference", {})
analysis_config = config.get("analysis", {})

MODEL_PATH = inference_config.get("model_path", "")
TEST_IMAGES_DIR = inference_config.get("test_images_dir", "")
TEST_LABELS_DIR = inference_config.get("test_labels_dir", "")
OUTPUT_DIR = inference_config.get("output_dir", "")
IOU_THRESHOLD = inference_config.get("iou_threshold", 0.5)
CONF_THRESHOLD = inference_config.get("conf_threshold", 0.001)
IMGSZ = inference_config.get("imgsz", 640)
DATA_YAML = str(Path(__file__).resolve().parent.parent / "data.yaml")
SPLIT = inference_config.get("val_split", 'val')
PANEL_BG = "#1e1e1e"
PANEL_FG = "#e0e0e0"

# Analysis bucket selection: 'all' or list of enabled bucket types
ACTIVE_BUCKETS = analysis_config.get("buckets", "all")

# Configurable thresholds per bucket type (names match config.yaml fields)
WIND_CFG = analysis_config.get("wind", {})
WIND_LOW = WIND_CFG.get("low", 4)
WIND_MEDIUM = WIND_CFG.get("medium", 8)

VESSEL_SIZE_CFG = analysis_config.get("vessel_size", {})
VESSEL_SIZE_SMALL_VESSEL = VESSEL_SIZE_CFG.get("small_vessel", 50)
VESSEL_SIZE_MEDIUM_VESSEL = VESSEL_SIZE_CFG.get("medium_vessel", 150)

DENSITY_CFG = analysis_config.get("density", {})
DENSITY_SINGLE_SHIP = DENSITY_CFG.get("single_ship", 1)
DENSITY_FEW_SHIPS = DENSITY_CFG.get("few_ships", 4)

SHORE_CFG = analysis_config.get("shore", {})
SHORE_NEAR_SHORE = SHORE_CFG.get("near_shore", 5)
SHORE_MID_SHORE = SHORE_CFG.get("mid_shore", 20)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def extract_patch_number(filename: str):
    m = re.search(r"VD_(\d+)", filename)
    return m.group(1) if m else None


def extract_polarization(filename: str):
    if "_VH_" in filename:
        return "VH"
    if "_VV_" in filename:
        return "VV"
    return "unknown"


# ------------------------------------------------------------
# Metadata
# ------------------------------------------------------------
def load_metadata_csv(csv_path):
    patches = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            patch_id = row["xml_file"].replace(".xml", "").split("_")[-1]
            if patch_id not in patches:
                patches[patch_id] = {
                    "wind_speed": float(row["wind_speed"]) if row.get("wind_speed") else None,
                    "distance_to_shore": float(row["distance_to_shore"]) if row.get("distance_to_shore") else None,
                    "swath": int(row["swath"]) if row.get("swath") else None,
                    "vessel_lengths": [],
                    "num_ships": 0,
                }
            if row.get("vessel_length"):
                patches[patch_id]["vessel_lengths"].append(float(row["vessel_length"]))
            patches[patch_id]["num_ships"] += 1

    for _, info in patches.items():
        lengths = info["vessel_lengths"]
        info["mean_vessel_length"] = float(np.mean(lengths)) if lengths else None
    return patches


# ------------------------------------------------------------
# Buckets (using configurable thresholds)
# ------------------------------------------------------------
ALL_BUCKET_TYPES = ["polarization", "wind", "vessel_size", "density", "shore", "swath"]


def _enabled_bucket_types():
    """Return list of active bucket type names based on config."""
    if ACTIVE_BUCKETS == "all":
        return ALL_BUCKET_TYPES
    if isinstance(ACTIVE_BUCKETS, list):
        return [b for b in ACTIVE_BUCKETS if b in ALL_BUCKET_TYPES]
    return ALL_BUCKET_TYPES


def wind_bucket(w):
    if w is None: return None
    if w < WIND_LOW: return "wind_low"
    if w < WIND_MEDIUM: return "wind_medium"
    return "wind_high"


def vessel_size_bucket(v):
    if v is None: return None
    if v < VESSEL_SIZE_SMALL_VESSEL: return "small_vessel"
    if v < VESSEL_SIZE_MEDIUM_VESSEL: return "medium_vessel"
    return "large_vessel"


def density_bucket(n):
    if n <= DENSITY_SINGLE_SHIP: return "single_ship"
    if n <= DENSITY_FEW_SHIPS: return "few_ships"
    return "dense_scene"


def shore_bucket(d):
    if d is None: return None
    if d < SHORE_NEAR_SHORE: return "near_shore"
    if d < SHORE_MID_SHORE: return "mid_shore"
    return "offshore"


def swath_bucket(s):
    if s is None: return None
    return f"swath_{int(s)}"


def get_image_groups(img_name, meta):
    """Return list of (bucket_type, bucket_name) for an image based on active buckets."""
    enabled = _enabled_bucket_types()
    groups = []
    if "polarization" in enabled:
        groups.append(extract_polarization(img_name))
    if "wind" in enabled:
        groups.append(wind_bucket(meta.get("wind_speed")))
    if "vessel_size" in enabled:
        groups.append(vessel_size_bucket(meta.get("mean_vessel_length")))
    if "density" in enabled:
        groups.append(density_bucket(meta.get("num_ships", 0)))
    if "shore" in enabled:
        groups.append(shore_bucket(meta.get("distance_to_shore")))
    if "swath" in enabled:
        groups.append(swath_bucket(meta.get("swath")))
    return [g for g in groups if g is not None]


# ------------------------------------------------------------
# Subset validation using model.val()
# ------------------------------------------------------------
def create_subset_dataset(image_files, images_dir, labels_dir, tmp_base):
    """Create a temporary dataset directory with symlinks to a subset of images/labels."""
    img_dir = os.path.join(tmp_base, "images", "test")
    lbl_dir = os.path.join(tmp_base, "labels", "test")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for img_name in image_files:
        src_img = os.path.join(images_dir, img_name)
        dst_img = os.path.join(img_dir, img_name)
        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)

        label_name = Path(img_name).stem + ".txt"
        src_lbl = os.path.join(labels_dir, label_name)
        dst_lbl = os.path.join(lbl_dir, label_name)
        if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
            os.symlink(src_lbl, dst_lbl)

    # Write data.yaml for this subset
    data_yaml_path = os.path.join(tmp_base, "data.yaml")
    data_config = {
        "path": tmp_base,
        "train": "images/test",  # not used but required
        "val": "images/test",
        "test": "images/test",
        "nc": 1,
        "names": ["vessel"],
        "kpt_shape": [1, 3],
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_config, f)

    return data_yaml_path


def run_val_on_subset(model, data_yaml, device, iou_thresh):
    """Run model.val() on a dataset subset and return metrics dict."""
    results = model.val(
        data=data_yaml,
        split=SPLIT,
        imgsz=IMGSZ,
        conf=CONF_THRESHOLD,
        iou=iou_thresh,
        device=device,
        verbose=False,
    )
    return {
        "box_precision": float(results.box.mp),
        "box_recall": float(results.box.mr),
        "box_mAP50": float(results.box.map50),
        "box_mAP50-95": float(results.box.map),
        "pose_precision": float(results.pose.mp),
        "pose_recall": float(results.pose.mr),
        "pose_mAP50": float(results.pose.map50),
        "pose_mAP50-95": float(results.pose.map),
        # Use pose metrics as primary (matches training output)
        "precision": float(results.pose.mp),
        "recall": float(results.pose.mr),
        "mAP50": float(results.pose.map50),
        "mAP50-95": float(results.pose.map),
    }


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def plot_group_metrics(group_metrics, output_dir):
    """Generate per-metric bar charts for each group."""
    metrics = ["precision", "recall", "mAP50", "mAP50-95"]
    groups = list(group_metrics.keys())

    for metric in metrics:
        vals = [group_metrics[g][metric] for g in groups]

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor(PANEL_BG)
        ax.set_facecolor(PANEL_BG)

        bars = ax.bar(groups, vals, color="#4a90d9")
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric, color=PANEL_FG)
        ax.set_title(f"{metric} by Group", color=PANEL_FG)
        ax.tick_params(axis="x", rotation=30, colors=PANEL_FG)
        ax.tick_params(axis="y", colors=PANEL_FG)

        # Add value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", color=PANEL_FG, fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_by_group.png"), dpi=150)
        plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def run_analysis(args):
    model = YOLO(args.model or MODEL_PATH)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. This script requires GPU.")

    device = "cuda:0"
    model.to(device)

    metadata = load_metadata_csv(args.metadata)
    output_dir = os.path.join(args.output or OUTPUT_DIR, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    images_dir = args.images or TEST_IMAGES_DIR
    labels_dir = args.labels or TEST_LABELS_DIR
    iou_thresh = args.iou

    # Profile model
    if str(model.ckpt_path).endswith(".pt"):
        try:
            model.model = model.model.to(next(model.model.parameters()).device)
            profile_model(model)
        except Exception as e:
            print(f"Model profiling skipped: {e}")

    # ══════════════════════════════════════════════════════════
    # 1. Overall validation on full test set
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("OVERALL TEST SET VALIDATION (model.val)")
    print("="*70)

    overall = run_val_on_subset(model, DATA_YAML, device, iou_thresh)
    print(f"  Box  — P={overall['box_precision']:.4f}  R={overall['box_recall']:.4f}  mAP50={overall['box_mAP50']:.4f}  mAP50-95={overall['box_mAP50-95']:.4f}")
    print(f"  Pose — P={overall['pose_precision']:.4f}  R={overall['pose_recall']:.4f}  mAP50={overall['pose_mAP50']:.4f}  mAP50-95={overall['pose_mAP50-95']:.4f}")

    # ══════════════════════════════════════════════════════════
    # 2. Split images into buckets
    # ══════════════════════════════════════════════════════════
    test_images = sorted(os.listdir(images_dir))
    buckets = {}  # bucket_name -> list of image filenames
    enabled = _enabled_bucket_types()

    print(f"\n  Active bucket types: {enabled}")
    print(f"  Thresholds:")
    if "wind" in enabled:
        print(f"    wind        — low: 0 ≤ wind < {WIND_LOW}, medium: {WIND_LOW} ≤ wind < {WIND_MEDIUM}, high: wind ≥ {WIND_MEDIUM}")
    if "vessel_size" in enabled:
        print(f"    vessel_size — small: length < {VESSEL_SIZE_SMALL_VESSEL}, medium: {VESSEL_SIZE_SMALL_VESSEL} ≤ length < {VESSEL_SIZE_MEDIUM_VESSEL}, large: length ≥ {VESSEL_SIZE_MEDIUM_VESSEL}")
    if "density" in enabled:
        print(f"    density     — single: n ≤ {DENSITY_SINGLE_SHIP}, few: {DENSITY_SINGLE_SHIP+1} ≤ n ≤ {DENSITY_FEW_SHIPS}, dense: n ≥ {DENSITY_FEW_SHIPS+1}")
    if "shore" in enabled:
        print(f"    shore       — near: 0–{SHORE_NEAR_SHORE} km, mid: {SHORE_NEAR_SHORE}–{SHORE_MID_SHORE} km, offshore: > {SHORE_MID_SHORE} km")

    for img_name in test_images:
        patch_id = extract_patch_number(img_name)
        if not patch_id:
            continue

        meta = metadata.get(patch_id, {})
        groups = get_image_groups(img_name, meta)

        for g in groups:
            buckets.setdefault(g, []).append(img_name)

    # ══════════════════════════════════════════════════════════
    # 3. Run model.val() per bucket
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("PER-BUCKET VALIDATION (model.val on each subset)")
    print("="*70)

    group_metrics = {"all": overall}
    tmp_root = tempfile.mkdtemp(prefix="yolo_analysis_")

    try:
        for bucket_name, img_list in sorted(buckets.items()):
            if len(img_list) == 0:
                continue

            print(f"\n  [{bucket_name}] ({len(img_list)} images) ... ", end="", flush=True)

            # Create temp subset
            tmp_dir = os.path.join(tmp_root, bucket_name)
            data_yaml = create_subset_dataset(img_list, images_dir, labels_dir, tmp_dir)

            # Run val
            try:
                metrics = run_val_on_subset(model, data_yaml, device, iou_thresh)
                group_metrics[bucket_name] = metrics
                print(f"\n    Box  — P={metrics['box_precision']:.4f}  R={metrics['box_recall']:.4f}  "
                      f"mAP50={metrics['box_mAP50']:.4f}  mAP50-95={metrics['box_mAP50-95']:.4f}")
                print(f"    Pose — P={metrics['pose_precision']:.4f}  R={metrics['pose_recall']:.4f}  "
                      f"mAP50={metrics['pose_mAP50']:.4f}  mAP50-95={metrics['pose_mAP50-95']:.4f}")
            except Exception as e:
                print(f"FAILED: {e}")
                group_metrics[bucket_name] = {
                    "box_precision": 0.0, "box_recall": 0.0, "box_mAP50": 0.0, "box_mAP50-95": 0.0,
                    "pose_precision": 0.0, "pose_recall": 0.0, "pose_mAP50": 0.0, "pose_mAP50-95": 0.0,
                    "precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50-95": 0.0,
                }
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # ══════════════════════════════════════════════════════════
    # 4. Latency profiling per image (quick pass)
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("LATENCY PROFILING")
    print("="*70)

    latencies = {k: [] for k in group_metrics}
    for img_name in test_images:
        patch_id = extract_patch_number(img_name)
        if not patch_id:
            continue

        img_path = os.path.join(images_dir, img_name)
        t0 = time.perf_counter()
        model.predict(img_path, imgsz=640, conf=0.25, verbose=False, device=device)
        infer_ms = (time.perf_counter() - t0) * 1000

        latencies["all"].append(infer_ms)

        meta = metadata.get(patch_id, {})
        groups = get_image_groups(img_name, meta)
        for g in groups:
            if g in latencies:
                latencies[g].append(infer_ms)

    avg_latency = {k: float(np.mean(v)) if v else 0.0 for k, v in latencies.items()}
    print(f"  Overall avg latency: {avg_latency['all']:.2f} ms")

    # ══════════════════════════════════════════════════════════
    # 5. Summary & Plots
    # ══════════════════════════════════════════════════════════
    plot_group_metrics(group_metrics, output_dir)

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    for group, metrics in group_metrics.items():
        n_imgs = len(buckets.get(group, test_images)) if group != "all" else len(test_images)
        print(f"\n[{group}]")
        print(f"  Images        : {n_imgs}")
        print(f"  Box  — P={metrics['box_precision']:.4f}  R={metrics['box_recall']:.4f}  mAP50={metrics['box_mAP50']:.4f}  mAP50-95={metrics['box_mAP50-95']:.4f}")
        print(f"  Pose — P={metrics['pose_precision']:.4f}  R={metrics['pose_recall']:.4f}  mAP50={metrics['pose_mAP50']:.4f}  mAP50-95={metrics['pose_mAP50-95']:.4f}")
        print(f"  Avg Latency   : {avg_latency.get(group, 0):.2f} ms")

    print(f"\nAnalysis complete -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="YOLO metadata-aware inference analysis")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--images", type=str, default=None)
    parser.add_argument("--labels", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--metadata",
        type=str,
        default="patch_vessel_metadata.csv",
        help="Path to metadata CSV (default: patch_vessel_metadata.csv)"
    )
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
