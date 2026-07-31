# type: ignore

"""
---------------------------------------------------------------------
Project: OpenSAR Insight
---------------------------------------------------------------------
train_pose.py

Tool: YOLO-Pose Validation Script for Vessel Detection on RC patches

Author: Abdulhameed Yunusa (ABHY)
E-mail: ayunusa@indracompany.com
Creation Date: 2026-04-08

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""
# type: ignore
import os
from pathlib import Path
import wandb
from ultralytics import YOLO
from utilities.read_yaml import read_yaml
from scripts.profile_model import *
# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
config = read_yaml(Path('config.yaml'))
training_config = config.get('training', {})
wandb_config = training_config.get('wandb', {})
inf_config = config.get('inference', {})
# ═══════════════════════════════════════════════════════════════
# Model Setup
# ═══════════════════════════════════════════════════════════════
model = YOLO(inf_config.get('model_path', 'yolo26s-pose.pt')) 

# ═══════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════
print("\nRunning validation...")
val_results = model.val(data=training_config.get('yaml_file', ''),
                        split=inf_config.get('val_split', 'val'),
                        conf=inf_config.get('conf_threshold', 0.25),
                        iou=inf_config.get('iou_threshold', 0.25)
)
print({
    "val/box_mAP50": val_results.box.map50,
    "val/box_mAP50-95": val_results.box.map,
    "val/box_precision": val_results.box.mp,
    "val/box_recall": val_results.box.mr,
    "val/pose_mAP50": val_results.pose.map50,
    "val/pose_mAP50-95": val_results.pose.map,
    "val/pose_precision": val_results.pose.mp,
    "val/pose_recall": val_results.pose.mr,
})
# Auto-prints on creation
profile_stats = profile_model(model, imgsz=training_config.get('imgsz', 512))

# ═══════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Validation Results")
print("-" * 60)
print(f"Pose mAP@0.5      : {val_results.pose.map50:.4f}")
print(f"Pose mAP@0.5:0.95 : {val_results.pose.map:.4f}")
print(f"Pose precision     : {val_results.pose.mp:.4f}")
print(f"Pose recall        : {val_results.pose.mr:.4f}")
print("=" * 60)