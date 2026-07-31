# type: ignore

"""
---------------------------------------------------------------------
Project: OpenSAR Insight
---------------------------------------------------------------------
train_pose.py

Tool: YOLO-Pose Training Script for Vessel Detection on RC patches

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

# ═══════════════════════════════════════════════════════════════
# Model Setup
# ═══════════════════════════════════════════════════════════════
model = YOLO(training_config.get('model', 'yolov26s-pose.pt'))  # or yolov26s-pose.pt, yolov26m-pose.pt
# model = YOLO('yolo26l-pose.pt')
# ═══════════════════════════════════════════════════════════════
# Training Summary
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("YOLO-Pose Training")
print("=" * 60)
print(f"Dataset     : data.yaml")
print(f"Epochs      : {training_config.get('epochs', 100)}")
print(f"Batch size  : {training_config.get('batch', 16)}")
print(f"Image size  : {training_config.get('imgsz', 512)}")
print(f"Device      : {training_config.get('device', 0)}")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# W&B Initialization
# ═══════════════════════════════════════════════════════════════

# --- Make wandb optional ---
wandb_key = wandb_config.get('api_key') or os.environ.get('WANDB_API_KEY')
wandb_enabled = bool(wandb_config.get('enabled', True))
wandb_mode = 'online' if wandb_enabled and wandb_key else 'disabled'
if wandb_enabled and wandb_key:
    wandb.login(key=wandb_key)

with wandb.init(
    project=wandb_config.get('project', 'VD YOLO on RC'),
    entity=wandb_config.get('entity', 'opensar-dvd-large-model'),
    name=wandb_config.get('name', ''),
    job_type="train",
    mode=wandb_mode
) as run:

    # ═══════════════════════════════════════════════════════════
    # Training
    # ═══════════════════════════════════════════════════════════
    results = model.train(
        data=training_config.get('yaml_file', ''),

        # -- Core training params --
        epochs=training_config.get('epochs', 100),
        batch=training_config.get('batch', 16),
        nbs=training_config.get('nbs', 16),
        imgsz=training_config.get('imgsz', 640),
        patience=training_config.get('patience', 50),

        # -- Optimizer --
        optimizer=training_config.get('optimizer', 'Adam'),
        lr0=training_config.get('lr0', 0.0001),
        lrf=training_config.get('lrf', 0.01),
        weight_decay=training_config.get('weight_decay', 0.005),
        cos_lr=training_config.get('cos_lr', False),

        # -- Detection thresholds --
        conf=training_config.get('conf', 0.001),
        iou=training_config.get('iou', 0.5),

        # -- Augmentation --
        augment=training_config.get('augment', False),
        auto_augment=training_config.get('auto_augment', ''),
        fliplr=training_config.get('fliplr', 0.5),
        flipud=training_config.get('flipud', 0.0),
        degrees=training_config.get('degrees', 0),
        scale=training_config.get('scale', 0.0),
        mosaic=training_config.get('mosaic', 1.0),

        # -- Hardware & output --
        pretrained=True,
        device=training_config.get('device', 0),
        workers=training_config.get('workers', 4),
        name=training_config.get('name', ''),
        exist_ok=training_config.get('exist_ok', True),
    )

    print("\nTraining complete")
    # ═══════════════════════════════════════════════════════════
    # Validation
    # ═══════════════════════════════════════════════════════════
    print("\nRunning validation...")
    val_results = model.val(data='data.yaml', conf=0.001, iou=0.5)
    wandb.log({
        "val/box_mAP50": val_results.box.map50,
        "val/box_mAP50-95": val_results.box.map,
        "val/box_precision": val_results.box.mp,
        "val/box_recall": val_results.box.mr,
        "val/pose_mAP50": val_results.pose.map50,
        "val/pose_mAP50-95": val_results.pose.map,
        "val/pose_precision": val_results.pose.mp,
        "val/pose_recall": val_results.pose.mr,
    })

# ═══════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Validation Results")
print("=" * 60)
print(f"Box  mAP@0.5      : {val_results.box.map50:.4f}")
print(f"Box  mAP@0.5:0.95 : {val_results.box.map:.4f}")
print(f"Box  precision     : {val_results.box.mp:.4f}")
print(f"Box  recall        : {val_results.box.mr:.4f}")
print("-" * 60)
print(f"Pose mAP@0.5      : {val_results.pose.map50:.4f}")
print(f"Pose mAP@0.5:0.95 : {val_results.pose.map:.4f}")
print(f"Pose precision     : {val_results.pose.mp:.4f}")
print(f"Pose recall        : {val_results.pose.mr:.4f}")
print("=" * 60)