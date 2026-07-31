# type: ignore

"""
---------------------------------------------------------------------
Project: OpenSAR Insight
---------------------------------------------------------------------
train_KD.py

YOLO-Pose Feature-Level Knowledge Distillation Training Script

Author: Giorgia Gobbi (GIOG)
E-mail: ggobbi@indra.es
Creation Date: 2026-05-01

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""

import argparse
import os
from pathlib import Path

import wandb
from ultralytics import YOLO
import sys 

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from pipeline.dvd_use_case.small_model.utilities.read_yaml import read_yaml
from pipeline.dvd_use_case.small_model.utilities.kd_utilities import PoseFeatureKDTrainer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO-Pose with feature-level knowledge distillation."
    )

    config_path = Path(__file__).parent.joinpath("config.yaml")
    if not config_path.exists():
        config_path = os.environ.get("LVDM_CONFIG_PATH")

    parser.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help="Path to the KD YAML config file.",
    )

    return parser.parse_args()

def main():

    args = parse_args()
    config_path = args.config

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    os.environ["POSE_KD_CONFIG_PATH"] = str(config_path)

    config = read_yaml(config_path)

    training_config = config.get("training", {})
    wandb_config = training_config.get("wandb", {})
    kd_config = config.get("kd", {})

    teacher_weights = kd_config.get("teacher_weights")
    student_weights = training_config.get("student_weights")
    data_yaml = training_config.get("data")

    run_name = training_config.get(
        "name",
        "yolo26n-pose_feature_kd_from_teacher",
    )

    model = YOLO(student_weights)

    print("=" * 60)
    print("YOLO-Pose Feature Knowledge Distillation Training")
    print("=" * 60)
    print(f"Config      : {config_path}")
    print(f"Teacher     : {teacher_weights}")
    print(f"Student     : {student_weights}")
    print(f"Dataset     : {data_yaml}")
    print(f"Run name    : {run_name}")
    print(f"Epochs      : {training_config.get('epochs', 100)}")
    print(f"Batch size  : {training_config.get('batch', 16)}")
    print(f"Image size  : {training_config.get('imgsz', 640)}")
    print(f"Device      : {training_config.get('device', 0)}")
    print(f"KD enabled  : {kd_config.get('enabled', True)}")
    print("=" * 60)

    # -------------------------------------------------------
    # W&B
    # -------------------------------------------------------

    # --- Make wandb optional ---
    wandb_key = wandb_config.get("api_key") or os.environ.get("WANDB_API_KEY")
    wandb_enabled = bool(wandb_config.get("enabled", True))
    wandb_mode = "online" if wandb_enabled and wandb_key else "disabled"
    if wandb_enabled and wandb_key:
        wandb.login(key=wandb_key)

    wandb_project = wandb_config.get("project", "VD YOLO on RC")
    wandb_entity = wandb_config.get("entity", "opensar-dvd-large-model")

    with wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        job_type="train",
        mode=wandb_mode,
        config={
            "training": training_config,
            "kd": kd_config,
        },
    ):

        # -------------------------------------------------------
        # Train
        # -------------------------------------------------------

        results = model.train(
            data=data_yaml,
            trainer=PoseFeatureKDTrainer,

            # -- Core training params --
            epochs=training_config.get("epochs", 100),
            batch=training_config.get("batch", 16),
            nbs=training_config.get("nbs", 16),
            imgsz=training_config.get("imgsz", 640),
            patience=training_config.get("patience", 50),

            # -- Optimizer --
            optimizer=training_config.get("optimizer", "AdamW"),
            lr0=training_config.get("lr0", 0.0001),
            lrf=training_config.get("lrf", 0.01),
            weight_decay=training_config.get("weight_decay", 0.0005),
            cos_lr=training_config.get("cos_lr", True),

            # -- Detection thresholds --
            conf=training_config.get("conf", 0.001),
            iou=training_config.get("iou", 0.5),

            # -- Augmentation --
            augment=training_config.get("augment", True),
            auto_augment=training_config.get("auto_augment", ""),
            fliplr=training_config.get("fliplr", 0.0),
            flipud=training_config.get("flipud", 0.0),
            degrees=training_config.get("degrees", 5.0),
            translate=training_config.get("translate", 0.1),
            scale=training_config.get("scale", 0.2),
            shear=training_config.get("shear", 0.0),
            perspective=training_config.get("perspective", 0.0),
            mosaic=training_config.get("mosaic", 0.2),
            close_mosaic=training_config.get("close_mosaic", 10),
            mixup=training_config.get("mixup", 0.0),

            # -- Hardware & output --
            pretrained=training_config.get("pretrained", True),
            device=training_config.get("device", 0),
            workers=training_config.get("workers", 4),
            name=run_name,
            exist_ok=training_config.get("exist_ok", True),
            project=training_config.get("project", None),
            save=training_config.get("save", True),
            plots=training_config.get("plots", True),
            verbose=training_config.get("verbose", True),
        )

        print("\nTraining complete")
        print("=" * 60)
        print("Training output paths")
        print("=" * 60)

        trainer = getattr(model, "trainer", None)

        if trainer is not None:
            print(f"save_dir: {trainer.save_dir}")
            print(f"best    : {trainer.best}")
            print(f"last    : {trainer.last}")
            best_path = trainer.best
        else:
            print("[WARNING] model.trainer not found. Falling back to current model.")
            best_path = None

        print("=" * 60)

        # -------------------------------------------------------
        # Validate explicit best checkpoint
        # -------------------------------------------------------

        print("\nRunning validation...")

        if best_path is not None and Path(best_path).exists():
            print(f"Validating explicit best checkpoint: {best_path}")
            val_model = YOLO(str(best_path))
        else:
            print("[WARNING] Could not find best.pt. Validating current model object.")
            val_model = model

        val_results = val_model.val(
            data=data_yaml,
            conf=training_config.get("val_conf", 0.001),
            iou=training_config.get("val_iou", 0.5),
            imgsz=training_config.get("imgsz", 640),
            device=training_config.get("device", 0),
        )

        val_log = {
            "val/box_mAP50": val_results.box.map50,
            "val/box_mAP50-95": val_results.box.map,
            "val/box_precision": val_results.box.mp,
            "val/box_recall": val_results.box.mr,
        }

        if hasattr(val_results, "pose") and val_results.pose is not None:
            val_log.update(
                {
                    "val/pose_mAP50": val_results.pose.map50,
                    "val/pose_mAP50-95": val_results.pose.map,
                    "val/pose_precision": val_results.pose.mp,
                    "val/pose_recall": val_results.pose.mr,
                }
            )

        wandb.log(val_log)

    # -------------------------------------------------------
    # Results
    # -------------------------------------------------------

    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    print(f"Box  mAP@0.5      : {val_results.box.map50:.8f}")
    print(f"Box  mAP@0.5:0.95 : {val_results.box.map:.8f}")
    print(f"Box  precision    : {val_results.box.mp:.8f}")
    print(f"Box  recall       : {val_results.box.mr:.8f}")

    if hasattr(val_results, "pose") and val_results.pose is not None:
        print("-" * 60)
        print(f"Pose mAP@0.5      : {val_results.pose.map50:.8f}")
        print(f"Pose mAP@0.5:0.95 : {val_results.pose.map:.8f}")
        print(f"Pose precision    : {val_results.pose.mp:.8f}")
        print(f"Pose recall       : {val_results.pose.mr:.8f}")

    print("=" * 60)


if __name__ == "__main__":
    main()