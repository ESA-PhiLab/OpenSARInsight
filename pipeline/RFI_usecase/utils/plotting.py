"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: plotting.py

Description:
    Utilities for saving per-epoch training metrics and plotting training
    and validation curves during model training.

History:
    - 2025-12-08:
        The script is ready to be used during model training.  
    
---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-12-08

© Copyright INDRA DEIMOS, 2025. All rights reserved.
-----
"""


from __future__ import annotations
import os
import csv
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from pipeline.RFI_usecase.utils.logging_setup import init_logging
import logging
import numpy as np
 

# ------------------------- Logging ------------------------- #
init_logging(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


# ------------------------- Plotting ------------------------- #
def save_metrics_csv(path: Path, rows: list[dict]) -> None:
    """
    Save per-epoch metrics to CSV.

    Args:
        path (Path): Output path where the CSV file will be saved.
        rows (list[dict]): List of dictionaries containing metric values for each epoch.
            Each dictionary represents one row in the CSV file.
    """
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    LOGGER.info("Saved metrics CSV: %s", path)


def _plot_pair(train, val, title, ylabel, fname, output_dir):
    """
    Plot a pair of training and validation metric curves and save the figure.

    """

    os.makedirs(output_dir, exist_ok=True)
    t = np.asarray(train, dtype=float)
    v = np.asarray(val,   dtype=float)

    plt.figure()
    plt.plot(t, label=f"Train {ylabel}")
    plt.plot(v, label=f"Val {ylabel}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()

def plot_training_curves(
    train_losses, val_losses,
    train_acc,    val_acc,
    train_dice=None, val_dice=None,
    train_iou=None,  val_iou=None,
    train_precision=None, val_precision=None,
    train_recall=None,  val_recall=None,
    output_dir="."
):
    """
    Plot and save training and validation curves for available metrics.

    """
    # Required
    _plot_pair(train_losses, val_losses, "Training vs Validation Loss", "Loss", "loss_curve.png", output_dir)
    _plot_pair(train_acc,    val_acc,    "Train & Validation Accuracy", "Accuracy", "acc_curve.png", output_dir)

    # Optional (only plot if provided)
    if train_dice is not None and val_dice is not None:
        _plot_pair(train_dice, val_dice, "Train & Validation Dice", "Dice", "dice_curve.png", output_dir)

    if train_iou is not None and val_iou is not None:
        _plot_pair(train_iou, val_iou, "Train & Validation IoU", "IoU", "iou_curve.png", output_dir)

    if train_precision is not None and val_precision is not None:
        _plot_pair(train_precision, val_precision, "Train & Validation Precision", "Precision", "precision_curve.png", output_dir)

    if train_recall is not None and val_recall is not None:
        _plot_pair(train_recall, val_recall, "Train & Validation Recall", "Recall", "recall_curve.png", output_dir)
