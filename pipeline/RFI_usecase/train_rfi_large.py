"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: train_rfi_large.py

Description:
    This is the main training code used for RFI large model training.

CLI
---
python train_rfi_large.py --config /path/to/config.yaml [--transforms /path/to/transforms.yaml] [--device cuda|cpu]

History:
    - 2025-12-08:
        The training code is ready to be used.
    - 2026-02-01:
        The final version is ready.

---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-12-08

© Copyright INDRA DEIMOS, 2025. All rights reserved.
---------------------------------------------------------------------
"""

from __future__ import annotations
from pathlib import Path
import sys
import torch
import logging
import argparse
import copy
from pipeline.RFI_usecase.utils.logging_setup import init_logging
from pipeline.RFI_usecase.utils.rc_dataloader import RFI4ChannelDataset, collate_pad_validmask, rfi_mask_resolver, build_transforms_from_cfg, load_yaml_config
from pipeline.RFI_usecase.models.unet import UNet
from pipeline.RFI_usecase.utils.utilities import set_seed, save_weights, ensure_dir, make_optimizer, make_scheduler, train_one_epoch, validate_one_epoch, estimate_pos_weight, evaluate_one_epoch_fixed_thr
from pipeline.RFI_usecase.utils.plotting import save_metrics_csv, plot_training_curves
import numpy as np
from torch.utils.data import Subset
import random
from dataset_generation_scripts.utils import get_config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

cfg_mu = get_config("MODEL_USECASES_PATH")



# ------------------------- Logging ------------------------- #
init_logging(level=logging.WARNING)  # Set default logging level to WARNING; individual loggers will be adjusted below
LOGGER = logging.getLogger(__name__)


#----------------------------mute noisy modules-----------------------------#
for name in (
    "pipeline.RFI_usecase.utils.rc_dataloader",
    "pipeline.RFI_usecase.utils.utilities",
    "PIL",
    "PIL.PngImagePlugin",
    "matplotlib",
    "matplotlib.font_manager",
):
    lg = logging.getLogger(name)
    lg.setLevel(logging.ERROR)      # or WARNING
    lg.propagate = False            # stop bubbling to root
    for h in list(lg.handlers):     # if module attached its own handler(s)
        h.setLevel(logging.ERROR)


#----------------------------CLI-----------------------------#
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a U-Net for RFI segmentation from scratch."
    )

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--transforms",
        default=None,
        type=str,
        help="Optional path to the transforms YAML file. Overrides config.transforms_cfg.",
    )

    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu"],
        help="Optional device override. Overrides config.device.",
    )

    return parser.parse_args()

# ------------------------- Data transformation utilities & build data loader ------------------------- #

def make_val_cfg_from_train_cfg(cfg: dict) -> dict:
    """
    Return a copy of the training transform config with all augmentations disabled.

    This is used to build a deterministic validation transform config from the
    training configuration by setting any block containing an "enable" flag to False.
    """
    val = copy.deepcopy(cfg)
    augs = val.get("augs", {})
    for _, block in augs.items():
        if isinstance(block, dict) and "enable" in block:
            block["enable"] = False 
    return val


def build_dataloader(data_dir: Path, cfg_yaml: dict, batch_size: int, num_workers: int, stride_multiple: int, ignore_index: int, shuffle: bool, split: str):
    """
    Build and return a DataLoader for the RFI dataset.

    Args:
        data_dir: Path to dataset directory.
        cfg_yaml: Transform configuration dictionary.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
        stride_multiple: Padding alignment used in collate function.
        ignore_index: Label value used to ignore padded/invalid pixels.
        shuffle: Whether to shuffle the dataset.
        split: Transform split name (e.g. "train" or "val").

    Returns:
        A torch.utils.data.DataLoader configured with the dataset and collate fn.
    """

    tfms = build_transforms_from_cfg(cfg_yaml, split=split)
    ds = RFI4ChannelDataset(data_dir=str(data_dir), use_masks=True, mask_resolver=rfi_mask_resolver, transform=tfms, strict_missing_masks=True)

    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
        collate_fn=lambda b: collate_pad_validmask(
            b, enforce_stride_multiple=True, stride_multiple=stride_multiple,
            bypass_padding=False, mask_ignore_index=ignore_index, device=torch.device("cpu")),
    )


# ------------------------- Main ------------------------------ #
def main():
    cfg = load_yaml_config(cfg_mu["rfi_train"]["unrestricted_model"]["cfg_path"])
    tfm_cfg_path = cfg.get("transforms_cfg", cfg_mu["rfi_train"]["unrestricted_model"]["tfm_cfg_path"])
    tfm_cfg = load_yaml_config(tfm_cfg_path)

    device_pref = str(cfg.get("device", "cuda")).lower()
    data_cfg  = cfg.get("data", {})
    loader_cfg = cfg.get("loader", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    save_cfg  = cfg.get("save", {})

    train_dir = Path(data_cfg["train_dir"])
    val_dir   = Path(data_cfg["val_dir"])
    save_dir  = Path(save_cfg.get("dir", cfg_mu["rfi_train"]["unrestricted_model"]["save_dir"]))
    ensure_dir(save_dir)

    device = torch.device("cuda" if (device_pref == "cuda" and torch.cuda.is_available()) else "cpu")
    LOGGER.info("Device: %s (CUDA available: %s)", device, torch.cuda.is_available())

    set_seed(int(train_cfg.get("seed", 42)))

    batch_size      = int(loader_cfg.get("batch_size"))
    train_batch_size = int(loader_cfg.get("train_batch_size", loader_cfg.get("batch_size")))
    val_batch_size   = int(loader_cfg.get("val_batch_size",   loader_cfg.get("batch_size")))
    num_workers     = int(loader_cfg.get("num_workers"))
    stride_multiple = int(loader_cfg.get("stride_multiple"))
    ignore_index    = int(loader_cfg.get("ignore_index"))
    
   
    train_loader = build_dataloader(train_dir, tfm_cfg, train_batch_size, num_workers, stride_multiple, ignore_index, True, split="train") # shuffle:True
    val_loader   = build_dataloader(val_dir, make_val_cfg_from_train_cfg(tfm_cfg), val_batch_size, num_workers, stride_multiple, ignore_index, False, split="val")

    
    model = UNet(
        in_channels=int(model_cfg.get("in_channels")),
        out_channels=int(model_cfg.get("out_channels")),
        base_channels=int(model_cfg.get("base_channels")),
        depth=int(model_cfg.get("depth")),
        bilinear=bool(model_cfg.get("bilinear")),
        dropout=float(model_cfg.get("dropout")),
    ).to(device)


    epochs        = int(train_cfg.get("epochs"))
    lr            = float(train_cfg.get("lr"))
    wd            = float(train_cfg.get("weight_decay"))
    optimizer_name= str(train_cfg.get("optimizer"))
    scheduler_name= str(train_cfg.get("scheduler"))
    warmup_epochs = int(train_cfg.get("warmup_epochs"))
    grad_clip     = float(train_cfg.get("grad_clip"))
    eval_every    = int(train_cfg.get("eval_every"))
    resize_to     = str(train_cfg.get("resize_to"))
    optimizer = make_optimizer(model, optimizer_name, lr, wd)
    scheduler = make_scheduler(optimizer, scheduler_name, epochs, warmup_epochs)


    # pos_weight
    posw = train_cfg.get("pos_weight")
    if isinstance(posw, (int, float)): pos_weight_value = float(posw)
    elif isinstance(posw, str) and posw.lower() == "none": pos_weight_value = None
    else:
        # simple on-the-fly estimator using a couple batches
        pos_weight_value = estimate_pos_weight(train_loader, max_batches=None, resize_to=resize_to, device=device)


    start_epoch, best_iou = 0, 0.0
    history = {k: [] for k in ["train_loss","val_loss","train_acc","val_acc","train_dice","val_dice","train_iou","val_iou", "train_precision", "val_precision","train_recall" ,"val_recall"]}
    rows = []


    # effective batch size target for gradient accumulation (adjust based on GPU memory)
    eff_batch_target = 64 
    accumulate_steps = max(1, eff_batch_target // train_batch_size)  # 64//2 = 32
    for epoch in range(start_epoch, epochs):
        tr = train_one_epoch(model, train_loader, optimizer, device, resize_to=None, pos_weight=pos_weight_value, grad_clip=grad_clip, metric_thr=None, accumulate_steps=accumulate_steps)

        if scheduler: scheduler.step()
        LOGGER.info("Epoch %3d/%3d | Train: loss=%.5f | LR=%.3e",
                    epoch+1, epochs, tr.loss, optimizer.param_groups[0]["lr"])
        


        if ((epoch + 1) % max(1, eval_every)) == 0:
            va, val_thr = validate_one_epoch(model, val_loader, device, resize_to=None,
                                    pos_weight=pos_weight_value)
            LOGGER.info("Epoch %3d/%3d | Valid: loss=%.5f dice=%.4f iou=%.4f acc=%.4f pre=%.4f rec=%.4f | thr=%.3f",
                        epoch+1, epochs, va.loss, va.dice, va.iou, va.acc, va.precision, va.recall, val_thr)
            
            

# ------------------------- Evaluate train at the same val_thr ------------------------------ #

            train_eval = evaluate_one_epoch_fixed_thr(
                model, train_loader, device,
                resize_to=None,
                pos_weight=pos_weight_value,
                fixed_thr=val_thr,
            )
            LOGGER.info(
                "Epoch %3d/%3d | Train@val_thr: dice=%.4f iou=%.4f acc=%.4f pre=%.4f rec=%.4f | thr=%.3f",
                epoch+1, epochs,
                train_eval.dice, train_eval.iou,
                train_eval.acc, train_eval.precision, train_eval.recall,
                val_thr,
            )

            
            history["train_loss"].append(tr.loss); history["val_loss"].append(va.loss)
            history["train_acc"].append(train_eval.acc);   history["val_acc"].append(va.acc)
            history["train_dice"].append(train_eval.dice); history["val_dice"].append(va.dice)
            history["train_iou"].append(train_eval.iou);   history["val_iou"].append(va.iou)
            history["train_precision"].append(train_eval.precision); history["val_precision"].append(va.precision)
            history["train_recall"].append(train_eval.recall);       history["val_recall"].append(va.recall)


            save_weights(model, save_dir / "last.pth")
            if va.iou > best_iou:
                best_iou = va.iou
                save_weights(model, save_dir / "best.pth")
            # Save one checkpoint per epoch (model state_dict only)
            save_weights(model, save_dir / f"epoch_{epoch+1:03d}.pth")

            rows.append({
                "epoch": epoch+1, "train_loss": tr.loss, "val_loss": va.loss,
                "train_acc": train_eval.acc, "val_acc": va.acc,
                "train_dice": train_eval.dice, "val_dice": va.dice,
                "train_iou": train_eval.iou, "val_iou": va.iou,
                "train_precision": train_eval.precision, "val_precision": va.precision,
                "train_recall": train_eval.recall, "val_recall": va.recall,
                "lr": optimizer.param_groups[0]["lr"],
                "thr": val_thr,
            })

    save_metrics_csv(save_dir / "metrics.csv", rows)

    plot_training_curves(
    history["train_loss"], history["val_loss"],
    history["train_acc"],  history["val_acc"],
    train_dice=history.get("train_dice"),
    val_dice=history.get("val_dice"),
    train_iou=history.get("train_iou"),
    val_iou=history.get("val_iou"),
    train_precision=history.get("train_precision"),
    val_precision=history.get("val_precision"),
    train_recall=history.get("train_recall"),
    val_recall=history.get("val_recall"),
    output_dir=str(save_dir)
)
    LOGGER.info("Training complete. Best validation iou: %.4f", best_iou)


if __name__ == "__main__":
    main()