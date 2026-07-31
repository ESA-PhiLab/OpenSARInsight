"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: inference.py

Description:
    Unified RFI inference script for both large and small segmentation models.

    The script always:
        1. Evaluates the selected model on the test set.
        2. Runs CPU/GPU profiling.

    The script optionally:
        1. Generates prediction masks.
        2. Runs RFI post-processing on the generated masks.

Model selection:
    Use --model-size to choose which model to run:
        --model-size large
        --model-size small

Model defaults:
    Large model:
        threshold = 0.404
        stride_multiple = 16

    Small model:
        threshold = 0.325
        stride_multiple = 8

Path configuration:
    Default paths are defined in get_run_config(), but they can be overridden
    from the command line using:
        --cfg-path
        --weights
        --thr
        --test-dir
        --postprocessing-output-dir

Post-processing:
    RFI post-processing is optional and runs only with:
        --run-postprocessing

    When post-processing is enabled, prediction masks are generated first,
    then post-processing is applied to those masks.

    A custom prediction mask directory must be provided:
        --prediction-mask-dir /path/to/prediction_masks

Example commands:
    Run large model evaluation and profiling with default paths:

        python pipeline/RFI_usecase/inference.py \
            --model-size large

    Run small model evaluation and profiling with default paths:

        python pipeline/RFI_usecase/inference.py \
            --model-size small

    Run large model with custom paths:

        python pipeline/RFI_usecase/inference.py \
            --model-size large \
            --config /path/to/config-rfi-large.yaml \
            --weights /path/to/large_model.pth \
            --thr 0.404 \
            --test-dir /path/to/test_set
            

    Run small model with custom paths and post-processing:

        python pipeline/RFI_usecase/inference.py \
            --model-size small \
            --config /path/to/config-rfi-kd.yaml \
            --weights /path/to/student_model.pth \
             --thr 0.325 \
            --test-dir /path/to/test_set \
            --run-postprocessing \
            --prediction-mask-dir /path/to/small_prediction_masks \
            --postprocessing-output-dir /path/to/small_postprocessing_outputs \
            --threshold-db 5.0 \
            --filter-mode remove_high_rfi

    Run large model with custom paths and post-processing without saving images:
        python pipeline/RFI_usecase/inference.py \
           --model-size large \
           --run-postprocessing \
           --prediction-mask-dir /path/to/large_prediction_masks \
           --postprocessing-only

        

Important:
    --prediction-mask-dir is required only when using --run-postprocessing.

History:
    - 2025-12-15:
        First version of the inference code is ready.
    - 2026-04-25:
        Integrating both large and small model with post-processing option is ready.
    - 2026-04-01:
        Added argparse overrides for config path, weights path, test directory,
        stride multiple, and profiling input size and final version is ready.

TODO: None.

---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-12-15

© Copyright INDRA DEIMOS, 2025. All rights reserved.
---------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from pipeline.RFI_usecase.models.unet import UNet
from pipeline.RFI_usecase.models.unet_small import UNetSmall
from pipeline.RFI_usecase.utils.utilities import resize_for_loss
from pipeline.RFI_usecase.post_processing.profiler import profile_inference
from pipeline.RFI_usecase.post_processing.rfi_postprocessing import (
    process_prediction_folder,
)
from pipeline.RFI_usecase.train_rfi_large import make_val_cfg_from_train_cfg
from pipeline.RFI_usecase.utils.rc_dataloader import (
    RFI4ChannelDataset,
    build_transforms_from_cfg,
    collate_pad_validmask,
    load_yaml_config,
    rfi_mask_resolver,
)

from dataset_generation_scripts.utils import get_config
cfg = get_config("MODEL_USECASES_PATH")


# ------------------------- Logging ------------------------- #
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ----------------------------- Silence noisy dataloader logs -----------------------------
logging.getLogger("pipeline.RFI_usecase.utils.rc_dataloader").setLevel(
    logging.WARNING)
logging.getLogger("RFI4ChannelDataset").setLevel(logging.WARNING)


# ----------------------------- Model-specific run configuration -----------------------------
def get_run_config(model_size: str) -> dict:
    """
    Return default paths and model-specific settings.

    Large model:
        threshold = 0.404
        stride_multiple = 16

    Small model:
        threshold = 0.325
        stride_multiple = 8
    """

    test_dir = cfg["rfi_inference"]["test_dir"]

    if model_size == "large":
        return {
            "cfg_path": Path(cfg["rfi_inference"]["unrestricted_model"]["cfg_path"]),
            "weights": Path(cfg["rfi_inference"]["unrestricted_model"]["weights"]),
            "test_dir": test_dir,
            "thr": cfg["rfi_inference"]["unrestricted_model"]["thr"],
            "stride_multiple": cfg["rfi_inference"]["unrestricted_model"]["stride_multiple"],
            "input_shape": tuple(cfg["rfi_inference"]["unrestricted_model"]["input_shape"]),
            "postprocessing_output_dir": Path(
                cfg["rfi_inference"]["unrestricted_model"]["postprocessing_output_dir"]
            ),
        }

    if model_size == "small":
        return {
            "cfg_path": Path(cfg["rfi_inference"]["lightweight_model"]["cfg_path"]),
            "weights": Path(cfg["rfi_inference"]["lightweight_model"]["weights"]),
            "test_dir": test_dir,
            "thr": cfg["rfi_inference"]["lightweight_model"]["thr"],
            "stride_multiple": cfg["rfi_inference"]["lightweight_model"]["stride_multiple"],
            "input_shape": tuple(cfg["rfi_inference"]["lightweight_model"]["input_shape"]),
            "postprocessing_output_dir": Path(
                cfg["rfi_inference"]["lightweight_model"]["postprocessing_output_dir"]
            ),
        }

    raise ValueError(
        f"Unknown model_size={model_size}. Expected 'large' or 'small'."
    )

# ----------------------------- Power measurement helpers -----------------------------
def read_cpu_energy_joules() -> Optional[float]:
    """
    Read CPU package energy using Intel RAPL.

    The file reports cumulative energy in microjoules.
    This function converts it to joules.
    """

    rapl_path = Path("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj")

    if not rapl_path.exists():
        return None

    try:
        energy_microjoules = float(rapl_path.read_text().strip())
        return energy_microjoules / 1_000_000.0

    except PermissionError:
        LOGGER.warning(
            "CPU RAPL file exists but permission is denied: %s",
            rapl_path,
        )
        return None

    except Exception as exc:
        LOGGER.warning("Could not read CPU RAPL energy: %s", exc)
        return None


def compute_cpu_power_watts(
    energy_before_j: Optional[float],
    energy_after_j: Optional[float],
    elapsed_seconds: float,
) -> Optional[float]:
    """
    Compute average CPU power in watts from Intel RAPL energy readings.

    Formula:
        power_watts = energy_used_joules / elapsed_seconds
    """

    if energy_before_j is None or energy_after_j is None:
        return None

    if elapsed_seconds <= 0:
        return None

    energy_used_j = energy_after_j - energy_before_j

    if energy_used_j < 0:
        return None

    return energy_used_j / elapsed_seconds


def read_gpu_power_watts() -> Optional[float]:
    """
    Read current GPU power draw using nvidia-smi.
    """

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        lines = result.stdout.strip().splitlines()

        if not lines:
            return None

        return float(lines[0].strip())

    except Exception:
        return None


class PowerSampler:
    """
    Periodically samples power in a background thread.
    """

    def __init__(
        self,
        read_power_fn: Callable[[], Optional[float]],
        interval_seconds: float = 0.1,
    ) -> None:
        self.read_power_fn = read_power_fn
        self.interval_seconds = interval_seconds
        self.samples: list[float] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """
        Start power sampling.
        """

        self.samples.clear()
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._sample_loop,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """
        Stop power sampling.
        """

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join()

    def _sample_loop(self) -> None:
        """
        Internal loop used by the sampling thread.
        """

        while not self._stop_event.is_set():
            power_watts = self.read_power_fn()

            if power_watts is not None:
                self.samples.append(power_watts)

            time.sleep(self.interval_seconds)

    def average_power_watts(self) -> Optional[float]:
        """
        Return average sampled power in watts.
        """

        if not self.samples:
            return None

        return sum(self.samples) / len(self.samples)


# ----------------------------- Data loader -----------------------------
def _build_loader(
    data_dir: Path,
    tfm_cfg: dict,
    batch_size: int,
    num_workers: int,
    stride_multiple: int,
    ignore_index: int,
    device: torch.device,
    shuffle: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Build DataLoader for RFI4ChannelDataset.
    """

    tfms = build_transforms_from_cfg(tfm_cfg, split="val")

    dataset = RFI4ChannelDataset(
        data_dir=str(data_dir),
        use_masks=True,
        mask_resolver=rfi_mask_resolver,
        transform=tfms,
        strict_missing_masks=False,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=lambda batch: collate_pad_validmask(
            batch,
            enforce_stride_multiple=True,
            stride_multiple=stride_multiple,
            bypass_padding=False,
            mask_ignore_index=ignore_index,
            device=torch.device("cpu"),
        ),
    )


# ----------------------------- Model loading -----------------------------
def _load_model(
    model_size: str,
    model_cfg: dict,
    weights: Path,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load either the large RFI UNet or the small RFI UNetSmall.
    """

    if model_size == "large":
        model = UNet(
            in_channels=int(model_cfg.get("in_channels", 4)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            base_channels=int(model_cfg.get("base_channels", 32)),
            depth=int(model_cfg.get("depth", 4)),
            bilinear=bool(model_cfg.get("bilinear", True)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        ).to(device)

    elif model_size == "small":
        model = UNetSmall(
            in_channels=int(model_cfg.get("in_channels", 4)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            base_channels=int(model_cfg.get("base_channels", 16)),
            depth=int(model_cfg.get("depth", 3)),
            bilinear=bool(model_cfg.get("bilinear", True)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        ).to(device)

    else:
        raise ValueError(
            f"Unknown model_size={model_size}. Expected 'large' or 'small'."
        )

    state = torch.load(str(weights), map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()
    return model


# ----------------------------- Metrics -----------------------------------
@torch.no_grad()
def compute_metrics(
    logits: Tensor,
    mask: Tensor,
    valid: Tensor,
    thr: Optional[float] = 0.99,
    thr_grid: Optional[Sequence[float]] = None,
    select_by: str = "dice",
    ignore_index: int = 255,
) -> Dict[str, float]:
    """
    Compute binary segmentation metrics for a batch.
    """

    if logits.dim() != 4:
        raise ValueError(f"Expected logits shape [B, 1, H, W], got {logits.shape}")

    _, channels, _, _ = logits.shape

    if channels != 1:
        raise ValueError(
            f"compute_metrics assumes 1-channel logits. Got {channels} channels."
        )

    if valid.dim() == 3:
        valid = valid.unsqueeze(1)

    if mask.dim() == 4:
        mask = mask.squeeze(1)

    probs = torch.sigmoid(logits)

    valid_mask = mask != ignore_index
    valid_bool = valid_mask & (valid > 0.5).squeeze(1)

    if not valid_bool.any():
        return {
            "dice": 0.0,
            "iou": 0.0,
            "acc": 0.0,
            "precision": float("nan"),
            "recall": float("nan"),
            "specificity": float("nan"),
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

    threshold = 0.5 if thr is None else float(thr)

    pred = (probs > threshold).to(torch.bool).squeeze(1)
    gt = mask == 1

    pred_flat = pred[valid_bool]
    gt_flat = gt[valid_bool]

    tp = (pred_flat & gt_flat).sum().item()
    fp = (pred_flat & ~gt_flat).sum().item()
    fn = (~pred_flat & gt_flat).sum().item()
    tn = (~pred_flat & ~gt_flat).sum().item()

    total = tp + fp + fn + tn

    acc = (tp + tn) / total if total > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    has_positive_gt = (tp + fn) > 0

    if not has_positive_gt:
        precision = float("nan")
        recall = float("nan")
        iou = float("nan")
        dice = float("nan")
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = (
            (2 * tp) / (2 * tp + fp + fn)
            if (2 * tp + fp + fn) > 0
            else 0.0
        )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# ----------------------------- Evaluation ----------------------------------
@torch.no_grad()
def evaluate_test(
    model_size: str,
    cfg_path: Path,
    weights_path: Path,
    test_dir: Path,
    batch_size: int = 4,
    num_workers: int = 4,
    stride_multiple: int = 16,
    ignore_index: int = 255,
    resize_to: Optional[int] = None,
    thr: Optional[float] = 0.7,
) -> Dict[str, float | int]:
    """
    Evaluate either RFI large or RFI small model on the test set.
    """

    model_cfg = load_yaml_config(str(cfg_path))
    tfm_cfg = load_yaml_config(str(model_cfg.get("transforms_cfg")))
    tfm_cfg_val = make_val_cfg_from_train_cfg(tfm_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = _build_loader(
        data_dir=Path(test_dir),
        tfm_cfg=tfm_cfg_val,
        batch_size=batch_size,
        num_workers=num_workers,
        stride_multiple=stride_multiple,
        ignore_index=ignore_index,
        device=device,
        shuffle=False,
    )

    model = _load_model(
        model_size=model_size,
        model_cfg=model_cfg.get("model", {}),
        weights=Path(weights_path),
        device=device,
    )

    picked_thr = float(thr if thr is not None else 0.5)

    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_tn = 0

    for batch in loader:
        rc = batch.rc.to(device, non_blocking=True)
        mask = batch.mask.to(device, non_blocking=True)
        valid = batch.valid_mask.to(device, non_blocking=True)

        logits = model(rc)

        if resize_to is None:
            lo, ma, va = resize_for_loss(
                logits,
                mask,
                valid,
                int(mask.shape[-1]),
            )
        else:
            lo, ma, va = resize_for_loss(
                logits,
                mask,
                valid,
                int(resize_to),
            )

        metrics = compute_metrics(
            logits=lo,
            mask=ma,
            valid=va,
            thr=picked_thr,
            ignore_index=ignore_index,
        )

        global_tp += metrics["tp"]
        global_fp += metrics["fp"]
        global_fn += metrics["fn"]
        global_tn += metrics["tn"]

    total = global_tp + global_fp + global_fn + global_tn

    acc = (global_tp + global_tn) / total if total > 0 else 0.0

    precision = (
        global_tp / (global_tp + global_fp)
        if (global_tp + global_fp) > 0
        else 0.0
    )

    recall = (
        global_tp / (global_tp + global_fn)
        if (global_tp + global_fn) > 0
        else 0.0
    )

    iou = (
        global_tp / (global_tp + global_fp + global_fn)
        if (global_tp + global_fp + global_fn) > 0
        else 0.0
    )

    dice = (
        (2 * global_tp) / (2 * global_tp + global_fp + global_fn)
        if (2 * global_tp + global_fp + global_fn) > 0
        else 0.0
    )

    output = {
        "model_size": model_size,
        "dice": float(dice),
        "iou": float(iou),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "thr": picked_thr,
    }

    print("\nPerformance metrics:")
    for key, value in output.items():
        print(f"{key}: {value}")

    return output


# ----------------------------- Prediction-mask generation for optional post-processing ----------------------------------
@torch.no_grad()
def generate_prediction_masks(
    model_size: str,
    cfg_path: Path,
    weights_path: Path,
    test_dir: Path,
    prediction_mask_dir: Path,
    batch_size: int = 2,
    num_workers: int = 1,
    stride_multiple: int = 16,
    ignore_index: int = 255,
    resize_to: Optional[int] = None,
    thr: float = 0.5,
) -> None:
    """
    Run inference on all VV/VH tile pairs in a folder and save prediction-only PNG masks.

    This follows the previous working implementation:
        - Iterates directly over RFI4ChannelDataset.
        - Uses dataset.groups[idx]["vv_path"] for the output filename.
        - Saves masks as <vv_npy_stem>_pred.png.

    Output mask values:
        0   = background
        255 = predicted RFI
    """

    model_cfg = load_yaml_config(str(cfg_path))
    tfm_cfg = load_yaml_config(str(model_cfg.get("transforms_cfg")))
    tfm_cfg_val = make_val_cfg_from_train_cfg(tfm_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_model(
        model_size=model_size,
        model_cfg=model_cfg.get("model", {}),
        weights=Path(weights_path),
        device=device,
    )

    prediction_mask_dir = Path(prediction_mask_dir)
    prediction_mask_dir.mkdir(parents=True, exist_ok=True)

    dataset = RFI4ChannelDataset(
        data_dir=str(test_dir),
        use_masks=False,
        mask_resolver=rfi_mask_resolver,
        transform=build_transforms_from_cfg(tfm_cfg_val, split="val"),
        strict_missing_masks=False,
    )

    if not hasattr(dataset, "groups"):
        raise AttributeError(
            "RFI4ChannelDataset does not expose dataset.groups. "
            "This function expects dataset.groups[idx]['vv_path'] to exist."
        )

    print(f"\nGenerating prediction masks for {len(dataset)} samples...")
    print(f"Prediction mask dir: {prediction_mask_dir}")

    saved_count = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]

        batch = collate_pad_validmask(
            [sample],
            enforce_stride_multiple=True,
            stride_multiple=stride_multiple,
            bypass_padding=False,
            mask_ignore_index=ignore_index,
            device=device,
        )

        rc = batch.rc
        valid = batch.valid_mask

        logits = model(rc)

        # Use valid mask size as target size when no GT mask is required.
        target_size = int(resize_to) if resize_to is not None else int(valid.shape[-1])

        logits_r = F.interpolate(
            logits,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )

        valid_r = F.interpolate(
            valid.float(),
            size=(target_size, target_size),
            mode="nearest",
        )

        # Crop to the valid region.
        valid_bool = valid_r > 0.5
        valid_2d = valid_bool.squeeze(0).squeeze(0)

        ys, xs = torch.where(valid_2d)

        if ys.numel() == 0:
            y0, y1 = 0, valid_2d.shape[0]
            x0, x1 = 0, valid_2d.shape[1]
        else:
            margin = 2
            y0 = max(int(ys.min().item()) - margin, 0)
            y1 = min(int(ys.max().item()) + 1 + margin, valid_2d.shape[0])
            x0 = max(int(xs.min().item()) - margin, 0)
            x1 = min(int(xs.max().item()) + 1 + margin, valid_2d.shape[1])

        logits_r = logits_r[..., y0:y1, x0:x1]
        valid_r = valid_r[..., y0:y1, x0:x1]

        probs = torch.sigmoid(logits_r)
        pred = (probs > float(thr)).float()
        pred = pred * (valid_r > 0.5).float()

        pred_np = (
            pred.squeeze(0)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype("uint8")
            * 255
        )

        group = dataset.groups[idx]

        if "vv_path" not in group:
            raise KeyError(
                "dataset.groups[idx] does not contain 'vv_path'. "
                f"Available keys: {list(group.keys())}"
            )

        vv_name = Path(group["vv_path"]).stem
        output_path = prediction_mask_dir / f"{vv_name}_pred.png"

        Image.fromarray(pred_np).save(output_path)

        saved_count += 1

    print(f"\nSaved prediction masks: {saved_count}")
    print(f"Prediction mask dir: {prediction_mask_dir}")

def inference_only(
    model_size: str,
    cfg_path: Path,
    weights_path: Path,
    test_dir: Path,
    prediction_mask_dir: Path,
    postprocessing_output_dir: Path,
    batch_size: int = 2,
    num_workers: int = 1,
    stride_multiple: int = 16,
    ignore_index: int = 255,
    resize_to: Optional[int] = None,
    thr: float = 0.5,
    run_postprocessing: bool = True,
    threshold_db: float = 5.0,
    filter_mode: str = "remove_high_rfi",
    filter_level: str = "patch",
    region_stat: str = "p90",
    background_stat: str = "median",
    min_region_area: int = 20,
    ring_inner_iters: int = 3,
    ring_outer_iters: int = 12,
    trim_fraction: float = 0.1,
    save_results: bool = False,
) -> None:
    """
    Run inference only: generate prediction masks and optionally apply post-processing.

    This function does NOT run evaluation or profiling. It is intended for
    use from the main pipeline when only inference output is needed.

    Steps:
        1. Generate prediction masks using the selected model.
        2. Optionally apply RFI post-processing on the generated masks.

    Args:
        model_size: "large" or "small".
        cfg_path: Path to model config YAML.
        weights_path: Path to model weights.
        test_dir: Path to the test data directory.
        prediction_mask_dir: Directory to save prediction masks.
        postprocessing_output_dir: Directory for post-processing outputs.
        batch_size: Inference batch size.
        num_workers: DataLoader workers.
        stride_multiple: Padding alignment stride.
        ignore_index: Label for invalid/padded pixels.
        resize_to: Optional fixed resize size.
        thr: Probability threshold for binary predictions.
        run_postprocessing: Whether to run post-processing (default True).
        threshold_db: dB threshold for post-processing.
        filter_mode: Post-processing filter mode.
        filter_level: Filtering granularity ("region" or "patch").
        region_stat: Statistic for region power summarisation.
        background_stat: Statistic for background power estimation.
        min_region_area: Minimum connected region area in pixels.
        ring_inner_iters: Inner dilation iterations for background ring.
        ring_outer_iters: Outer dilation iterations for background ring.
        trim_fraction: Trim fraction for trimmed_mean background stat.
        save_results: Whether to save comparison images.
    """
    LOGGER.info("Running inference_only (model_size=%s, run_postprocessing=%s)", model_size, run_postprocessing)

    generate_prediction_masks(
        model_size=model_size,
        cfg_path=cfg_path,
        weights_path=weights_path,
        test_dir=test_dir,
        prediction_mask_dir=prediction_mask_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        stride_multiple=stride_multiple,
        ignore_index=ignore_index,
        resize_to=resize_to,
        thr=thr,
    )

    if run_postprocessing:
        process_prediction_folder(
            sar_data_dir=test_dir,
            prediction_mask_dir=prediction_mask_dir,
            output_dir=postprocessing_output_dir,
            threshold_db=threshold_db,
            region_stat=region_stat,
            background_stat=background_stat,
            min_region_area=min_region_area,
            ring_inner_iters=ring_inner_iters,
            ring_outer_iters=ring_outer_iters,
            trim_fraction=trim_fraction,
            pred_suffix="_pred",
            filter_mode=filter_mode,
            filter_level=filter_level,
            save_results=save_results,
        )
        LOGGER.info("inference_only complete. Masks: %s, Post-processed: %s", prediction_mask_dir, postprocessing_output_dir)
    else:
        LOGGER.info("inference_only complete. Masks: %s (post-processing skipped)", prediction_mask_dir)


# ----------------------------- Profiling ----------------------------------
def print_profiling_results(title: str, results: dict) -> None:
    """
    Print profiling results in a consistent format.
    """

    print(f"\n{title}")
    for key, value in results.items():
        print(f"{key}: {value}")


def run_cpu_gpu_profiling(
    model_size: str,
    model_cfg: dict,
    weights_path: Path,
    input_shape: tuple[int, int, int],
    fallback_cpu_power_watts: Optional[float],
    fallback_gpu_power_watts: Optional[float],
) -> None:
    """
    Profile the selected model on CPU and GPU.
    """

    print("\nStarting CPU/GPU profiling...")

    # ------------------------------------------------------------------
    # CPU profiling
    # ------------------------------------------------------------------
    print("\nProfiling CPU...")

    cpu_device = torch.device("cpu")

    cpu_model = _load_model(
        model_size=model_size,
        model_cfg=model_cfg,
        weights=weights_path,
        device=cpu_device,
    )

    cpu_energy_before_j = read_cpu_energy_joules()
    cpu_time_before = time.perf_counter()

    cpu_results = profile_inference(
        model=cpu_model,
        input_shape=input_shape,
        device=cpu_device,
        dtype=torch.float32,
        warmup=1,
        iters=3,
        use_channels_last=False,
    )

    cpu_time_after = time.perf_counter()
    cpu_energy_after_j = read_cpu_energy_joules()

    cpu_elapsed_seconds = cpu_time_after - cpu_time_before

    cpu_power_watts = compute_cpu_power_watts(
        energy_before_j=cpu_energy_before_j,
        energy_after_j=cpu_energy_after_j,
        elapsed_seconds=cpu_elapsed_seconds,
    )

    if cpu_power_watts is not None:
        cpu_power_source = "intel_rapl"
    else:
        cpu_power_watts = fallback_cpu_power_watts
        cpu_power_source = "fallback" if fallback_cpu_power_watts is not None else "unavailable"

    cpu_results["power_watts"] = cpu_power_watts
    cpu_results["power_source"] = cpu_power_source

    print_profiling_results("CPU profiling results:", cpu_results)

    # ------------------------------------------------------------------
    # GPU profiling
    # ------------------------------------------------------------------
    if not torch.cuda.is_available():
        print("\nGPU profiling skipped: CUDA is not available.")
        return

    print("\nProfiling GPU...")

    gpu_device = torch.device("cuda")

    gpu_model = _load_model(
        model_size=model_size,
        model_cfg=model_cfg,
        weights=weights_path,
        device=gpu_device,
    )

    gpu_power_sampler = PowerSampler(
        read_power_fn=read_gpu_power_watts,
        interval_seconds=0.1,
    )

    gpu_power_sampler.start()

    gpu_results = profile_inference(
        model=gpu_model,
        input_shape=input_shape,
        device=gpu_device,
        dtype=torch.float16,
        warmup=5,
        iters=20,
        use_channels_last=True,
    )

    gpu_power_sampler.stop()

    gpu_power_watts = gpu_power_sampler.average_power_watts()

    if gpu_power_watts is not None:
        gpu_power_source = "nvidia-smi"
    else:
        gpu_power_watts = fallback_gpu_power_watts
        gpu_power_source = "fallback" if fallback_gpu_power_watts is not None else "unavailable"

    gpu_results["power_watts"] = gpu_power_watts
    gpu_results["power_source"] = gpu_power_source

    print_profiling_results("GPU profiling results:", gpu_results)


# ----------------------------- CLI ----------------------------------
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run RFI large or small model evaluation and profiling, "
            "with optional RFI post-processing."
        )
    )

    parser.add_argument(
        "--model-size",
        type=str,
        choices=["large", "small"],
        required=True,
        help="Choose which model to run.",
    )

    parser.add_argument(
        "--cfg-path",
        type=Path,
        default=None,
        help=(
            "Path to the model YAML config. "
            "If not provided, the model-specific default config is used."
        ),
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=(
            "Path to model checkpoint weights. "
            "If not provided, the model-specific default checkpoint is used."
        ),
    )

    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help=(
            "Path to the test data directory. "
            "If not provided, the default test directory is used."
        ),
    )

    parser.add_argument(
        "--stride-multiple",
        type=int,
        default=None,
        help=(
            "Stride multiple used during padding. "
            "If not provided, the model-specific default is used."
        ),
    )

    parser.add_argument(
        "--input-height",
        type=int,
        default=None,
        help=(
            "Input height used for CPU/GPU profiling. "
            "If not provided, the model-specific default height is used."
        ),
    )

    parser.add_argument(
        "--input-width",
        type=int,
        default=None,
        help=(
            "Input width used for CPU/GPU profiling. "
            "If not provided, the model-specific default width is used."
        ),
    )

    parser.add_argument(
        "--run-postprocessing",
        action="store_true",
        help=(
            "Run RFI dB post-processing. "
            "Prediction masks are generated automatically before post-processing."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of DataLoader workers.",
    )

    parser.add_argument(
        "--thr",
        type=float,
        default=None,
        help=(
            "Optional threshold override. "
            "If not provided, model-specific default threshold is used."
        ),
    )

    parser.add_argument(
        "--resize-to",
        type=int,
        default=None,
        help="Optional fixed resize size for logits, masks, and valid masks.",
    )

    parser.add_argument(
        "--prediction-mask-dir",
        type=Path,
        default=None,
        help=(
            "Directory where prediction PNG masks will be saved for post-processing. "
            "Required when using --run-postprocessing."
        ),
    )

    parser.add_argument(
        "--postprocessing-output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for post-processed masks, comparisons, and CSV files. "
            "If not provided, a model-specific default is used."
        ),
    )


    parser.add_argument(
    "--postprocessing-only",
    action="store_true",
    help=(
        "Run RFI post-processing without saving post-processed masks, "
        "comparison images, or CSV result files. Prediction masks may still "
        "be generated because post-processing uses them as input."
         ),
   )

    parser.add_argument(
        "--threshold-db",
        type=float,
        default=5.0,
        help="dB threshold for post-processing.",
    )

    parser.add_argument(
        "--filter-mode",
        type=str,
        choices=["keep_high_rfi", "remove_high_rfi"],
        default="remove_high_rfi",
        help="Post-processing filter mode.",
    )

    parser.add_argument(
        "--filter-level",
        type=str,
        choices=["region", "patch"],
        default="region",
        help=(
            "Filtering granularity. "
            "region: filter individual connected regions (slower, more precise). "
            "patch: filter entire patch based on aggregate statistics (faster, simpler)."
        ),
    )

    parser.add_argument(
        "--region-stat",
        type=str,
        choices=["mean", "median", "p90"],
        default="p90",
        help="Statistic used to summarise power inside each predicted region.",
    )

    parser.add_argument(
        "--background-stat",
        type=str,
        choices=["mean", "median", "trimmed_mean"],
        default="median",
        help="Statistic used to estimate local background power.",
    )

    parser.add_argument(
        "--min-region-area",
        type=int,
        default=20,
        help="Minimum connected region area in pixels.",
    )

    parser.add_argument(
        "--ring-inner-iters",
        type=int,
        default=3,
        help="Inner dilation iterations for local background ring.",
    )

    parser.add_argument(
        "--ring-outer-iters",
        type=int,
        default=12,
        help="Outer dilation iterations for local background ring.",
    )

    parser.add_argument(
        "--trim-fraction",
        type=float,
        default=0.1,
        help="Trim fraction used only when background_stat=trimmed_mean.",
    )

    parser.add_argument(
        "--fallback-cpu-power-watts",
        type=float,
        default=165.0,
        help="Fallback CPU power in watts if Intel RAPL measurement is unavailable.",
    )

    parser.add_argument(
        "--fallback-gpu-power-watts",
        type=float,
        default=350.0,
        help="Fallback GPU power in watts if nvidia-smi measurement is unavailable.",
    )

    return parser.parse_args()


# ----------------------------- Main ----------------------------------
def main() -> None:
    """
    Main entry point.

    Evaluation and profiling always run.
    Post-processing runs only when --run-postprocessing is provided.
    """

    args = parse_args()

    run_cfg = get_run_config(args.model_size)

    cfg_path = run_cfg["cfg_path"]
    weights = run_cfg["weights"]
    test_dir = run_cfg["test_dir"]
    default_thr = run_cfg["thr"]
    stride_multiple = run_cfg["stride_multiple"]
    input_shape = run_cfg["input_shape"]

    threshold = default_thr if args.thr is None else args.thr

    prediction_mask_dir = args.prediction_mask_dir

    if args.run_postprocessing and prediction_mask_dir is None:
        raise ValueError(
            "--prediction-mask-dir must be provided when using --run-postprocessing."
        )

    postprocessing_output_dir = (
        run_cfg["postprocessing_output_dir"]
        if args.postprocessing_output_dir is None
        else args.postprocessing_output_dir
    )

    model_cfg = load_yaml_config(str(cfg_path))

    print("\n" + "=" * 80)
    print(f"Running RFI {args.model_size} model")
    print(f"Config: {cfg_path}")
    print(f"Weights: {weights}")
    print(f"Test dir: {test_dir}")
    print(f"Threshold: {threshold}")
    print(f"Stride multiple: {stride_multiple}")
    print("Evaluation: enabled")
    print("Profiling: enabled")

    if args.run_postprocessing:
        print("Post-processing: enabled")
        print(f"Prediction mask dir: {prediction_mask_dir}")
        if args.postprocessing_only:
            print("Post-processing output: disabled generating figures")
        else:
            print(f"Post-processing output dir: {postprocessing_output_dir}")
        print(f"Post-processing threshold dB: {args.threshold_db}")
        print(f"Post-processing filter mode: {args.filter_mode}")
    else:
        print("Post-processing: disabled")

    print("=" * 80)

    evaluate_test(
        model_size=args.model_size,
        cfg_path=cfg_path,
        weights_path=weights,
        test_dir=test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stride_multiple=stride_multiple,
        ignore_index=255,
        resize_to=args.resize_to,
        thr=threshold,
    )

    run_cpu_gpu_profiling(
        model_size=args.model_size,
        model_cfg=model_cfg.get("model", {}),
        weights_path=weights,
        input_shape=input_shape,
        fallback_cpu_power_watts=args.fallback_cpu_power_watts,
        fallback_gpu_power_watts=args.fallback_gpu_power_watts,
    )

    if args.run_postprocessing:
        generate_prediction_masks(
            model_size=args.model_size,
            cfg_path=cfg_path,
            weights_path=weights,
            test_dir=test_dir,
            prediction_mask_dir=prediction_mask_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            stride_multiple=stride_multiple,
            ignore_index=255,
            resize_to=args.resize_to,
            thr=threshold,
        )

        process_prediction_folder(
            sar_data_dir=test_dir,
            prediction_mask_dir=prediction_mask_dir,
            output_dir=postprocessing_output_dir,
            threshold_db=args.threshold_db,
            region_stat=args.region_stat,
            background_stat=args.background_stat,
            min_region_area=args.min_region_area,
            ring_inner_iters=args.ring_inner_iters,
            ring_outer_iters=args.ring_outer_iters,
            trim_fraction=args.trim_fraction,
            pred_suffix="_pred",
            filter_mode=args.filter_mode,
            filter_level=args.filter_level,
            save_results=not args.postprocessing_only,
        )


if __name__ == "__main__":
    main()