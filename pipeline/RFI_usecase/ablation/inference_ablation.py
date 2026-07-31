"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: inference_ablation.py

Description:
    Small-model RFI inference script for UNetSmall segmentation models.

    This script supports:
        1. Small model trained from scratch.
        2. Small student model trained with Knowledge Distillation.
        3. Base-channel ablation, for example base_channels=6, and 8.
        4. Test-set evaluation.
        5. CPU/GPU profiling.
        6. Optional prediction mask generation.
        7. Optional RFI post-processing.

Important:
    Knowledge Distillation is only used during training.
    During inference, only the trained small model checkpoint is loaded.

Model:
    UNetSmall only.

Examples:
    Run scratch small model with base_channels=16:

        python pipeline/RFI_usecase/ablation/inference_ablation.py \
            --config .../ablation/config-rfi-small-scratch.yaml \
            --weights .../path/to/checkpoint.pth\
            --base-channels 16 \
            --test-dir path/to/test_set \
            --thr 0.27

    Run KD small student with base_channels=8:

        python pipeline/RFI_usecase/ablation/inference_ablation.py \
            --config .../ablation/config-rfi-kd-ablation.yaml \
            --weights .../path/to/checkpoint.pth \
            --base-channels 8 \
            --test-dir path/to/test_set \
            --thr 0.313


    Run KD small student with base_channels=6:

    python pipeline/RFI_usecase/ablation/inference_ablation.py \
        --config .../ablation/config-rfi-kd-ablation.yaml \
        --weights .../path/to/checkpoint.pth\
        --base-channels 6 \
        --test-dir path/to/test_set \
        --thr 0.41

    Run with post-processing:

        python pipeline/RFI_usecase/ablation/inference_ablation.py \
            --config .../ablation/config-rfi-kd-ablation.yaml \
            --weights .../path/to/checkpoint.pth \
            --base-channels 8 \
            --test-dir path/to/test_set \
            --thr 0.313 \
            --run-postprocessing \
            --prediction-mask-dir /path/to/prediction_masks \
            --postprocessing-output-dir /path/to/postprocessing_outputs \
            --threshold-db 5.0 \
            --filter-mode remove_high_rfi

History:
    - 2026-05-07:
        Small-only inference script for scratch and KD UNetSmall models.

TODO: None.

---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2026-05-07

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import copy
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from pipeline.RFI_usecase.models.unet_small import UNetSmall
from pipeline.RFI_usecase.utils.logging_setup import init_logging
from pipeline.RFI_usecase.post_processing.profiler import profile_inference
from pipeline.RFI_usecase.post_processing.rfi_postprocessing import (
    process_prediction_folder,
)
from pipeline.RFI_usecase.utils.rc_dataloader import (
    RFI4ChannelDataset,
    build_transforms_from_cfg,
    collate_pad_validmask,
    load_yaml_config,
    rfi_mask_resolver,
)
from pipeline.RFI_usecase.utils.utilities import resize_for_loss

init_logging(level=logging.INFO)  
LOGGER = logging.getLogger(__name__)


# ----------------------------- Logging control ----------------------------- #
for logger_name in (
    "pipeline.RFI_usecase.utils.rc_dataloader",
    "pipeline.RFI_usecase.utils.utilities",
    "RFI4ChannelDataset",
    "PIL",
    "PIL.PngImagePlugin",
):
    logging.getLogger(logger_name).setLevel(logging.WARNING)


# ----------------------------- Transform config helper ----------------------------- #
def make_val_cfg_from_train_cfg(cfg: dict) -> dict:
    """
    Return a copy of the training transform config with augmentations disabled.

    Args:
        cfg: Transform configuration dictionary.

    Returns:
        Validation transform configuration dictionary.
    """
    val_cfg = copy.deepcopy(cfg)
    augs = val_cfg.get("augs", {})

    for _, block in augs.items():
        if isinstance(block, dict) and "enable" in block:
            block["enable"] = False

    return val_cfg


# ----------------------------- Power measurement helpers ----------------------------- #
def read_cpu_energy_joules() -> Optional[float]:
    """
    Read CPU package energy using Intel RAPL.

    Returns:
        Energy in joules, or None if unavailable.
    """
    rapl_path = Path("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj")

    if not rapl_path.exists():
        return None

    try:
        energy_microjoules = float(rapl_path.read_text().strip())
        return energy_microjoules / 1_000_000.0

    except PermissionError:
        LOGGER.warning("CPU RAPL permission denied: %s", rapl_path)
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
    Compute average CPU power in watts.

    Args:
        energy_before_j: Energy before profiling in joules.
        energy_after_j: Energy after profiling in joules.
        elapsed_seconds: Profiling elapsed time.

    Returns:
        Average CPU power in watts, or None.
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

    Returns:
        GPU power in watts, or None.
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
        Start background power sampling.
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
        Stop background power sampling.
        """
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join()

    def _sample_loop(self) -> None:
        """
        Internal sampling loop.
        """
        while not self._stop_event.is_set():
            power_watts = self.read_power_fn()

            if power_watts is not None:
                self.samples.append(power_watts)

            time.sleep(self.interval_seconds)

    def average_power_watts(self) -> Optional[float]:
        """
        Return the average sampled power.

        Returns:
            Average power in watts, or None.
        """
        if not self.samples:
            return None

        return sum(self.samples) / len(self.samples)


# ----------------------------- Data loader ----------------------------- #
def build_loader(
    data_dir: Path,
    tfm_cfg: dict,
    batch_size: int,
    num_workers: int,
    stride_multiple: int,
    ignore_index: int,
    shuffle: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Build DataLoader for RFI4ChannelDataset.

    Args:
        data_dir: Test dataset directory.
        tfm_cfg: Validation transform configuration.
        batch_size: Batch size.
        num_workers: Number of DataLoader workers.
        stride_multiple: Padding stride multiple.
        ignore_index: Ignore label value.
        shuffle: Whether to shuffle samples.

    Returns:
        DataLoader instance.
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


# ----------------------------- Model loading ----------------------------- #
def build_small_model(
    model_cfg: dict,
    device: torch.device,
) -> torch.nn.Module:
    """
    Build the UNetSmall model.

    Args:
        model_cfg: Model configuration dictionary.
        device: Target torch device.

    Returns:
        UNetSmall model.
    """
    model = UNetSmall(
        in_channels=int(model_cfg.get("in_channels", 4)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 16)),
        depth=int(model_cfg.get("depth", 3)),
        bilinear=bool(model_cfg.get("bilinear", True)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)

    return model


def normalise_state_dict_keys(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """
    Remove common wrapper prefixes from checkpoint keys.

    Args:
        state_dict: Raw checkpoint state dict.

    Returns:
        Cleaned state dict.
    """
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]

        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]

        cleaned[new_key] = value

    return cleaned


def extract_state_dict(checkpoint: object, weights_path: Path) -> dict[str, Tensor]:
    """
    Extract a model state dict from common checkpoint formats.

    Args:
        checkpoint: Loaded checkpoint object.
        weights_path: Checkpoint path, used for error messages.

    Returns:
        Model state dict.

    Raises:
        TypeError: If checkpoint format is unsupported.
    """
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        else:
            state = checkpoint

        if isinstance(state, dict):
            return state

    raise TypeError(
        f"Unsupported checkpoint format in {weights_path}. "
        "Expected a state dict or a dict containing 'state_dict' or 'model_state_dict'."
    )


def load_small_model(
    model_cfg: dict,
    weights: Path,
    device: torch.device,
) -> torch.nn.Module:
    """
    Build and load a UNetSmall model.

    Args:
        model_cfg: Model configuration dictionary.
        weights: Path to checkpoint.
        device: Target torch device.

    Returns:
        Loaded model in eval mode.
    """
    model = build_small_model(model_cfg=model_cfg, device=device)

    checkpoint = torch.load(str(weights), map_location=device)
    state = extract_state_dict(checkpoint=checkpoint, weights_path=weights)
    state = normalise_state_dict_keys(state)

    model.load_state_dict(state, strict=True)
    model.eval()

    return model


# ----------------------------- Metrics ----------------------------- #
@torch.no_grad()
def compute_metrics(
    logits: Tensor,
    mask: Tensor,
    valid: Tensor,
    thr: Optional[float] = 0.5,
    thr_grid: Optional[Sequence[float]] = None,
    select_by: str = "dice",
    ignore_index: int = 255,
) -> Dict[str, float]:
    """
    Compute binary segmentation metrics for a batch.

    Args:
        logits: Raw model logits with shape [B, 1, H, W].
        mask: Ground-truth mask with shape [B, H, W] or [B, 1, H, W].
        valid: Valid mask with shape [B, H, W] or [B, 1, H, W].
        thr: Prediction threshold.
        thr_grid: Unused placeholder for compatibility.
        select_by: Unused placeholder for compatibility.
        ignore_index: Ignore label value.

    Returns:
        Dictionary of metrics and confusion counts.
    """
    del thr_grid
    del select_by

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
    threshold = 0.5 if thr is None else float(thr)

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

    pred = (probs > threshold).to(torch.bool).squeeze(1)
    gt = mask == 1

    pred_flat = pred[valid_bool]
    gt_flat = gt[valid_bool]

    tp = int((pred_flat & gt_flat).sum().item())
    fp = int((pred_flat & ~gt_flat).sum().item())
    fn = int((~pred_flat & gt_flat).sum().item())
    tn = int((~pred_flat & ~gt_flat).sum().item())

    total = tp + fp + fn + tn

    acc = (tp + tn) / total if total > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

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
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ----------------------------- Evaluation ----------------------------- #
@torch.no_grad()
def evaluate_test(
    cfg: dict,
    weights_path: Path,
    test_dir: Path,
    batch_size: int,
    num_workers: int,
    stride_multiple: int,
    ignore_index: int,
    resize_to: Optional[int],
    thr: float,
) -> Dict[str, float | int]:
    """
    Evaluate the small model on the test set.

    Args:
        cfg: Loaded YAML configuration.
        weights_path: Path to checkpoint.
        test_dir: Test dataset directory.
        batch_size: Evaluation batch size.
        num_workers: Number of DataLoader workers.
        stride_multiple: Padding stride multiple.
        ignore_index: Ignore label value.
        resize_to: Optional resize size.
        thr: Prediction threshold.

    Returns:
        Global test metrics.
    """
    transforms_cfg_path = cfg.get("transforms_cfg")

    if transforms_cfg_path is None:
        raise ValueError("Config must contain transforms_cfg.")

    tfm_cfg = load_yaml_config(str(transforms_cfg_path))
    tfm_cfg_val = make_val_cfg_from_train_cfg(tfm_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = build_loader(
        data_dir=Path(test_dir),
        tfm_cfg=tfm_cfg_val,
        batch_size=batch_size,
        num_workers=num_workers,
        stride_multiple=stride_multiple,
        ignore_index=ignore_index,
        shuffle=False,
    )

    model = load_small_model(
        model_cfg=cfg.get("model", {}),
        weights=Path(weights_path),
        device=device,
    )

    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_tn = 0

    for batch in loader:
        rc = batch.rc.to(device, non_blocking=True)
        mask = batch.mask.to(device, non_blocking=True)
        valid = batch.valid_mask.to(device, non_blocking=True)

        logits = model(rc)

        target_size = int(resize_to) if resize_to is not None else int(mask.shape[-1])

        logits_r, mask_r, valid_r = resize_for_loss(
            logits,
            mask,
            valid,
            target_size,
        )

        metrics = compute_metrics(
            logits=logits_r,
            mask=mask_r,
            valid=valid_r,
            thr=thr,
            ignore_index=ignore_index,
        )

        global_tp += int(metrics["tp"])
        global_fp += int(metrics["fp"])
        global_fn += int(metrics["fn"])
        global_tn += int(metrics["tn"])

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
        "model": "unet_small",
        "base_channels": int(cfg.get("model", {}).get("base_channels", 16)),
        "depth": int(cfg.get("model", {}).get("depth", 3)),
        "dice": float(dice),
        "iou": float(iou),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "thr": float(thr),
    }

    print("\nPerformance metrics:")
    for key, value in output.items():
        print(f"{key}: {value}")

    return output


# ----------------------------- Prediction mask generation ----------------------------- #
@torch.no_grad()
def generate_prediction_masks(
    cfg: dict,
    weights_path: Path,
    test_dir: Path,
    prediction_mask_dir: Path,
    stride_multiple: int,
    ignore_index: int,
    resize_to: Optional[int],
    thr: float,
) -> None:
    """
    Generate prediction-only PNG masks for post-processing.

    Args:
        cfg: Loaded YAML configuration.
        weights_path: Path to checkpoint.
        test_dir: Test dataset directory.
        prediction_mask_dir: Output directory for prediction masks.
        stride_multiple: Padding stride multiple.
        ignore_index: Ignore label value.
        resize_to: Optional resize size.
        thr: Prediction threshold.
    """
    transforms_cfg_path = cfg.get("transforms_cfg")

    if transforms_cfg_path is None:
        raise ValueError("Config must contain transforms_cfg.")

    tfm_cfg = load_yaml_config(str(transforms_cfg_path))
    tfm_cfg_val = make_val_cfg_from_train_cfg(tfm_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_small_model(
        model_cfg=cfg.get("model", {}),
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
            device=torch.device("cpu"),
        )

        rc = batch.rc.to(device, non_blocking=True)
        valid = batch.valid_mask.to(device, non_blocking=True)

        logits = model(rc)

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


# ----------------------------- Profiling ----------------------------- #
def print_profiling_results(title: str, results: dict) -> None:
    """
    Print profiling results.

    Args:
        title: Result title.
        results: Profiling result dictionary.
    """
    print(f"\n{title}")

    for key, value in results.items():
        print(f"{key}: {value}")


def run_cpu_gpu_profiling(
    cfg: dict,
    weights_path: Path,
    input_shape: tuple[int, int, int],
    fallback_cpu_power_watts: Optional[float],
    fallback_gpu_power_watts: Optional[float],
) -> None:
    """
    Profile UNetSmall inference on CPU and GPU.

    Args:
        cfg: Loaded YAML configuration.
        weights_path: Path to checkpoint.
        input_shape: Input tensor shape as C, H, W.
        fallback_cpu_power_watts: Fallback CPU power.
        fallback_gpu_power_watts: Fallback GPU power.
    """
    print("\nStarting CPU/GPU profiling...")

    print("\nProfiling CPU...")

    cpu_device = torch.device("cpu")

    cpu_model = load_small_model(
        model_cfg=cfg.get("model", {}),
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
        cpu_power_source = (
            "fallback" if fallback_cpu_power_watts is not None else "unavailable"
        )

    cpu_results["power_watts"] = cpu_power_watts
    cpu_results["power_source"] = cpu_power_source

    print_profiling_results("CPU profiling results:", cpu_results)

    if not torch.cuda.is_available():
        print("\nGPU profiling skipped: CUDA is not available.")
        return

    print("\nProfiling GPU...")

    gpu_device = torch.device("cuda")

    gpu_model = load_small_model(
        model_cfg=cfg.get("model", {}),
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
        gpu_power_source = (
            "fallback" if fallback_gpu_power_watts is not None else "unavailable"
        )

    gpu_results["power_watts"] = gpu_power_watts
    gpu_results["power_source"] = gpu_power_source

    print_profiling_results("GPU profiling results:", gpu_results)


# ----------------------------- CLI ----------------------------- #
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run inference, evaluation, profiling, and optional post-processing "
            "for a UNetSmall RFI model trained from scratch or with KD."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config.",
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the trained small-model checkpoint.",
    )

    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Path to the test dataset directory.",
    )

    parser.add_argument(
        "--base-channels",
        type=int,
        default=None,
        help=(
            "Optional override for model.base_channels. "
            "This must match the checkpoint architecture."
        ),
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help=(
            "Optional override for model.depth. "
            "This must match the checkpoint architecture."
        ),
    )

    parser.add_argument(
        "--stride-multiple",
        type=int,
        default=8,
        help="Padding stride multiple. For UNetSmall depth=3, use 8.",
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
        default=0.5,
        help=(
            "Segmentation probability threshold. "
            "Use the best validation threshold for the specific checkpoint."
        ),
    )

    parser.add_argument(
        "--resize-to",
        type=int,
        default=None,
        help="Optional fixed resize size for logits, masks, and valid masks.",
    )

    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip test-set evaluation.",
    )

    parser.add_argument(
        "--skip-profiling",
        action="store_true",
        help="Skip CPU/GPU profiling.",
    )

    parser.add_argument(
        "--profile-height",
        type=int,
        default=1520,
        help="Input height used for profiling.",
    )

    parser.add_argument(
        "--profile-width",
        type=int,
        default=1520,
        help="Input width used for profiling.",
    )

    parser.add_argument(
        "--run-postprocessing",
        action="store_true",
        help="Run RFI dB post-processing. Prediction masks are generated first.",
    )

    parser.add_argument(
        "--prediction-mask-dir",
        type=Path,
        default=None,
        help=(
            "Directory where prediction PNG masks will be saved. "
            "Required when using --run-postprocessing."
        ),
    )

    parser.add_argument(
        "--postprocessing-output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for post-processed masks, comparisons, and CSV files. "
            "Required when using --run-postprocessing."
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
        help="Fallback CPU power in watts if Intel RAPL is unavailable.",
    )

    parser.add_argument(
        "--fallback-gpu-power-watts",
        type=float,
        default=350.0,
        help="Fallback GPU power in watts if nvidia-smi is unavailable.",
    )

    return parser.parse_args()


# ----------------------------- Main ----------------------------- #
def main() -> None:
    """
    Main entry point.
    """
    args = parse_args()
    run_cfg = load_yaml_config(str(args.config))

    if args.base_channels is not None:
        run_cfg["model"]["base_channels"] = int(args.base_channels)

    if args.depth is not None:
        run_cfg["model"]["depth"] = int(args.depth)

    model_cfg = run_cfg.get("model", {})

    in_channels = int(model_cfg.get("in_channels", 4))
    base_channels = int(model_cfg.get("base_channels", 16))
    depth = int(model_cfg.get("depth", 3))

    input_shape = (
        in_channels,
        int(args.profile_height),
        int(args.profile_width),
    )

    if args.run_postprocessing:
        if args.prediction_mask_dir is None:
            raise ValueError(
                "--prediction-mask-dir must be provided when using --run-postprocessing."
            )

        if args.postprocessing_output_dir is None:
            raise ValueError(
                "--postprocessing-output-dir must be provided when using --run-postprocessing."
            )

    print("\n" + "=" * 80)
    print("Running RFI small model inference")
    print(f"Config: {args.config}")
    print(f"Weights: {args.weights}")
    print(f"Test dir: {args.test_dir}")
    print("Model: UNetSmall")
    print(f"Base channels: {base_channels}")
    print(f"Depth: {depth}")
    print(f"Threshold: {args.thr}")
    print(f"Stride multiple: {args.stride_multiple}")
    print(f"Evaluation: {'disabled' if args.skip_eval else 'enabled'}")
    print(f"Profiling: {'disabled' if args.skip_profiling else 'enabled'}")

    if args.run_postprocessing:
        print("Post-processing: enabled")
        print(f"Prediction mask dir: {args.prediction_mask_dir}")
        print(f"Post-processing output dir: {args.postprocessing_output_dir}")
        print(f"Post-processing threshold dB: {args.threshold_db}")
        print(f"Post-processing filter mode: {args.filter_mode}")
    else:
        print("Post-processing: disabled")

    print("=" * 80)

    if not args.skip_eval:
        evaluate_test(
            cfg=run_cfg,
            weights_path=args.weights,
            test_dir=args.test_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            stride_multiple=args.stride_multiple,
            ignore_index=255,
            resize_to=args.resize_to,
            thr=args.thr,
        )

    if not args.skip_profiling:
        run_cpu_gpu_profiling(
            cfg=run_cfg,
            weights_path=args.weights,
            input_shape=input_shape,
            fallback_cpu_power_watts=args.fallback_cpu_power_watts,
            fallback_gpu_power_watts=args.fallback_gpu_power_watts,
        )

    if args.run_postprocessing:
        generate_prediction_masks(
            cfg=run_cfg,
            weights_path=args.weights,
            test_dir=args.test_dir,
            prediction_mask_dir=args.prediction_mask_dir,
            stride_multiple=args.stride_multiple,
            ignore_index=255,
            resize_to=args.resize_to,
            thr=args.thr,
        )

        process_prediction_folder(
            sar_data_dir=args.test_dir,
            prediction_mask_dir=args.prediction_mask_dir,
            output_dir=args.postprocessing_output_dir,
            threshold_db=args.threshold_db,
            region_stat=args.region_stat,
            background_stat=args.background_stat,
            min_region_area=args.min_region_area,
            ring_inner_iters=args.ring_inner_iters,
            ring_outer_iters=args.ring_outer_iters,
            trim_fraction=args.trim_fraction,
            pred_suffix="_pred",
            filter_mode=args.filter_mode,
        )


if __name__ == "__main__":
    main()