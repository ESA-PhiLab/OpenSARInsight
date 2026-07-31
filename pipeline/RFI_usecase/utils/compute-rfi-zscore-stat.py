"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: compute_rfi_zscore_stats.py

Description:
    Compute per-dataset, per-channel mean/std for 4-channel RFI RC data and save them into a JSON file.

    Assumes:
  - You already have RFI4ChannelDataset (returning "rc" as HWC or CHW)
  - You have ensure_channel_first(rc) -> (C,H,W) with C=4

    Usage (example):
  python compute_rfi_zscore_stats.py \
    --data_dir /path/to/train_set \
    --output_json /path/to/output \
    --batch_size 4 \
    --num_workers 4

History:
    - 2025-12-04:
        First version is ready.
    - 2025-12-05:
        The script is finalized and used for normalization.    
    
---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-12-04

© Copyright INDRA DEIMOS, 2025. All rights reserved.
---------------------------------------------------------------------
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import List
from pipeline.RFI_usecase.utils.logging_setup import init_logging
import torch
from torch.utils.data import DataLoader
from pipeline.RFI_usecase.utils.rc_dataloader import RFI4ChannelDataset, ensure_channel_first


# ------------------------- Logging ------------------------- #
init_logging(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


# ------------------------- CLI --------------------------- #
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the statistics computation script.

    Returns:
        Parsed command-line arguments containing:
            - data_dir: input RFI split directory.
            - output_json: destination JSON path.
            - batch_size: number of samples loaded per DataLoader batch.
            - num_workers: number of DataLoader worker processes.
            - verbosity: logging verbosity level.
    """
    parser = argparse.ArgumentParser(description="Compute per-dataset z-score stats for RFI 4-channel RC data.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="RFI split directory containing vv/vh .npy files (e.g. .../rfi-v2/train)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to output JSON with mean/std stats.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for iteration.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="0=warn, 1=info, 2=debug."
    )
    return parser.parse_args()


# ------------------------- Main --------------------------- #
def main() -> None:
    """
    Compute per-channel mean and standard deviation for the RFI RC dataset.

    The script iterates over the dataset without transforms, augmentation, masks,
    padding, or normalization. For each sample, it converts the RC data to
    channel-first format, accumulates the per-channel sum and squared sum, and
    computes:

        mean = sum(x) / N
        variance = sum(x^2) / N - mean^2
        std = sqrt(variance)

    The final statistics are saved to a JSON file.
    """
    args = parse_args()

    # ---- Logging ----
    level = logging.INFO if args.verbosity == 1 else logging.DEBUG if args.verbosity >= 2 else logging.WARN
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    LOGGER.setLevel(level)

    data_dir = Path(args.data_dir)
    out_path = Path(args.output_json)

    LOGGER.info("Computing dataset stats for RFI data under: %s", data_dir)

    # ---- Dataset without transforms/masks ----
    ds = RFI4ChannelDataset(
        data_dir=data_dir,
        use_masks=False,
        transform=None,              # stats on raw RC, no norm/augs
        strict_missing_masks=False,  # irrelevant here
    )
    LOGGER.info("Dataset size: %d samples", len(ds))

    # ---- DataLoader: simple list collate (no padding) ----
    def simple_collate(batch: List[dict]) -> List[dict]:
        return batch

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
        collate_fn=simple_collate,
    )

    # ---- Initialise accumulators ----
    C = 4  # 4-channel RFI data
    sum_c = torch.zeros(C, dtype=torch.float64)
    sum_sq_c = torch.zeros(C, dtype=torch.float64)
    n_pixels = 0  # scalar: pixels per channel

    # ---- Iterate through the dataset and accumulate statistics ----
    for batch_idx, batch in enumerate(loader):
        for sample in batch:
            rc = sample["rc"]           # HWC or CHW
            rc_chw = ensure_channel_first(rc)  # (4,H,W), float32

            # Flatten: (4, H*W)
            rc_flat = rc_chw.to(torch.float64).view(C, -1)
            sum_c += rc_flat.sum(dim=1)
            sum_sq_c += (rc_flat ** 2).sum(dim=1)
            n_pixels += rc_flat.shape[1]

        if batch_idx % 50 == 0:
            LOGGER.info(
                "Processed batch %d, running n_pixels_per_channel=%d",
                batch_idx,
                n_pixels,
            )

    if n_pixels == 0:
        raise RuntimeError("No pixels processed; check that the dataset is not empty.")

    # ---- Compute mean/std ----
    mean = sum_c / n_pixels
    var = (sum_sq_c / n_pixels) - mean ** 2
    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var)

    mean_list = mean.tolist()
    std_list = std.tolist()

    LOGGER.info("Final per-channel mean: %s", mean_list)
    LOGGER.info("Final per-channel std : %s", std_list)

    # ---- Save to JSON ----
    stats = {
        "mean": mean_list,
        "std": std_list,
        "n_pixels_per_channel": int(n_pixels),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    LOGGER.info("Saved stats to %s", out_path)


if __name__ == "__main__":
    main()
