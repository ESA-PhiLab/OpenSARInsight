"""
rc_make_rms_stats.py

Compute a train-only RMS-shared scalar for Range-Compressed (RC) SAR data and
save it as a JSON artifact suitable for `norm.profiles.per_dataset.stats_path`.

RMS definition (shared across I & Q):
    RMS = sqrt( sum(I^2 + Q^2) / N_valid_pixels )

Assumptions
-----------
- Your dataset yields samples with at least `sample["rc"]`.
- If available, `sample["valid_mask"]` marks valid pixels (True/1 = valid).
- Build the dataset WITHOUT stochastic amplitude-changing augmentations.
- Geometry-only deterministic steps are fine, but simplest is `transform=None`.

DDP
---
- If torch.distributed is initialized, the script will all-reduce S and N so all
  ranks compute the same RMS. Only rank 0 writes the JSON.

Usage
-----
python rc_make_rms_stats.py \
  --train-dir /path/to/train \
  --out /path/to/rc_norm_stats.json \
  --batch-size 8 --num-workers 4

Optional:
  --dataset-id opensar_flood_train_2025-10-10
  --max-batches 200             # sample a subset for speed
  --no-valid-mask               # ignore valid_mask even if present
  --device cuda:0               # or cpu (default auto)
"""
from __future__ import annotations
from typing import Optional, Dict, Any
import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# ----- import your project bits (adapt paths/names if needed) -----
try:
    from pipeline.data_preprocessing.dataloader.rc_dataloader import RCDataset, collate_pad_validmask, default_mask_resolver
except ImportError as exc:
    raise SystemExit(
        "ERROR: Could not import rc_dataloader components. "
        "Run from your project root or fix PYTHONPATH."
    ) from exc


LOG = logging.getLogger("rc_make_rms_stats")


# ----------------------------- DDP utils ----------------------------- #
def is_ddp() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def ddp_allreduce_sum(x: torch.Tensor) -> torch.Tensor:
    if is_ddp():
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    return x


# ------------------------- Tensor shape helpers ---------------------- #
def rc_to_bhw2(t: torch.Tensor) -> torch.Tensor:
    """
    Ensure batched RC tensor is (B, H, W, 2) float32.

    Accepts:
      - (B, 2, H, W)  -> transpose
      - (B, H, W, 2)  -> as-is
    """
    if not isinstance(t, torch.Tensor) or t.ndim != 4:
        raise ValueError(f"Expected 4D tensor for 'rc'; got {type(t)} with shape={getattr(t, 'shape', None)}")
    if t.shape[1] == 2:  # (B, 2, H, W)
        return t.permute(0, 2, 3, 1).contiguous().float()
    if t.shape[-1] == 2:  # (B, H, W, 2)
        return t.float()
    raise ValueError(f"Unsupported RC layout: shape={tuple(t.shape)} (expected (B,2,H,W) or (B,H,W,2))")


def validmask_to_bhw(vm: torch.Tensor) -> torch.Tensor:
    """
    Normalize valid_mask to (B, H, W) of bool.
    Accepts:
      - (B, 1, H, W)
      - (B, H, W)
    """
    if vm.ndim == 4 and vm.shape[1] == 1:
        vm = vm[:, 0]
    if vm.ndim != 3:
        raise ValueError(f"Unexpected valid_mask shape: {tuple(vm.shape)} (expected (B,1,H,W) or (B,H,W))")
    return vm.to(dtype=torch.bool)


# --------------------------- Core compute ---------------------------- #
@torch.no_grad()
def compute_rms_shared_for_loader(
    loader: DataLoader,
    device: torch.device,
    respect_valid_mask: bool = True,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Stream a loader and accumulate S = sum(I^2 + Q^2), N = count(valid pixels).
    Returns dict with float values: {"S": ..., "N": ..., "samples": ...}
    """
    S_local = torch.zeros((), dtype=torch.float64, device=device)
    N_local = torch.zeros((), dtype=torch.float64, device=device)
    samples = 0

    for b_idx, batch in enumerate(loader):
        # Extract fields regardless of structure (dict or namespace)
        rc = batch["rc"] if isinstance(batch, dict) else getattr(batch, "rc", None)
        vm = None
        if respect_valid_mask:
            vm = batch.get("valid_mask") if isinstance(batch, dict) else getattr(batch, "valid_mask", None)

        if rc is None:
            raise KeyError("Batch missing 'rc' tensor.")

        rc = rc_to_bhw2(rc).to(device, non_blocking=True)  # (B, H, W, 2)
        B, H, W, _ = rc.shape
        if H == 0 or W == 0:
            # Skip degenerate tiles
            continue

        # Power over channels -> (B, H, W)
        power = (rc ** 2).sum(dim=-1)

        if respect_valid_mask and vm is not None:
            vm_t = validmask_to_bhw(vm).to(device=device, non_blocking=True)
            S_local += power.masked_select(vm_t).to(dtype=torch.float64).sum()
            N_local += vm_t.to(dtype=torch.float64).sum()
        else:
            S_local += power.to(dtype=torch.float64).sum()
            N_local += torch.tensor(B * H * W, dtype=torch.float64, device=device)

        samples += B
        if max_batches is not None and (b_idx + 1) >= max_batches:
            break

    # DDP: sum across ranks
    S_total = ddp_allreduce_sum(S_local)
    N_total = ddp_allreduce_sum(N_local)

    return {"S": float(S_total.item()), "N": float(N_total.item()), "samples": float(samples)}


def make_loader_for_calibration(
    train_dir: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: Optional[int],
    strict_missing_masks: bool,
) -> DataLoader:
    """
    Build a calibration dataset WITHOUT stochastic augs and WITHOUT normalization.
    """
    ds = RCDataset(
        data_dir=train_dir,
        use_masks=True,
        mask_resolver=default_mask_resolver,
        transform=None,  # IMPORTANT: no augs, no normalization
        strict_missing_masks=strict_missing_masks,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,
        collate_fn=lambda b: collate_pad_validmask(b, enforce_stride_multiple=False, stride_multiple=None),
    )
    return loader


# ------------------------------ CLI/main ----------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute train-only RMS-shared for RC data and write JSON stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-dir", required=True, help="Path to TRAIN split root directory.")
    p.add_argument("--out", required=True, help="Output JSON path for stats (e.g., rc_norm_stats.json).")
    p.add_argument("--dataset-id", default=None, help="Identifier saved in JSON (name/hash/date).")
    p.add_argument("--batch-size", type=int, default=8, help="Calibration batch size.")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    p.add_argument("--pin-memory", action="store_true", help="Enable pin_memory in DataLoader.")
    p.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch_factor (workers>0).")
    p.add_argument("--max-batches", type=int, default=None, help="Limit number of batches for faster approximate RMS.")
    p.add_argument("--no-valid-mask", action="store_true", help="Ignore valid_mask even if dataset provides it.")
    p.add_argument("--device", default=None, help="Device to use: e.g., 'cuda:0' or 'cpu'. Default: auto.")
    p.add_argument("--eps", type=float, default=1e-12, help="Epsilon to store alongside RMS.")
    p.add_argument("--log-level", default="INFO", help="Logging level.")
    p.add_argument("--strict-missing-masks", action="store_true", help="Error if masks are missing.")
    return p.parse_args()


def setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    # Resolve device
    if args.device is None:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
    else:
        device = torch.device(args.device)

    # If DDP, ensure process group is already initialized by your launcher
    if is_ddp():
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
        LOG.info("Running in DDP mode | rank=%d world=%d device=%s", rank, world, device)
    else:
        LOG.info("Running single-process | device=%s", device)

    # Build calibration loader (no augs, no norm)
    loader = make_loader_for_calibration(
        train_dir=args.train_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        strict_missing_masks=args.strict_missing_masks,
    )

    # Compute S and N
    stats = compute_rms_shared_for_loader(
        loader=loader,
        device=device,
        respect_valid_mask=not args.no_valid_mask,
        max_batches=args.max_batches,
    )
    S = stats["S"]
    N = stats["N"]
    samples = int(stats["samples"])

    if N <= 0.0:
        raise SystemExit("ERROR: No valid pixels counted; check dataset and valid_mask.")
    rms = math.sqrt(S / N)

    # Write JSON on rank 0 (or single process)
    is_writer = True
    if is_ddp():
        is_writer = torch.distributed.get_rank() == 0

    if is_writer:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        js: Dict[str, Any] = {
            "rms": rms,
            "mode": "rms_shared",
            "scope": "train_only",
            "channels": "shared",
            "eps": float(args.eps),
            "dataset_id": args.dataset_id or f"{Path(args.train_dir).name}_{time.strftime('%Y-%m-%d')}",
            "samples": samples,
            "pixels": int(N),
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": {
                "train_dir": os.path.abspath(args.train_dir),
                "max_batches": args.max_batches,
                "respect_valid_mask": not args.no_valid_mask,
                "strict_missing_masks": bool(args.strict_missing_masks),
            },
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(js, f, ensure_ascii=False, indent=2)

        LOG.info("Wrote RMS stats JSON to %s | rms=%.8e | samples=%d | pixels=%d", str(out_path), rms, samples, int(N))
    else:
        LOG.info("Computed RMS=%.8e (writer is rank 0; this rank will not write JSON).", rms)


if __name__ == "__main__":
    main()
