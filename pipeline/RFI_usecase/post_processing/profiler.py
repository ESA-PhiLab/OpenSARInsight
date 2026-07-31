"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: profiler.py

Key properties
--------------
- Provides utilities for profiling PyTorch model inference.
- Measures parameter count, compute cost, peak GPU memory, latency,
  and throughput.
- Supports optional MACs/FLOPs estimation with `thop` or `ptflops`.

Recommended usage
-----------------
- Use this module to compare model efficiency under consistent
  inference settings.
- Useful for benchmarking different segmentation model variants.

History:
    - 2026-02-16:
        First version of profiling utility.
    - 2026-03-20:
        Final version is ready.

---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2026-02-16

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""
from __future__ import annotations
from contextlib import contextmanager
import time
from typing import Optional, Tuple
import torch
from torch import nn

# Optional MACs/FLOPs libs
try:
    from thop import profile as thop_profile  # pip install thop
    _HAS_THOP = True
except Exception:
    _HAS_THOP = False

try:
    from ptflops import get_model_complexity_info  # pip install ptflops
    _HAS_PTFLOPS = True
except Exception:
    _HAS_PTFLOPS = False



def count_params_m(model: nn.Module) -> float:
    """
    Return the number of trainable parameters in `model` (in millions).
    Args:
        model: A torch.nn.Module instance.
    Returns:
        float: Trainable parameter count divided by 1e6.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


@contextmanager
def _cuda_peak_meter(device: torch.device):
    """
    Context manager to prepare and finalize CUDA peak memory measurement.

    For a CUDA device this resets peak memory stats and synchronizes before
    entering the context, and ensures synchronization on exit so that
    torch.cuda.max_memory_allocated() reports the correct peak.

    For non-CUDA devices this is a no-op.
    """
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = torch.cuda.max_memory_allocated(device)
        try:
            yield
        finally:
            torch.cuda.synchronize(device)
    else:
        yield


@torch.no_grad()
def profile_inference(
    model: nn.Module,
    input_shape: Tuple[int, int, int] = (2, 512, 512),
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
    warmup: int = 10,
    iters: int = 50,
    use_channels_last: bool = True,
) -> dict:
    """
    Profile a model's inference performance and resource usage.

    Measures trainable parameters (in millions), optional MACs/FLOPs (if
    thop or ptflops are available), peak GPU memory (GB), average
    latency per inference (ms), and throughput (frames per second).

    Returns a dict with keys: params_m, macs_g, flops_g, peak_vram_gb,
    latency_ms, fps.
    """
    model.eval().to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    x = torch.randn(1, *input_shape, device=device)
    if use_channels_last:
        x = x.to(memory_format=torch.channels_last)

    # ---------- MACs / FLOPs ----------
    macs = flops = None
    # Prefer THOP (reports MACs); FLOPs commonly = 2 * MACs for convs
    if _HAS_THOP:
        m = model
        # thop needs fp32 for some ops; run on cpu to be safe
        m_cpu = model.to("cpu").eval()
        x_cpu = x.detach().to("cpu", dtype=torch.float32)
        macs_thop, params = thop_profile(m_cpu, inputs=(x_cpu,), verbose=False)
        macs = macs_thop / 1e9  # GMac
        flops = 2.0 * macs      # heuristic for conv layers
        model.to(device)
    elif _HAS_PTFLOPS:
        # ptflops expects (C,H,W) string of size; run on CPU graph
        ch, h, w = input_shape
        flops_str, params_str = get_model_complexity_info(
            model.to("cpu"), (ch, h, w), as_strings=True, print_per_layer_stat=False, verbose=False
        )
        model.to(device)
        # parse numbers like '45.23 GMac'
        def _parse(s): return float(s.split()[0])
        macs = _parse(flops_str.replace('GMac', ''))  # ptflops usually reports MACs
        flops = 2.0 * macs
    else:
        macs = None
        flops = None

    # ---------- VRAM & latency ----------
    # AMP + inference_mode to mimic real inference memory/throughput
    amp_dtype = torch.float16 if dtype == torch.float16 else torch.bfloat16

    # warmup
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type=="cuda")):
        for _ in range(warmup):
            _ = model(x)

    # measure latency + peak memory
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    t0 = time.perf_counter()
    with _cuda_peak_meter(device):
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type=="cuda")):
            for _ in range(iters):
                _ = model(x)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
    t1 = time.perf_counter()

    # peak VRAM
    if device.type == "cuda":
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_gb = peak_bytes / (1024**3)
    else:
        peak_gb = 0.0

    dt = (t1 - t0) / iters
    latency_ms = dt * 1000.0
    fps = 1.0 / dt if dt > 0 else float('inf')

    return {
        "params_m": count_params_m(model),
        "macs_g": macs,
        "flops_g": flops,
        "peak_vram_gb": peak_gb,
        "latency_ms": latency_ms,
        "fps": fps,
    }