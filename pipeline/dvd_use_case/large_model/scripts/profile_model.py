# type: ignore

"""
---------------------------------------------------------------------
Project: OpenSAR Insight
---------------------------------------------------------------------
profile_model.py

Tool:Script to calculate model complexity

Author: Abdulhameed Yunusa (ABHY)
E-mail: ayunusa@indracompany.com
Creation Date: 2026-04-08

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""

import os
import time
import threading
import torch
import psutil
import pynvml
from thop import profile as thop_profile


# ─────────────────────────────────────────────────────────────
# Units map
# ─────────────────────────────────────────────────────────────
UNITS = {
    "profile/GMACs":            "GMACs",
    "profile/GFLOPs":           "GFLOPs",
    "profile/params_M":         "M params",
    "profile/latency_ms":       "ms",
    "profile/fps":              "FPS",
    "profile/gpu_mem_alloc_GB": "GB",
    "profile/gpu_mem_peak_GB":  "GB",
    "profile/gpu_util_avg_%":   "%",
    "profile/gpu_util_peak_%":  "%",
    "profile/gpu_power_avg_W":  "W",
    "profile/gpu_power_peak_W": "W",
    "profile/cpu_util_avg_%":   "%",
    "profile/cpu_util_peak_%":  "%",
    "profile/ram_GB":           "GB",
    "profile/cpu_power_avg_W":  "W",
    "profile/cpu_power_peak_W": "W",
}

SECTIONS = {
    "Complexity":  ["profile/GMACs",            "profile/GFLOPs",
                    "profile/params_M"],
    "Latency":     ["profile/latency_ms",        "profile/fps"],
    "GPU Memory":  ["profile/gpu_mem_alloc_GB",  "profile/gpu_mem_peak_GB"],
    "GPU Compute": ["profile/gpu_util_avg_%",    "profile/gpu_util_peak_%"],
    "GPU Power":   ["profile/gpu_power_avg_W",   "profile/gpu_power_peak_W"],
    "CPU":         ["profile/cpu_util_avg_%",    "profile/cpu_util_peak_%",
                    "profile/ram_GB",
                    "profile/cpu_power_avg_W",   "profile/cpu_power_peak_W"],
}


# ─────────────────────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────────────────────
def pretty_print_profile(stats: dict):
    W = 60
    print("\n" + "═" * W)
    print("  Model Profile")
    print("═" * W)

    for section, keys in SECTIONS.items():
        print(f"\n  > {section}")
        print("  " + "─" * (W - 2))
        for k in keys:
            if k not in stats:
                continue
            v    = stats[k]
            unit = UNITS.get(k, "")
            name = k.replace("profile/", "")

            if v is None:
                val_str = "N/A  (no RAPL)"
            else:
                val_str = f"{v:>10.3f}  {unit}"

            print(f"  {name:<28} {val_str}")

    print("\n" + "═" * W + "\n")


# ─────────────────────────────────────────────────────────────
# Pretty dict wrapper
# ─────────────────────────────────────────────────────────────
class ProfileStats(dict):
    """dict subclass whose print/repr renders the formatted table."""
    def __repr__(self):
        pretty_print_profile(self)
        return ""          # suppress the raw dict line

    def __str__(self):
        pretty_print_profile(self)
        return ""


# ─────────────────────────────────────────────────────────────
# Background sampler
# ─────────────────────────────────────────────────────────────
class HardwareSampler:
    def __init__(self, device_idx: int = 0, interval: float = 0.05):
        self.device_idx = device_idx
        self.interval   = interval
        self._stop      = threading.Event()

        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

        self.gpu_util_samples  = []
        self.gpu_power_samples = []
        self._proc             = psutil.Process(os.getpid())
        self.cpu_util_samples  = []
        self.cpu_power_samples = []

        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._stop.clear()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _read_rapl_uj(self):
        candidates = [
            "/sys/class/powercap/intel-rapl:0/energy_uj",
            "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
            "/sys/class/powercap/amd_energy/amd_energy:0/energy_uj",  # AMD fallback
        ]
        for path in candidates:
            try:
                with open(path) as f:
                    return int(f.read())
            except (FileNotFoundError, PermissionError):
                continue
        return None

    def _run(self):
        rapl_t0 = self._read_rapl_uj()
        time_t0 = time.perf_counter()

        while not self._stop.is_set():
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            self.gpu_util_samples.append(util.gpu)

            try:
                pw = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                self.gpu_power_samples.append(pw)
            except pynvml.NVMLError:
                pass

            self.cpu_util_samples.append(self._proc.cpu_percent())

            rapl_now = self._read_rapl_uj()
            time_now = time.perf_counter()
            if rapl_t0 is not None and rapl_now is not None:
                elapsed = time_now - time_t0
                if elapsed > 0:
                    self.cpu_power_samples.append(
                        (rapl_now - rapl_t0) / 1e6 / elapsed
                    )

            time.sleep(self.interval)

    @staticmethod
    def _safe(samples, fn):
        return fn(samples) if samples else None


# ─────────────────────────────────────────────────────────────
# Main profiler
# ─────────────────────────────────────────────────────────────
def profile_model(model, imgsz=512, device=0, runs=200) -> ProfileStats:
    dev   = torch.device(f"cuda:{device}")
    dummy = torch.zeros(1, 3, imgsz, imgsz).to(dev)

    # FLOPs / MACs / Params
    macs, params = thop_profile(model.model, inputs=(dummy,), verbose=False)

    # Warm-up
    for _ in range(50):
        model.model(dummy)
    torch.cuda.synchronize()

    # Reset peak memory
    torch.cuda.reset_peak_memory_stats(device)

    # Start sampler
    sampler = HardwareSampler(device_idx=device, interval=0.05)
    sampler.start()

    # Timed loop
    t0 = time.perf_counter()
    for _ in range(runs):
        model.model(dummy)
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) / runs * 1000

    sampler.stop()

    # Memory
    gpu_mem_alloc_gb = torch.cuda.memory_allocated(device)     / 1e9
    gpu_mem_peak_gb  = torch.cuda.max_memory_allocated(device) / 1e9

    # CPU snapshot
    proc   = psutil.Process(os.getpid())
    ram_gb = proc.memory_info().rss / 1e9

    s = sampler
    stats = ProfileStats({
        "profile/GMACs":            macs / 1e9,
        "profile/GFLOPs":           2 * macs / 1e9,
        "profile/params_M":         params / 1e6,
        "profile/latency_ms":       latency_ms,
        "profile/fps":              1000 / latency_ms,
        "profile/gpu_mem_alloc_GB": gpu_mem_alloc_gb,
        "profile/gpu_mem_peak_GB":  gpu_mem_peak_gb,
        "profile/gpu_util_avg_%":   s._safe(s.gpu_util_samples,  lambda x: sum(x)/len(x)),
        "profile/gpu_util_peak_%":  s._safe(s.gpu_util_samples,  max),
        "profile/gpu_power_avg_W":  s._safe(s.gpu_power_samples, lambda x: sum(x)/len(x)),
        "profile/gpu_power_peak_W": s._safe(s.gpu_power_samples, max),
        "profile/cpu_util_avg_%":   s._safe(s.cpu_util_samples,  lambda x: sum(x)/len(x)),
        "profile/cpu_util_peak_%":  s._safe(s.cpu_util_samples,  max),
        "profile/ram_GB":           ram_gb,
        "profile/cpu_power_avg_W":  s._safe(s.cpu_power_samples, lambda x: sum(x)/len(x)),
        "profile/cpu_power_peak_W": s._safe(s.cpu_power_samples, max),
    })

    # print on creation
    pretty_print_profile(stats)

    # Strip None before returning
    return ProfileStats({k: v for k, v in stats.items() if v is not None})