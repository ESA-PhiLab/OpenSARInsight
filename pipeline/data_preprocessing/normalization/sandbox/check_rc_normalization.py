import numpy as np, torch
from sar_normalization import NormalizeRCTransform
from logging_setup import init_logging
import logging

init_logging(level=logging.DEBUG)   # configure once
LOGGER = logging.getLogger(__name__)

a = np.load("", allow_pickle=False)          # complex (H,W) or float (H,W,2)
tf = NormalizeRCTransform(method="rms_shared")
y = tf({"rc": a})["rc"]                            # torch.float32 (H,W,2)
print("E[|y|^2] ≈", ((y**2).sum(dim=-1)).mean().item())  # expect ~1.0

# Compare logged s with direct computation on input (float32 cast like the class does):
if np.iscomplexobj(a):
    s_est = np.sqrt(np.mean((a.real.astype(np.float32)**2 + a.imag.astype(np.float32)**2)))
else:
    s_est = np.sqrt(np.mean(a[...,0].astype(np.float32)**2 + a[...,1].astype(np.float32)**2))
print("s_est ≈", float(s_est))                     # should be close to 3.758753e-02


# y is the normalized (H,W,2) tensor
# post = (y**2).sum(-1).mean(0)              # per-column E[|y|^2]
# float(post.min().item()), float(post.max().item())  # both ≈ 1.0
# float((post-1).abs().max().item())         # max deviation (target < 1e-3)



#########################Quick test showing the results of normalization for each pixel#############################
import numpy as np
import torch
from sar_normalization import NormalizeRCTransform
from logging_setup import init_logging
import logging
from pathlib import Path
from dataset_generation_scripts.utils import get_config
cfg = get_config("MODEL_USECASES_PATH")

init_logging(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

IN_PATH = cfg["check_rc_normalization"]["rc1_path"]
OUT_NPY = Path(cfg["check_rc_normalization"]["out_path"]).joinpath("rc_div_by_s.npy")          # full normalized array saved here (float32, H,W,2)
OUT_TXT = Path(cfg["check_rc_normalization"]["out_path"]).joinpath("rc_div_by_s_preview.txt")  # small human-readable preview

# --- load
a = np.load(IN_PATH, allow_pickle=False)   # complex (H,W) or float (H,W,2)

# --- compute global RMS scale s over |x|^2 in float32 (same as the class)
if np.iscomplexobj(a):
    I = a.real.astype(np.float32, copy=False)
    Q = a.imag.astype(np.float32, copy=False)
    s = float(np.sqrt(np.mean(I**2 + Q**2)))
    # manual normalization: divide every complex pixel by s, then to (H,W,2)
    y_manual_hw2 = np.stack([I / s, Q / s], axis=-1).astype(np.float32, copy=False)
else:
    I = a[..., 0].astype(np.float32, copy=False)
    Q = a[..., 1].astype(np.float32, copy=False)
    s = float(np.sqrt(np.mean(I**2 + Q**2)))
    # divide both channels by s (every pixel)
    y_manual_hw2 = np.stack([I / s, Q / s], axis=-1).astype(np.float32, copy=False)

print(f"s (global RMS) = {s:.8e}")

# --- verify against the class (optional)
tf = NormalizeRCTransform(method="rms_shared", log_stats=True)
y_class = tf({"rc": a})["rc"]  # torch.float32, (H,W,2)

# mean power after normalization (~1.0)
power_mean = ((y_class ** 2).sum(dim=-1)).mean().item()
print("E[|y|^2] ≈", power_mean)

# numeric closeness check (manual vs class)
y_manual_t = torch.from_numpy(y_manual_hw2)
rel_err = torch.linalg.norm(y_class - y_manual_t) / (torch.linalg.norm(y_class) + 1e-12)
print("relative error (manual vs class) =", float(rel_err))

# --- SAVE full per-pixel result so you can "see every pixel ÷ s"
np.save(OUT_NPY, y_manual_hw2)  # (H,W,2) float32
print(f"Saved full normalized array to {OUT_NPY.resolve()} with shape {y_manual_hw2.shape}")

# --- Optional: write a small preview to text (don’t dump everything to console)
#   Shows a 5x5 top-left block of I' and Q' after division by s
h_show, w_show = 5, 5
with OUT_TXT.open("w") as f:
    f.write(f"s = {s:.8e}\n")
    f.write("Top-left 5x5 of I' (after divide by s):\n")
    np.savetxt(f, y_manual_hw2[:h_show, :w_show, 0], fmt="%.6e")
    f.write("\nTop-left 5x5 of Q' (after divide by s):\n")
    np.savetxt(f, y_manual_hw2[:h_show, :w_show, 1], fmt="%.6e")
print(f"Wrote preview to {OUT_TXT.resolve()}")
