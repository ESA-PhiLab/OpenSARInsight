"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: normalization.py

Description:
    Applies precomputed per-dataset, per-channel z-score normalization to
    4-channel range-compressed SAR data: VV_I, VV_Q, VH_I, VH_Q.
    
History:
    - 2025-12-03:
        First version is ready.
    - 2025-12-04:
        The script is finalized and used to normalize the RC data.    
    
---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-12-03

© Copyright INDRA DEIMOS, 2025. All rights reserved.
---------------------------------------------------------------------
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Mapping, Sequence, Union
import numpy as np
from pipeline.RFI_usecase.utils.logging_setup import init_logging
import torch
from torch import Tensor



# ------------------------- Logging ------------------------- #
init_logging(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


# ------------------------- Main class ------------------------- #
class NormalizeZScoreRC4PerDataset:
    """
    Apply per-dataset, per-channel z-score normalization to 4-channel
    range-compressed SAR data.

    The transform expects the input sample to contain a 4-channel RC tensor under
    sample[key]. The channel order is assumed to be:

        VV_I, VV_Q, VH_I, VH_Q

    Supported input layouts:
        - HWC: (H, W, 4)
        - CHW: (4, H, W)

    Supported input types:
        - numpy.ndarray
        - torch.Tensor

    The normalization is applied using fixed, precomputed dataset-level
    statistics:

        x_norm_c = (x_c - mean_c) / max(std_c, eps)

    The original layout is preserved. If the input is CHW, the output is CHW.
    If the input is HWC, the output is HWC.

    Args:
        mean: Sequence, NumPy array, or torch.Tensor containing four channel means.
        std: Sequence, NumPy array, or torch.Tensor containing four channel standard
            deviations.
        key (str): Dictionary key used to read and write the RC data in the sample.
        eps (float): Minimum standard deviation value used for numerical stability.
        log_stats (bool): If True, log the normalization statistics when the transform is applied.
    """

    def __init__(
        self,
        mean: Union[Sequence[float], np.ndarray, torch.Tensor],
        std: Union[Sequence[float], np.ndarray, torch.Tensor],
        key: str = "rc",
        eps: float = 1e-6,
        log_stats: bool = False,
    ) -> None:
        if isinstance(mean, torch.Tensor):
            mean_t = mean.to(dtype=torch.float32)
        else:
            mean_t = torch.tensor(mean, dtype=torch.float32)

        if isinstance(std, torch.Tensor):
            std_t = std.to(dtype=torch.float32)
        else:
            std_t = torch.tensor(std, dtype=torch.float32)

        if mean_t.numel() != 4 or std_t.numel() != 4:
            raise ValueError(
                f"Expected mean/std with 4 elements (for 4 channels); "
                f"got mean={mean_t.numel()}, std={std_t.numel()}"
            )

        self.mean = mean_t.view(1, 1, 4)  # for broadcasting over (H,W,4)
        self.std = std_t.view(1, 1, 4)
        self.key = key
        self.eps = float(eps)
        self.log_stats = bool(log_stats)

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        x = sample[self.key]

        # ---- Convert to torch.float32 tensor ----
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x)
        elif isinstance(x, torch.Tensor):
            x_t = x
        else:
            raise TypeError(f"{self.key} must be np.ndarray or torch.Tensor, got {type(x)}")

        x_t = x_t.to(torch.float32)

        # ---- Detect layout and convert to HWC for normalization ----
        if x_t.ndim != 3:
            raise ValueError(f"{self.key} must be 3D (H,W,4) or (4,H,W); got shape {tuple(x_t.shape)}")

        if x_t.shape[-1] == 4:
            # (H,W,4) already
            layout = "HWC"
            x_hwc = x_t
        elif x_t.shape[0] == 4:
            # (4,H,W) -> (H,W,4)
            layout = "CHW"
            x_hwc = x_t.permute(1, 2, 0).contiguous()
        else:
            raise ValueError(
                f"{self.key} has invalid shape {tuple(x_t.shape)}; "
                "expected (H,W,4) or (4,H,W)."
            )

        # ---- Per-dataset z-score: (x - mean) / std ----
        mean = self.mean.to(x_hwc.device)
        std = self.std.to(x_hwc.device).clamp_min(self.eps)

        x_norm = (x_hwc - mean) / std  # (H,W,4)

        # ---- Convert back to original layout ----
        if layout == "CHW":
            x_norm = x_norm.permute(2, 0, 1).contiguous()  # -> (4,H,W)

        out = dict(sample) 
        out[self.key] = x_norm

        if self.log_stats:
            LOGGER.info(
                "NormalizeZScoreRC4PerDataset | mean=%s std=%s device=%s",
                self.mean.view(-1).tolist(),
                self.std.view(-1).tolist(),
                x_norm.device,
            )

        return out