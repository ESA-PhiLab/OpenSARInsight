"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: sar_normalizatin.py

Description:
   This script is a pyTorch-compatible transforms that normalizes Synthetic-Aperture Radar (SAR) inputs__Range_Compressed data and SLC data__
   for the focusing head and the detection heads for ship, RFI and flood-detection use cases.

History:
    - 2025-08-07:
        This is the first version of the normalization code.
    - 2025-08-18:
        Added the required comments.
        Updated the error handling.
    - 2025-09-24:
        Replaced the L0 normalization with range compressed normalization

---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-08-04

© Copyright INDRA DEIMOS, 2025. All rights reserved.
---------------------------------------------------------------------
"""
from __future__ import annotations
from typing import Dict, Mapping, Any
import logging
import torch
from torch import Tensor
from logging_setup import init_logging

init_logging(level=logging.DEBUG)   # configure once
LOGGER = logging.getLogger(__name__)



__all__ = [
    "NormalizeRCTransform",
    "NormalizeSLCAmpTransform",
]

EPSILON: float = 1e-12  # numeric stability for division / ranges


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _to_float_tensor(arr: Any) -> Tensor:
    """
        Convert input to torch.float32 tensor.

        Args:
            arr (Any): Input array (torch.Tensor or numpy.ndarray).

        Returns:
            Tensor: Converted float32 tensor.

        Raises:
            TypeError: If input is not torch.Tensor or numpy.ndarray.
    """
    if isinstance(arr, Tensor):
        return arr.float()

    try:
        import numpy as np
    except ModuleNotFoundError as exc:  
        raise TypeError("Input must be torch.Tensor or numpy.ndarray") from exc

    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr).float()
    raise TypeError("Input must be torch.Tensor or numpy.ndarray")

# ---------------------------------------------------------------------
# Normalization for Range Compressed SAR data
# ---------------------------------------------------------------------
from typing import Mapping, Any, Optional, Literal, Dict
import math
import logging
import torch
from torch import Tensor

LOGGER = logging.getLogger(__name__)

class NormalizeRCTransform:
    """
    Normalize Range-Compressed (RC) complex SAR stored as (H, W, 2) [I, Q] or as complex (H, W).

    Methods:
        - "rms_shared" (default): one RMS scale from |I + jQ| over H×W, applied to both I and Q.
        - "rms_shared_per_column": RMS computed per column (per range bin), applied to both I and Q
          when mode="per_patch". For mode="per_dataset", we **still use a shared global scalar**.

    Modes:
        - mode="per_patch": compute RMS from the current sample (existing behavior).
        - mode="per_dataset": use a fixed RMS scalar provided via `rms_value` (train-only statistic).

    Args:
        method ({"rms_shared", "rms_shared_per_column"}): Normalization method.
        eps (float): Small constant to avoid division by zero.
        log_stats (bool): If True, logs normalization statistics.
        mode ({"per_patch", "per_dataset"}): Per-sample vs dataset-level normalization.
        rms_value (Optional[float]): Required when mode="per_dataset". Ignored for mode="per_patch".

    Notes:
        - Expects sample["rc"]; returns a new dict with "rc" as float32 (H, W, 2).
        - Masks (if present elsewhere) must not be normalized; mirror only spatial ops with NN.
    """

    def __init__(
        self,
        method: Literal["rms_shared", "rms_shared_per_column"] = "rms_shared",
        eps: float = 1e-12,
        log_stats: bool = False,
        mode: Literal["per_patch", "per_dataset"] = "per_patch",
        rms_value: Optional[float] = None,
    ) -> None:
        if method not in ("rms_shared", "rms_shared_per_column"):
            raise ValueError('method must be "rms_shared" or "rms_shared_per_column"')
        if mode not in ("per_patch", "per_dataset"):
            raise ValueError('mode must be "per_patch" or "per_dataset"')

        self.method = method
        self.eps = float(eps)
        self.log_stats = bool(log_stats)
        self.mode = mode

        # Validate dataset-level RMS when requested
        if self.mode == "per_dataset":
            if rms_value is None:
                raise ValueError("mode='per_dataset' requires a valid rms_value (float).")
            if not isinstance(rms_value, (int, float)) or not math.isfinite(rms_value) or rms_value <= 0.0:
                raise ValueError(f"Invalid rms_value for per_dataset mode: {rms_value!r}")
            self.rms_value: float = float(rms_value)
        else:
            self.rms_value = float("nan")  # unused in per_patch

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        if "rc" not in sample:
            raise KeyError('sample must contain key "rc" for range-compressed data')
        rc = self._ensure_hw2(self._to_float_tensor_rc(sample["rc"]))  # (H, W, 2) float32

        if self.method == "rms_shared":
            power = (rc ** 2).sum(dim=-1, keepdim=True)  # (H, W, 1)

            if self.mode == "per_patch":
                rms = power.mean(dim=(0, 1), keepdim=True).sqrt().clamp_min(self.eps)  # (1,1,1)
                rc_norm = rc / rms
                if self.log_stats:
                    LOGGER.info("NormalizeRCTransform:rms_shared | per_patch rms=%.6e", rms.flatten()[0].item())
            else:  # per_dataset
                rms = torch.tensor(self.rms_value, dtype=rc.dtype, device=rc.device).view(1, 1, 1).clamp_min(self.eps)
                rc_norm = rc / rms
                if self.log_stats:
                    LOGGER.info("NormalizeRCTransform:rms_shared | per_dataset rms=%.6e", self.rms_value)

        else:  # "rms_shared_per_column"
            power = (rc ** 2).sum(dim=-1, keepdim=True)  # (H, W, 1)

            if self.mode == "per_patch":
                rms = power.mean(dim=0, keepdim=True).sqrt().clamp_min(self.eps)  # (1, W, 1)
                rc_norm = rc / rms
                if self.log_stats:
                    LOGGER.info(
                        "NormalizeRCTransform:rms_shared_per_column | per_patch rms[min=%.6e, max=%.6e]",
                        rms.min().item(),
                        rms.max().item(),
                    )
            else:
                # For dataset mode, we keep semantics as a single shared scalar.
                rms = torch.tensor(self.rms_value, dtype=rc.dtype, device=rc.device).view(1, 1, 1).clamp_min(self.eps)
                rc_norm = rc / rms
                if self.log_stats:
                    LOGGER.info(
                        "NormalizeRCTransform:rms_shared_per_column | per_dataset uses shared scalar rms=%.6e",
                        self.rms_value,
                    )

        out = dict(sample)
        out["rc"] = rc_norm
        return out

    # ---------- helpers ----------
    @staticmethod
    def _safe_torch_from_numpy(a: object) -> torch.Tensor:
        try:
            import numpy as np
            if not isinstance(a, np.ndarray):
                raise TypeError("Input is not a numpy.ndarray")
            return torch.from_numpy(a)
        except Exception:
            return torch.tensor(a, dtype=torch.float32, copy=True)

    @staticmethod
    def _to_float_tensor_rc(x: object) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            if x.is_complex():
                x = torch.view_as_real(x)
                return x.float()
            if x.ndim == 3 and x.shape[-1] == 2:
                return x.float()
            if x.ndim == 3 and x.shape[0] == 2:
                return x.permute(1, 2, 0).contiguous().float()
            raise ValueError(
                f"Unsupported torch tensor shape/dtype for RC; expected (H,W,2) float "
                f"or (H,W) complex; got dtype={x.dtype}, shape={tuple(x.shape)}"
            )
        try:
            import numpy as np
            xa = np.asarray(x)
            if np.iscomplexobj(xa) and xa.ndim == 2:
                iq = np.stack([xa.real, xa.imag], axis=-1).astype(np.float32, copy=False)
                return NormalizeRCTransform._safe_torch_from_numpy(iq)
            if xa.ndim == 3 and xa.shape[-1] == 2 and np.issubdtype(xa.dtype, np.floating):
                return NormalizeRCTransform._safe_torch_from_numpy(xa.astype(np.float32, copy=False))
            if xa.ndim == 3 and xa.shape[0] == 2 and np.issubdtype(xa.dtype, np.floating):
                iq = xa.transpose(1, 2, 0).astype(np.float32, copy=False)
                return NormalizeRCTransform._safe_torch_from_numpy(iq)
            raise ValueError(
                f"Unsupported numpy array shape/dtype for RC; expected (H,W,2) float "
                f"or (H,W) complex; got dtype={xa.dtype}, shape={tuple(xa.shape)}"
            )
        except ModuleNotFoundError:
            raise TypeError("NumPy is required for numpy.ndarray inputs")

    @staticmethod
    def _ensure_hw2(x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Internal error: expected torch.Tensor")
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError(f"Expected (H, W, 2), got {tuple(x.shape)}")
        return x.float()

# ---------------------------------------------------------------------
# Normalization for SLC SAR data
# ---------------------------------------------------------------------
class NormalizeSLCAmpTransform:
    """
    A class to apply log compression, percentile clipping, and scaling to SLC amplitude.

    Attributes:
        lower_pct (float): Lower percentile for clipping.
        upper_pct (float): Upper percentile for clipping.
    """

    def __init__(self, *, lower_pct: float = 1.0, upper_pct: float = 99.0) -> None:
        """
        Initialise the NormalizeSLCAmpTransform class.

        Args:
            lower_pct (float): Lower percentile for clipping.
            upper_pct (float): Upper percentile for clipping.

        Raises:
            ValueError: If percentiles are out of range or in invalid order.
        """
        if not 0.0 <= lower_pct < upper_pct <= 100.0:
            raise ValueError("Percentiles must satisfy 0 ≤ lower < upper ≤ 100")
        self.lower_pct = float(lower_pct)
        self.upper_pct = float(upper_pct)

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Apply normalization on the slc data using log compression and percentile-based clipping.

        Args:
            sample (Mapping[str, Any]): Input sample with key slc of shape (H, W).

        Returns:
            Dict[str, Any]: Output dictionary with normalized slc tensor.

        Raises:
            KeyError: If "slc" key is missing from sample.
            ValueError: If "slc" is not 2D (H, W).
            TypeError: If the array type is not supported.
        """
        # Key validation
        if "slc" not in sample:
            raise KeyError('sample must contain key "slc"')


        slc: Tensor = _to_float_tensor(sample["slc"])  # (H, W)

        # Shape validation
        if slc.ndim != 2:
            raise ValueError(f'"slc" must have shape (H, W); got {tuple(slc.shape)}')

        # log(1+x) tolerates zeros; preserves device & dtype
        slc_log = torch.log1p(slc)

        # compute percentiles on the flattened tensor
        q = torch.tensor(
            [self.lower_pct, self.upper_pct], device=slc_log.device
        ) / 100.0
        p_low, p_high = torch.quantile(slc_log.flatten(), q)

        slc_clipped = slc_log.clamp(min=p_low, max=p_high)
        slc_scaled = (slc_clipped - p_low) / (p_high - p_low + EPSILON)

        out = dict(sample)
        out["slc"] = slc_scaled

        LOGGER.info(
            "NormalizeSLCAmpTransform | p_low=%.4f p_high=%.4f device=%s",
            p_low.item(),
            p_high.item(),
            slc_scaled.device,
        )
        return out



#####################################Z-score normalization#####################################
# class NormalizeL0IQTransform:
#     """
#     **Per-sample, per-channel z-score** for raw I/Q data.

#     The incoming sample **must** be a mapping that contains the key
#     ``"iq"`` whose value has shape ``(H, W, 2)`` and dtype
#     ``float32``/``float64`` (NumPy or torch).  The transform converts it to a
#     `torch.float32` tensor with zero mean and unit variance in each
#     channel independently.

#     Example
#     -------
#     ```python
#     tf = NormalizeL0IQTransform()
#     sample = tf({"iq": iq_array})          # dict → dict
#     iq_norm = sample["iq"]                 # torch.Tensor
#     ```
#     """

#     def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
#         iq: Tensor = _to_float_tensor(sample["iq"])  # (H, W, 2)

#         mean = iq.mean(dim=(0, 1), keepdim=True)
#         std = iq.std(dim=(0, 1), keepdim=True).clamp_min(EPSILON)

#         iq_norm = (iq - mean) / std
#         out = dict(sample)  # shallow copy to avoid mutating caller’s dict
#         out["iq"] = iq_norm

#         LOGGER.info(
#             "NormalizeL0IQTransform | mean=%s std=%s device=%s",
#             mean.flatten().tolist(),
#             std.flatten().tolist(),
#             iq_norm.device,
#         )
#         return out