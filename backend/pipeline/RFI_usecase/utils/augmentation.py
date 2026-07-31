"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: augmentation.py

Description:
    PyTorch-compatible data augmentation transforms for range-compressed
    SAR complex data.

Key properties
--------------
Expected input for all transforms:
    • sample["rc"] : (H, W, C) array-like, where:
        - C == 2  : single complex channel (I, Q)
        - C == 4  : two complex channels (I0, Q0, I1, Q1), e.g. VV(I,Q) and VH(I,Q)
        (dtype float32; accepts torch.Tensor or numpy.ndarray)

Provided transforms:
    • RCAmplitudePhasePerturb
        - Amplitude scaling ∈ [a_min, a_max]
        - Phase offset ∈ [-max_phase, +max_phase]
        - Per-pixel or per-patch control

    • RCNarrowbandSpectralDropout
        - Drops one or more narrow spectral bands in range and/or azimuth
            (attenuating or zeroing narrow stripes in frequency domain)

    • RCBandwidthTrim
        - Apodizes spectrum with a narrower window in range/azimuth
            (simulates reduced bandwidth / resolution)

    • RCAzimuthDefocus
        - Applies a quadratic phase term in azimuth frequency (defocusing)

    • RCSubsampleAndInterpolate
        - Downsample in azimuth/range by integer factors, then upsample back
            to original size (simulates coarser sampling / resolution)

All transforms:
    • Accept and return a dict-like sample (e.g., {"rc": ..., ...})
    • Pass through untouched keys
    • Device-aware (preserve tensor device when possible)

History:
    - 2025-09-02:
        First version of augmentation techniques with range-compressed data.
    - 2025-09-19:
        Second version added that is an updated version with the helper functions.
    - 2025-09-22:
        First draft is ready.
    - 2026-01-15:
        Final version updated to support 4-channel (two complex channels) range-compressed data.

---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-09-02

© Copyright INDRA DEIMOS, 2025. All rights reserved.
---------------------------------------------------------------------
"""

from __future__ import annotations
import math
import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from pipeline.RFI_usecase.utils.logging_setup import init_logging


# ------------------------- Logging ------------------------- #
init_logging(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


__all__ = [
    "RCAmplitudePhasePerturb",
    "RCNarrowbandSpectralDropout",
    "RCBandwidthTrim",
    "RCAzimuthDefocus",
    "RCSubsampleAndInterpolate",
]



# ------------------------- Constants ------------------------- #
EPSILON: float = 1e-12


# ------------------------- Helpers ------------------------- #
def _k_from_frac(length: int, frac: float) -> int:
    """Floor to avoid overshoot; clamp to [1, length]."""
    k = int(math.floor(frac * length + 1e-9))
    return max(1, min(length, k))


def _center_indices(length: int, k: int) -> Tuple[int, int]:
    """Symmetric placement about the center bin; bounds-clamped."""
    center = length // 2
    a = center - (k - 1) // 2
    a = max(0, min(a, length - k))
    b = a + k
    return a, b


def _is_identity(H: int, W: int, trim_frac_azimuth: float, trim_frac_range: float) -> bool:
    """
    True if both axes keep the full band (no trimming).
    Mirrors the rounding policy used for window length selection.
    """
    return (_k_from_frac(H, trim_frac_azimuth) == H) and (_k_from_frac(W, trim_frac_range) == W)


def _build_axis_window(
    length: int,
    keep_frac: float,
    window_type: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Build a 1D apodization window.

    Args:
        length: FFT axis length.
        keep_frac: Fraction in (0,1] of the kept bandwidth (centered).
        window_type: {"cosine","hann","hamming"}; "cosine" treated as Hann.
        device: Torch device to place the window.
        dtype: Torch dtype for the window.

    Returns:
        Tensor of shape (length,) with values in [0,1].
    """
    k = _k_from_frac(length, keep_frac)

    if k == length:
        return torch.ones((length,), device=device, dtype=dtype)

    win = torch.zeros((length,), device=device, dtype=dtype)

    if k == 1:
        c = length // 2
        win[c] = 1.0
        return win

    a, b = _center_indices(length, k)
    idx = torch.arange(k, device=device, dtype=dtype)

    wt = "hann" if window_type == "cosine" else window_type
    if wt == "hann":
        w = 0.5 - 0.5 * torch.cos(2.0 * math.pi * idx / (k - 1))
    elif wt == "hamming":
        w = 0.54 - 0.46 * torch.cos(2.0 * math.pi * idx / (k - 1))
    else:
        raise ValueError('window_type must be one of {"cosine","hann","hamming"}')

    win[a:b] = w
    return win


def _to_float_tensor(arr: Any) -> Tensor:
    """
    Convert an input array to a float32 torch.Tensor.
    Supports torch.Tensor and numpy.ndarray.
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


def _ensure_hw_c_even(name: str, x: Tensor, allowed_c: Tuple[int, ...] = (2, 4)) -> None:
    """
    Validate that a tensor has shape (H, W, C) where C is allowed and even.

    For this codebase:
      - C=2 means one complex channel (I,Q)
      - C=4 means two complex channels (I0,Q0,I1,Q1)
    """
    if x.ndim != 3:
        raise ValueError(f'"{name}" must have shape (H, W, C); got {tuple(x.shape)}')
    c = int(x.shape[-1])
    if c not in allowed_c or (c % 2) != 0:
        raise ValueError(f'"{name}" must have last-dim C in {allowed_c} (even); got C={c} with shape {tuple(x.shape)}')


def _expand_to_batch(x: Tensor) -> Tuple[Tensor, bool]:
    """
    Ensure a tensor has a batch dimension.

    Args:
        x: Tensor of shape (H, W, C) or (N, H, W, C)

    Returns:
        (x_batched, added_batch)
    """
    if x.ndim == 3:
        return x.unsqueeze(0), True
    return x, False


def _complex_from_iq_multi(iq: Tensor) -> Tensor:
    """
    Build complex tensors from interleaved I/Q channels.

    Args:
        iq: (N, H, W, C) or (H, W, C) with C=2*K where K is number of complex channels.

    Returns:
        Complex tensor of shape (N, H, W, K) (or (H, W, K) if input is non-batched).
    """
    is_batched = (iq.ndim == 4)
    if not is_batched:
        iq = iq.unsqueeze(0)

    _N, _H, _W, C = iq.shape
    if C % 2 != 0:
        raise ValueError(f"I/Q channel count must be even; got C={C}")

    K = C // 2
    iq_pairs = iq.view(iq.shape[0], iq.shape[1], iq.shape[2], K, 2)  # (N,H,W,K,2)
    I = iq_pairs[..., 0]
    Q = iq_pairs[..., 1]
    z = torch.complex(I, Q)  # (N,H,W,K)

    if is_batched:
        return z
    return z[0]


def _iq_from_complex_multi(z: Tensor) -> Tensor:
    """
    Split complex tensors into interleaved I/Q channels.

    Args:
        z: Complex tensor of shape (N, H, W, K) or (H, W, K)

    Returns:
        Real tensor of shape (N, H, W, 2*K) or (H, W, 2*K)
        with channel order: [I0, Q0, I1, Q1, ...].
    """
    is_batched = (z.ndim == 4)
    if not is_batched:
        z = z.unsqueeze(0)

    z_iq = torch.stack([z.real, z.imag], dim=-1)  # (N,H,W,K,2)
    out = z_iq.reshape(z.shape[0], z.shape[1], z.shape[2], z.shape[3] * 2)  # (N,H,W,2K)

    if is_batched:
        return out
    return out[0]


def _fft2c(x: Tensor) -> Tensor:
    """
    Centered 2D FFT for complex tensors.

    Supports:
      - (N,H,W) or (H,W)
      - (N,H,W,K) or (H,W,K) via FFT over H,W (dims -3 and -2 for 4D, -2 and -1 for 3D).
    """
    if x.ndim == 2:
        return torch.fft.fftshift(torch.fft.fft2(x, norm="backward"), dim=(-2, -1))
    if x.ndim == 3:
        return torch.fft.fftshift(torch.fft.fft2(x, norm="backward"), dim=(-2, -1))
    if x.ndim == 4:
        # FFT over H,W, keep K as last dim
        X = torch.fft.fft2(x, dim=(-3, -2), norm="backward")
        return torch.fft.fftshift(X, dim=(-3, -2))
    raise ValueError(f"_fft2c expects 2D, 3D, or 4D complex tensor; got shape {tuple(x.shape)}")


def _ifft2c(X: Tensor) -> Tensor:
    """
    Inverse centered 2D FFT for complex tensors.
    Mirrors _fft2c.
    """
    if X.ndim == 2:
        return torch.fft.ifft2(torch.fft.ifftshift(X, dim=(-2, -1)), norm="backward")
    if X.ndim == 3:
        return torch.fft.ifft2(torch.fft.ifftshift(X, dim=(-2, -1)), norm="backward")
    if X.ndim == 4:
        x = torch.fft.ifftshift(X, dim=(-3, -2))
        return torch.fft.ifft2(x, dim=(-3, -2), norm="backward")
    raise ValueError(f"_ifft2c expects 2D, 3D, or 4D complex tensor; got shape {tuple(X.shape)}")



#------------------------- Transforms ------------------------- #
class RCAmplitudePhasePerturb:
    """
    Amplitude and phase perturbation for range-compressed SAR patches with I/Q channels.

    For each complex channel, this perturbs:
        - amplitude scaled by alpha in [a_min, a_max]
        - phase offset dphi in [-max_phase_shift, +max_phase_shift]

    Behavior with C=4:
        Applies the perturbation independently per complex channel by default.

    Args:
        amp_scale_range: (a_min, a_max)
        max_phase_shift: radians
        per_pixel: if True draw per pixel; else per patch
    """

    def __init__(
        self,
        amp_scale_range: Tuple[float, float] = (0.98, 1.02),
        max_phase_shift: float = 0.0,
        per_pixel: bool = False,
    ) -> None:
        a0, a1 = amp_scale_range
        if a0 < 0 or a1 < 0 or a0 > a1:
            raise ValueError("amp_scale_range must satisfy 0 <= a_min <= a_max")
        if max_phase_shift < 0:
            raise ValueError("max_phase_shift must be >= 0")

        self.amp_scale_range = (float(a0), float(a1))
        self.max_phase_shift = float(max_phase_shift)
        self.per_pixel = bool(per_pixel)

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        if "rc" not in sample:
            raise KeyError('sample must contain key "rc"')

        rc = _to_float_tensor(sample["rc"])  # (H,W,C) or (N,H,W,C)
        rc_b, added_batch = _expand_to_batch(rc)
        _ensure_hw_c_even("rc", rc_b[0], allowed_c=(2, 4))

        N, H, W, C = rc_b.shape
        device = rc_b.device
        K = C // 2  # number of complex channels

        # Convert to complex: (N,H,W,K)
        z = _complex_from_iq_multi(rc_b)

        if self.per_pixel:
            alpha = torch.empty((N, H, W, K), device=device).uniform_(*self.amp_scale_range)
            dphi = torch.empty((N, H, W, K), device=device).uniform_(-self.max_phase_shift, self.max_phase_shift)
        else:
            alpha = torch.empty((N, 1, 1, K), device=device).uniform_(*self.amp_scale_range)
            dphi = torch.empty((N, 1, 1, K), device=device).uniform_(-self.max_phase_shift, self.max_phase_shift)

        amp = torch.abs(z).clamp_min(EPSILON)
        phs = torch.angle(z)

        amp_new = amp * alpha
        phs_new = phs + dphi

        z_new = amp_new * torch.exp(1j * phs_new)
        rc_out = _iq_from_complex_multi(z_new)  # (N,H,W,C)

        if added_batch:
            rc_out = rc_out[0]

        out = dict(sample)
        out["rc"] = rc_out
        LOGGER.info(
            "RCAmplitudePhasePerturb | per_pixel=%s amp in [%.3f,%.3f] max_dphi=%.3f rad C=%d",
            self.per_pixel,
            self.amp_scale_range[0],
            self.amp_scale_range[1],
            self.max_phase_shift,
            C,
        )
        return out


class RCNarrowbandSpectralDropout:
    """
    Drop narrow spectral bands in the 2D frequency domain.
    Works for both C=2 and C=4 by applying the same frequency mask to each complex channel.
    """

    def __init__(
        self,
        bands_range: Optional[Sequence[Tuple[float, float]]] = None,
        bands_azimuth: Optional[Sequence[Tuple[float, float]]] = None,
        soft: bool = False,
        edge_taper: float = 0.1,
    ) -> None:
        self.bands_range = bands_range or []
        self.bands_azimuth = bands_azimuth or []
        if not (0.0 <= edge_taper < 0.5):
            raise ValueError("edge_taper must be in [0, 0.5)")
        self.soft = bool(soft)
        self.edge_taper = float(edge_taper)

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        if "rc" not in sample:
            raise KeyError('sample must contain key "rc"')

        rc = _to_float_tensor(sample["rc"])
        rc_b, added_batch = _expand_to_batch(rc)
        _ensure_hw_c_even("rc", rc_b[0], allowed_c=(2, 4))

        N, H, W, C = rc_b.shape
        K = C // 2

        z = _complex_from_iq_multi(rc_b)  # (N,H,W,K)
        Z = _fft2c(z)  # centered spectrum (N,H,W,K)

        # mask on (N,H,W), then broadcast to K
        mask = torch.ones((N, H, W), device=Z.device, dtype=Z.real.dtype)

        def _band_to_idx(length: int, frac0: float, frac1: float) -> Tuple[int, int]:
            a = int(round(frac0 * length))
            b = int(round(frac1 * length))
            a = max(0, min(length, a))
            b = max(0, min(length, b))
            if b < a:
                a, b = b, a
            return a, b

        # Range-axis bands (columns)
        for (f0, f1) in self.bands_range:
            c0, c1 = _band_to_idx(W, f0, f1)
            if c1 <= c0:
                continue
            if self.soft:
                bw = c1 - c0
                taper = int(max(1, round(self.edge_taper * bw)))
                win = torch.ones((W,), device=Z.device, dtype=mask.dtype)
                core0, core1 = c0 + taper, c1 - taper
                if taper > 0:
                    t = torch.linspace(0, math.pi, steps=taper, device=Z.device, dtype=mask.dtype)
                    win[c0:core0] = 0.5 * (1 + torch.cos(t))
                    win[core1:c1] = 0.5 * (1 + torch.cos(t - math.pi))
                if core1 > core0:
                    win[core0:core1] = 0.0
                mask = mask * win.view(1, 1, W)
            else:
                mask[..., c0:c1] = 0.0

        # Azimuth-axis bands (rows)
        for (f0, f1) in self.bands_azimuth:
            r0, r1 = _band_to_idx(H, f0, f1)
            if r1 <= r0:
                continue
            if self.soft:
                bw = r1 - r0
                taper = int(max(1, round(self.edge_taper * bw)))
                win = torch.ones((H,), device=Z.device, dtype=mask.dtype)
                core0, core1 = r0 + taper, r1 - taper
                if taper > 0:
                    t = torch.linspace(0, math.pi, steps=taper, device=Z.device, dtype=mask.dtype)
                    win[r0:core0] = 0.5 * (1 + torch.cos(t))
                    win[core1:r1] = 0.5 * (1 + torch.cos(t - math.pi))
                if core1 > core0:
                    win[core0:core1] = 0.0
                mask = mask * win.view(1, H, 1)
            else:
                mask[:, r0:r1, :] = 0.0

        mask4 = mask.unsqueeze(-1)  # (N,H,W,1)
        Zd = Z * mask4  # broadcast over K
        z_out = _ifft2c(Zd)  # (N,H,W,K)
        rc_out = _iq_from_complex_multi(z_out)  # (N,H,W,C)

        if added_batch:
            rc_out = rc_out[0]

        out = dict(sample)
        out["rc"] = rc_out
        LOGGER.info(
            "RCNarrowbandSpectralDropout | range_bands=%s az_bands=%s soft=%s taper=%.2f C=%d",
            self.bands_range,
            self.bands_azimuth,
            self.soft,
            self.edge_taper,
            C,
        )
        return out


class RCBandwidthTrim:
    """
    Spectral bandwidth trimming via apodization windowing.
    Works for both C=2 and C=4 by applying the same 2D window to each complex channel.
    """

    def __init__(
        self,
        trim_frac_range: float = 1.0,
        trim_frac_azimuth: float = 1.0,
        window: str = "cosine",
    ) -> None:
        if not (0.0 < trim_frac_range <= 1.0):
            raise ValueError("trim_frac_range must be in (0, 1]")
        if not (0.0 < trim_frac_azimuth <= 1.0):
            raise ValueError("trim_frac_azimuth must be in (0, 1]")
        if window not in {"cosine", "hann", "hamming"}:
            raise ValueError('window must be one of {"cosine", "hann", "hamming"}')

        self.window = "hann" if window == "cosine" else window
        self.trim_frac_range = float(trim_frac_range)
        self.trim_frac_azimuth = float(trim_frac_azimuth)

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        if "rc" not in sample:
            raise KeyError('sample must contain key "rc"')

        rc = _to_float_tensor(sample["rc"])
        rc_b, added_batch = _expand_to_batch(rc)
        _ensure_hw_c_even("rc", rc_b[0], allowed_c=(2, 4))

        N, H, W, C = rc_b.shape

        # Identity fast-path
        if _is_identity(H, W, self.trim_frac_azimuth, self.trim_frac_range):
            out = dict(sample)
            out["rc"] = rc_b[0] if added_batch else rc
            LOGGER.info(
                "RCBandwidthTrim | identity (trim_range=%.2f trim_az=%.2f window=%s) C=%d",
                self.trim_frac_range,
                self.trim_frac_azimuth,
                self.window,
                C,
            )
            return out

        z = _complex_from_iq_multi(rc_b)  # (N,H,W,K)
        Z = _fft2c(z)  # (N,H,W,K)

        wr = _build_axis_window(W, self.trim_frac_range, self.window, Z.device, dtype=Z.real.dtype)
        wa = _build_axis_window(H, self.trim_frac_azimuth, self.window, Z.device, dtype=Z.real.dtype)

        W2D = (wa.view(1, H, 1) * wr.view(1, 1, W)).unsqueeze(-1)  # (1,H,W,1)
        Zd = Z * W2D
        z_out = _ifft2c(Zd)
        rc_out = _iq_from_complex_multi(z_out)

        if added_batch:
            rc_out = rc_out[0]

        out = dict(sample)
        out["rc"] = rc_out
        LOGGER.info(
            "RCBandwidthTrim | trim_range=%.2f trim_az=%.2f window=%s C=%d",
            self.trim_frac_range,
            self.trim_frac_azimuth,
            self.window,
            C,
        )
        return out


class RCAzimuthDefocus:
    """
    Apply a quadratic phase error in azimuth frequency to simulate azimuth defocus.
    Works for both C=2 and C=4 by applying the same phase term to each complex channel.
    """

    def __init__(self, kappa: float = 5e-4) -> None:
        self.kappa = float(kappa)

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        if "rc" not in sample:
            raise KeyError('sample must contain key "rc"')

        rc = _to_float_tensor(sample["rc"])
        rc_b, added_batch = _expand_to_batch(rc)
        _ensure_hw_c_even("rc", rc_b[0], allowed_c=(2, 4))

        N, H, W, C = rc_b.shape

        z = _complex_from_iq_multi(rc_b)  # (N,H,W,K)
        Z = _fft2c(z)  # (N,H,W,K)

        # Centered normalized azimuth frequency in [-0.5, 0.5)
        k = (torch.arange(H, device=Z.device, dtype=Z.real.dtype) - H // 2) / H  
        theta = self.kappa * k.square()  

        phase_1d = torch.polar(torch.ones_like(theta), theta).to(dtype=Z.dtype)  
        phase = phase_1d.view(1, H, 1, 1)  # (1,H,1,1), broadcast over N,W,K

        Zd = Z * phase
        z_out = _ifft2c(Zd)
        rc_out = _iq_from_complex_multi(z_out)

        if added_batch:
            rc_out = rc_out[0]

        out = dict(sample)
        out["rc"] = rc_out
        LOGGER.info("RCAzimuthDefocus | kappa=%.3e C=%d", self.kappa, C)
        return out


class RCSubsampleAndInterpolate:
    """
    Downsample and then upsample range-compressed data to simulate coarser sampling.
    Works for both C=2 and C=4 (treats channels as regular image channels).
    """

    def __init__(
        self,
        az_factor: int = 1,
        rg_factor: int = 1,
        down_mode: str = "avg",
        up_mode: str = "bilinear",
    ) -> None:
        if az_factor < 1 or rg_factor < 1:
            raise ValueError("az_factor and rg_factor must be >= 1")
        if down_mode not in {"stride", "avg"}:
            raise ValueError('down_mode must be one of {"stride", "avg"}')
        if up_mode not in {"nearest", "bilinear"}:
            raise ValueError('up_mode must be one of {"nearest", "bilinear"}')

        self.az_factor = int(az_factor)
        self.rg_factor = int(rg_factor)
        self.down_mode = down_mode
        self.up_mode = up_mode

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        if "rc" not in sample:
            raise KeyError('sample must contain key "rc"')

        rc = _to_float_tensor(sample["rc"])
        rc_b, added_batch = _expand_to_batch(rc)
        _ensure_hw_c_even("rc", rc_b[0], allowed_c=(2, 4))

        _N, H, W, C = rc_b.shape

        # (N, C, H, W)
        x = rc_b.permute(0, 3, 1, 2).contiguous()

        if self.down_mode == "stride":
            x_ds = x[:, :, :: self.az_factor, :: self.rg_factor]
        else:
            k_h = self.az_factor
            k_w = self.rg_factor
            x_ds = F.avg_pool2d(x, kernel_size=(k_h, k_w), stride=(k_h, k_w), ceil_mode=False)

        align_corners = False if self.up_mode != "nearest" else None
        x_up = F.interpolate(x_ds, size=(H, W), mode=self.up_mode, align_corners=align_corners)

        rc_out = x_up.permute(0, 2, 3, 1).contiguous()  # (N,H,W,C)
        if added_batch:
            rc_out = rc_out[0]

        out = dict(sample)
        out["rc"] = rc_out
        LOGGER.info(
            "RCSubsampleAndInterpolate | az=%d rg=%d down=%s up=%s C=%d",
            self.az_factor,
            self.rg_factor,
            self.down_mode,
            self.up_mode,
            C,
        )
        return out