"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: sar_rc_augmentation.py

Description:
    PyTorch-compatible data augmentation transforms for range-compressed
    SAR complex data.

    Expected input for all transforms:
        • sample["rc"] : (H, W, 2) array-like, where rc[..., 0] = I, rc[..., 1] = Q
          (dtype float32; accepts torch.Tensor or numpy.ndarray)

    Provided transforms:
        • RCAmplitudePhasePerturb
            - Amplitude scaling α ∈ [a_min, a_max]
            - Phase offset Δφ ∈ [-max_phase, +max_phase]
            - Per-pixel or per-patch control

        • RCNarrowbandSpectralDropout
            - Drops one or more narrow spectral bands in range and/or azimuth
              (zeroing-out narrow stripes in frequency domain)

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
        First vesion of augmentation techniques with range-compressed data.
    - 2025-09-19:
        Second version added that is an updated version with the helper functions.
    - 2025-09-22:
        First draft is ready.
    
---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-09-02

© Copyright INDRA DEIMOS, 2025. All rights reserved.
---------------------------------------------------------------------
"""

from __future__ import annotations
from typing import Any, Dict, Mapping, Optional, Tuple, Sequence
import math
import torch
from torch import Tensor
import torch.nn.functional as F
import logging
from logging_setup import init_logging
from typing import Any, Dict, Mapping, Tuple


# Configure logging once
init_logging(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


__all__ = [
    "RCAmplitudePhasePerturb",
    "RCNarrowbandSpectralDropout",
    "RCBandwidthTrim",
    "RCAzimuthDefocus",
    "RCSubsampleAndInterpolate",
]


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
EPSILON: float = 1e-12


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

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
    device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Build a 1D apodization window (top-level helper, no self).

    Args:
        length: FFT axis length.
        keep_frac: Fraction in (0,1] of the *kept* bandwidth (centered).
        window_type: {"cosine","hann","hamming"}; "cosine" treated as Hann.
        device: Torch device to place the window.
        dtype: Torch dtype for the window (default float32).

    Returns:
        Tensor of shape (length,) with values in [0,1].
    """
    k = _k_from_frac(length, keep_frac)

    if k == length:
        return torch.ones((length,), device=device, dtype=dtype)

    win = torch.zeros((length,), device=device, dtype=dtype)

    # Single-bin keep: delta at DC (center)
    if k == 1:
        c = length // 2
        win[c] = 1.0
        return win

    a, b = _center_indices(length, k)
    idx = torch.arange(k, device=device, dtype=dtype)  # [0..k-1]

    # Treat "cosine" as symmetric raised-cosine (Hann).
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

    Args:
        arr (Any):
            The input array. Supported types are:
              • torch.Tensor (any dtype) will be converted to float32
              • numpy.ndarray (any dtype) will be converted to float32

    Returns:
        torch.Tensor:
            A float32 tensor view/copy of the input on CPU or the original device
            if arr is already a torch.Tensor.

    Raises:
        TypeError:
            If the input is not a torch.Tensor or numpy.ndarray.
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


def _ensure_hw2(name: str, x: Tensor) -> None:
    """
    Validate that a tensor has shape (H, W, 2) for I/Q complex data.

    Args:
        name (str):
            Field name used in error messages (e.g., "rc").
        x (torch.Tensor):
            Input tensor to validate.

    Raises:
        ValueError:
            If 'x' is not a 3D tensor with the last dimension equal to 2.
    """
    if x.ndim != 3 or x.shape[-1] != 2:
        raise ValueError(f'"{name}" must have shape (H, W, 2); got {tuple(x.shape)}')


def _complex_from_iq(iq: Tensor) -> Tensor:
    """
    Build a complex tensor from I/Q channels.

    Args:
        iq (torch.Tensor):
            Tensor of shape (H, W, 2) where [..., 0]=I and [..., 1]=Q.

    Returns:
        torch.Tensor:
            Complex tensor of shape (H, W), dtype complex64 (matching float32 components).
    """
    I, Q = iq[..., 0], iq[..., 1]
    return torch.complex(I, Q)


def _iq_from_complex(z: Tensor) -> Tensor:
    """
    Split a complex tensor into I/Q channels.

    Args:
        z (torch.Tensor):
            Complex tensor of shape (H, W) or (N, H, W).

    Returns:
        torch.Tensor:
            Real-valued tensor of shape (H, W, 2) (or (N, H, W, 2)) with
            [..., 0]=real(z), [..., 1]=imag(z).
    """
    return torch.stack([z.real, z.imag], dim=-1)


def _fft2c(x: Tensor) -> Tensor:
    """
    Centered 2D FFT (fftshifted) for complex tensors.

    Args:
        x (torch.Tensor):
            Complex tensor (H, W) or (N, H, W).

    Returns:
        torch.Tensor:
            Complex tensor with centered spectrum (same shape as input).
    """
    return torch.fft.fftshift(torch.fft.fft2(x, norm="backward"), dim=(-2, -1))


def _ifft2c(X: Tensor) -> Tensor:
    """
    Inverse centered 2D FFT (with ifftshift) for complex tensors.

    Args:
        X (torch.Tensor):
            Complex tensor (H, W) or (N, H, W) in centered frequency domain.

    Returns:
        torch.Tensor:
            Complex tensor in the spatial/time domain (same shape as input).
    """
    return torch.fft.ifft2(torch.fft.ifftshift(X, dim=(-2, -1)), norm="backward")


def _expand_to_batch(x: Tensor) -> Tuple[Tensor, bool]:
    """
    Ensure a tensor has a batch dimension.

    Args:
        x (torch.Tensor):
            Tensor of shape (H, W, 2) **or** (N, H, W, 2).

    Returns:
        Tuple[torch.Tensor, bool]:
            - Tensor of shape (N, H, W, 2).
            - Boolean flag indicating whether the batch dim was added (True if input was non-batched).
    """
    if x.ndim == 3:
        return x.unsqueeze(0), True
    return x, False


# ---------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------
class RCAmplitudePhasePerturb:
    """
    Amplitude and phase perturbation for range-compressed I/Q SAR patches.

    This transform perturbs the complex signal by:
        • Scaling the amplitude by α ∈ [a_min, a_max]
        • Adding a phase offset Δφ ∈ [-max_phase_shift, +max_phase_shift]

    Optionally, perturbations can be sampled per-pixel (stronger) or per-patch (milder).

    Args:
        amp_scale_range (Tuple[float, float], optional):
            Closed interval [a_min, a_max] for amplitude scaling. Must satisfy 0 ≤ a_min ≤ a_max.
            Default: (0.9, 1.1)

        max_phase_shift (float, optional):
            Maximum absolute phase offset (radians). Phase is sampled uniformly from
            [-max_phase_shift, +max_phase_shift]. Default: 0.1

        per_pixel (bool, optional):
            If True, draws α and Δφ per pixel. If False, draws one α and one Δφ for the whole patch.
            Default: True
            
    """

    def __init__(
        self,
        amp_scale_range: Tuple[float, float] = (0.9, 1.1),
        max_phase_shift: float = 0.1,
        per_pixel: bool = True,
    ) -> None:
        a0, a1 = amp_scale_range
        if a0 < 0 or a1 < 0 or a0 > a1:
            raise ValueError("amp_scale_range must satisfy 0 ≤ a_min ≤ a_max")
        if max_phase_shift < 0:
            raise ValueError("max_phase_shift must be ≥ 0")

        self.amp_scale_range = (float(a0), float(a1))
        self.max_phase_shift = float(max_phase_shift)
        self.per_pixel = bool(per_pixel)

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Apply amplitude and phase perturbation to the "rc" field of the sample.

        Args:
            sample (Mapping[str, Any]):
                Dictionary with key "rc" mapping to a (H, W, 2) array-like of float32.

        Returns:
            Dict[str, Any]:
                New dictionary with "rc" replaced by perturbed data.
                Other keys are preserved.

        Raises:
            KeyError:
                If "rc" key is missing from the input sample.
            ValueError:
                If "rc" does not have shape (H, W, 2).
        """
        if "rc" not in sample:
            raise KeyError('sample must contain key "rc"')

        rc = _to_float_tensor(sample["rc"])  # (H, W, 2) or (N, H, W, 2)
        # Support both single patch and batched input
        rc_b, added_batch = _expand_to_batch(rc)
        _ensure_hw2("rc", rc_b[0])  # check one sample

        N, H, W, _ = rc_b.shape
        device = rc_b.device

        # Convert to complex
        z = _complex_from_iq(rc_b)

        # Draw α and Δφ
        if self.per_pixel:
            alpha = torch.empty((N, H, W), device=device).uniform_(*self.amp_scale_range)
            dphi = torch.empty((N, H, W), device=device).uniform_(-self.max_phase_shift, self.max_phase_shift)
        else:
            alpha = torch.empty((N, 1, 1), device=device).uniform_(*self.amp_scale_range)
            dphi = torch.empty((N, 1, 1), device=device).uniform_(-self.max_phase_shift, self.max_phase_shift)

        # Apply in polar form
        amp = torch.abs(z).clamp_min(EPSILON)
        phs = torch.angle(z)
        amp_new = amp * alpha
        phs_new = phs + dphi

        z_new = amp_new * torch.exp(1j * phs_new)
        rc_out = _iq_from_complex(z_new)

        # Squeeze batch if needed
        if added_batch:
            rc_out = rc_out[0]

        out = dict(sample)
        out["rc"] = rc_out
        LOGGER.info(
            "RCAmplitudePhasePerturb | per_pixel=%s amp∈[%.3f,%.3f] maxΔφ=%.3f rad",
            self.per_pixel, self.amp_scale_range[0], self.amp_scale_range[1], self.max_phase_shift
        )
        return out


class RCNarrowbandSpectralDropout:
    """
    Drop narrow spectral bands in the 2D frequency domain.

    This simulates missing/nulled sub-bands (e.g., interference excision or instrument gaps).
    Frequency-domain stripes are zeroed along the range and/or azimuth axes.

    Args:
        bands_range (Sequence[Tuple[float, float]], optional):
            Sequence of fractional bands (start, end) to drop along range frequency (W-axis),
            each in [0, 1], where 0=start of spectrum and 1=end. Example: [(0.45, 0.55)].

        bands_azimuth (Sequence[Tuple[float, float]], optional):
            Sequence of fractional bands to drop along azimuth frequency (H-axis).

        soft (bool, optional):
            If True, apply smooth attenuation (taper to zero with cosine edges).
            If False, hard zeroing inside selected bands. Default: False

        edge_taper (float, optional): Fraction of band width used for cosine taper at edges when soft=True.
            Must be in [0, 0.5). Default: 0.1

    Notes:
        - Operates on complex data in frequency domain with centered FFT.
        - Returns to spatial domain via inverse FFT.
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
        """
        Apply spectral dropout on "rc".

        Args:
            sample (Mapping[str, Any]):
                Dictionary with key "rc" mapping to (H, W, 2) or (N, H, W, 2).

        Returns:
            Dict[str, Any]:
                New dictionary with dropout applied; shape preserved.

        Raises:
            KeyError:
                If "rc" key is missing.
            ValueError:
                If "rc" does not have shape (H, W, 2) (or batch variant).
        """
        if "rc" not in sample:
            raise KeyError('sample must contain key "rc"')

        rc = _to_float_tensor(sample["rc"])
        rc_b, added_batch = _expand_to_batch(rc)
        _ensure_hw2("rc", rc_b[0])

        # Complex
        z = _complex_from_iq(rc_b)    # (N, H, W)
        Z = _fft2c(z)                  # centered spectrum

        N, H, W = Z.shape
        mask = torch.ones_like(Z.real)  # (N, H, W) real-valued mask

        # Helper to convert fractional [0,1] band to index range
        def _band_to_idx(length: int, frac0: float, frac1: float) -> Tuple[int, int]:
            a = int(round(frac0 * length))
            b = int(round(frac1 * length))
            a = max(0, min(length, a))
            b = max(0, min(length, b))
            if b < a:
                a, b = b, a
            return a, b

        # ---------------- Range-axis bands (columns) ----------------
        for (f0, f1) in self.bands_range:
            c0, c1 = _band_to_idx(W, f0, f1)
            if c1 <= c0:
                continue
            if self.soft:
                bw = c1 - c0
                taper = int(max(1, round(self.edge_taper * bw)))
                win = torch.ones((W,), device=Z.device)
                # Cosine-notch with smooth edges
                core0, core1 = c0 + taper, c1 - taper
                if taper > 0:
                    t = torch.linspace(0, math.pi, steps=taper, device=Z.device)
                    # left ramp 1 -> 0 entering the band
                    win[c0:core0] = 0.5 * (1 + torch.cos(t))                 # 1 → 0
                    # right ramp 0 -> 1 exiting the band
                    win[core1:c1] = 0.5 * (1 + torch.cos(t - math.pi))       # 0 → 1
                if core1 > core0:
                    win[core0:core1] = 0.0
                mask = mask * win.view(1, 1, W)  # broadcast across (N,H,W)
            else:
                mask[..., c0:c1] = 0.0

        # ---------------- Azimuth-axis bands (rows) ----------------
        for (f0, f1) in self.bands_azimuth:
            r0, r1 = _band_to_idx(H, f0, f1)
            if r1 <= r0:
                continue
            if self.soft:
                bw = r1 - r0
                taper = int(max(1, round(self.edge_taper * bw)))
                win = torch.ones((H,), device=Z.device)
                core0, core1 = r0 + taper, r1 - taper
                if taper > 0:
                    t = torch.linspace(0, math.pi, steps=taper, device=Z.device)
                    # top ramp 1 -> 0 entering the band
                    win[r0:core0] = 0.5 * (1 + torch.cos(t))                 # 1 → 0
                    # bottom ramp 0 -> 1 exiting the band
                    win[core1:r1] = 0.5 * (1 + torch.cos(t - math.pi))       # 0 → 1
                if core1 > core0:
                    win[core0:core1] = 0.0
                mask = mask * win.view(1, H, 1)  # broadcast across (N,H,W)
            else:
                mask[:, r0:r1, :] = 0.0

        # Apply mask and invert FFT
        Zd = Z * mask
        z_out = _ifft2c(Zd)
        rc_out = _iq_from_complex(z_out)

        if added_batch:
            rc_out = rc_out[0]

        out = dict(sample)
        out["rc"] = rc_out
        LOGGER.info(
            "RCNarrowbandSpectralDropout | range_bands=%s az_bands=%s soft=%s taper=%.2f",
            self.bands_range, self.bands_azimuth, self.soft, self.edge_taper
        )
        return out
    

class RCBandwidthTrim:
    """
    Spectral bandwidth trimming via apodization windowing.It applies a symmetric window in the frequency domain to narrow the effective
    bandwidth in range (W) and/or azimuth (H). Outside the kept band → zeros.

    Args:
        trim_frac_range: Fraction in (0, 1] of *kept* bandwidth along range (W).
        trim_frac_azimuth: Same for azimuth (H).
        window: {"cosine", "hann", "hamming"}; "cosine" is treated as Hann.

    Notes:
        - If trim_frac_* == 1.0 → exact identity (no change).
        - k == 1 keeps exactly the DC (center) bin.
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

        # Alias "cosine" to the symmetric raised-cosine (Hann).
        self.window = "hann" if window == "cosine" else window
        self.trim_frac_range = float(trim_frac_range)
        self.trim_frac_azimuth = float(trim_frac_azimuth)

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        if "rc" not in sample:
            raise KeyError('sample must contain key "rc"')

        rc = _to_float_tensor(sample["rc"])
        rc_b, added_batch = _expand_to_batch(rc)
        _ensure_hw2("rc", rc_b[0])

        z = _complex_from_iq(rc_b)  # (N,H,W) complex
        _, H, W = z.shape

        # Fast-path identity: nothing to trim
        if _is_identity(H, W, self.trim_frac_azimuth, self.trim_frac_range):
            out = dict(sample)
            out["rc"] = rc_b[0] if added_batch else rc
            LOGGER.info(
                "RCBandwidthTrim | identity (trim_range=%.2f trim_az=%.2f window=%s)",
                self.trim_frac_range, self.trim_frac_azimuth, self.window
            )
            return out

        Z = _fft2c(z)  # centered FFT

        # Use the top-level helper for window construction
        wr = _build_axis_window(W, self.trim_frac_range, self.window, Z.device)
        wa = _build_axis_window(H, self.trim_frac_azimuth, self.window, Z.device)

        W2D = wa.view(1, H, 1) * wr.view(1, 1, W)  # (1,H,W), real
        Zd = Z * W2D  # broadcast to (N,H,W)

        z_out = _ifft2c(Zd)
        rc_out = _iq_from_complex(z_out)

        if added_batch:
            rc_out = rc_out[0]

        out = dict(sample)
        out["rc"] = rc_out
        LOGGER.info(
            "RCBandwidthTrim | trim_range=%.2f trim_az=%.2f window=%s",
            self.trim_frac_range, self.trim_frac_azimuth, self.window
        )
        return out


class RCAzimuthDefocus:
    """
    Apply a quadratic phase error in azimuth frequency to simulate residual
    motion / Doppler-rate mismatch, causing azimuth defocus.

    Phase term: exp(j * kappa * k^2), where k is the centered normalized
    azimuth frequency in [-0.5, 0.5).

    Args:
        kappa (float): Strength of the quadratic phase (rad per (cycles/sample)^2).
                       Larger |kappa| -> stronger defocus. Default: 5e-4
    """

    def __init__(self, kappa: float = 5e-4) -> None:
        self.kappa = float(kappa)

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        if "rc" not in sample:
            raise KeyError('sample must contain key "rc"')

        rc = _to_float_tensor(sample["rc"])
        rc_b, added_batch = _expand_to_batch(rc)
        _ensure_hw2("rc", rc_b[0])

        # (N, H, W) complex
        z = _complex_from_iq(rc_b)
        Z = _fft2c(z)

        H = int(Z.size(-2))


        # Centered normalized azimuth frequency k in [-0.5, 0.5)
        k = (torch.arange(H, device=Z.device, dtype=Z.real.dtype) - H // 2) / H  # (H,)

        # Quadratic phase: exp(j * kappa * k^2)
        # Build as unit-magnitude complex via polar to keep dtype stable (complex64 if Z is complex64)
        theta = self.kappa * k.square()                                  # (H,)
        phase_1d = torch.polar(torch.ones_like(theta), theta)             # complex (H,)
        phase = phase_1d.view(1, H, 1).to(dtype=Z.dtype)                  # (1, H, 1)

        Zd = Z * phase                                                   

        z_out = _ifft2c(Zd)
        rc_out = _iq_from_complex(z_out)

        if added_batch:
            rc_out = rc_out[0]

        out = dict(sample)
        out["rc"] = rc_out
        LOGGER.info("RCAzimuthDefocus | kappa=%.3e", self.kappa)
        return out


class RCSubsampleAndInterpolate:
    """
    Downsample and then upsample range-compressed I/Q data to simulate coarser sampling.

    The transform:
        1) Downsamples by integer factors in azimuth and/or range (stride slicing or avg-pooling)
        2) Upsamples back to original (H, W) using interpolation (nearest/bilinear)

    Args:
        az_factor (int, optional):
            Downsampling factor along azimuth (rows). Must be ≥ 1. Default: 1 (no-op).

        rg_factor (int, optional):
            Downsampling factor along range (cols). Must be ≥ 1. Default: 1 (no-op).

        down_mode (str, optional):
            Downsampling method: {"stride", "avg"}. "stride" picks every k-th sample,
            "avg" uses average pooling with kernel=stride. Default: "avg"

        up_mode (str, optional):
            Upsampling interpolation for returning to original size: {"nearest", "bilinear"}.
            Default: "bilinear"

    """

    def __init__(
        self,
        az_factor: int = 1,
        rg_factor: int = 1,
        down_mode: str = "avg",
        up_mode: str = "bilinear",
    ) -> None:
        if az_factor < 1 or rg_factor < 1:
            raise ValueError("az_factor and rg_factor must be ≥ 1")
        if down_mode not in {"stride", "avg"}:
            raise ValueError('down_mode must be one of {"stride", "avg"}')
        if up_mode not in {"nearest", "bilinear"}:
            raise ValueError('up_mode must be one of {"nearest", "bilinear"}')

        self.az_factor = int(az_factor)
        self.rg_factor = int(rg_factor)
        self.down_mode = down_mode
        self.up_mode = up_mode

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Apply subsampling and interpolation on "rc".

        Args:
            sample (Mapping[str, Any]):
                Dictionary with key "rc" mapping to (H, W, 2) or (N, H, W, 2).

        Returns:
            Dict[str, Any]:
                New dictionary with "rc" downsampled and then upsampled back to
                original size (shape preserved).

        Raises:
            KeyError:
                If "rc" key is missing.
            ValueError:
                If "rc" does not have shape (H, W, 2) (or batch variant).
        """
        if "rc" not in sample:
            raise KeyError('sample must contain key "rc"')

        rc = _to_float_tensor(sample["rc"])
        rc_b, added_batch = _expand_to_batch(rc)
        _ensure_hw2("rc", rc_b[0])

        _N, H, W, _C = rc_b.shape
        # Prepare for grid ops: use (N, C, H, W)
        x = rc_b.permute(0, 3, 1, 2).contiguous()   # (N, 2, H, W)

        # 1) Downsample
        if self.down_mode == "stride":
            x_ds = x[:, :, :: self.az_factor, :: self.rg_factor]
        else:  # "avg" pooling
            k_h = self.az_factor
            k_w = self.rg_factor
            x_ds = F.avg_pool2d(x, kernel_size=(k_h, k_w), stride=(k_h, k_w), ceil_mode=False)

        # 2) Upsample back to original size
        x_up = F.interpolate(x_ds, size=(H, W), mode=self.up_mode, align_corners=False if self.up_mode != "nearest" else None)

        rc_out = x_up.permute(0, 2, 3, 1).contiguous()  # (N, H, W, 2)
        if added_batch:
            rc_out = rc_out[0]

        out = dict(sample)
        out["rc"] = rc_out
        LOGGER.info(
            "RCSubsampleAndInterpolate | az=%d rg=%d down=%s up=%s",
            self.az_factor, self.rg_factor, self.down_mode, self.up_mode
        )
        return out
