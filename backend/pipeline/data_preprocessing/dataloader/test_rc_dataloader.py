"""
---------------------------------------------------------------------
Project: OpenSAR Insight / AI4SAR
Customer: ESA
---------------------------------------------------------------------
File: test_rc_dataloader.py

Unit tests for rc_dataloader.py

Covers:
- tensor_from_npy(): loads complex/(H,W,2)/(2,H,W) -> (2,H,W) float32; raises on unsupported shapes
- ensure_channel_first(): enforces (2,H,W) and sanitizes NaN/Inf to 0.0
- magnitude(): sqrt(I^2+Q^2) correctness on a toy case
- Stride helpers: ceil_to_multiple() rounding; check_stride_ready() messaging
- RCDataset:
  • length and item keys {rc, mask, meta}
  • meta fields (size_pre_transform, size_post_transform)
  • transform pipeline with no-op config (norm off, augs disabled)
- collate_pad_validmask():
  • pads to batch maxima with stride enforcement
  • valid_mask correctness (sum equals original area; padded zeros)
  • bypass mode returns raw samples unchanged
- build_transforms_from_cfg():
  • respects config toggles (no-op path)
  • includes expected classes in Compose
- Normalization (real modules):
  • NormalizeRCTransform presence
  • effect: L2(post_norm, raw) > 0 and finite output
- Each augmentation (real modules), parameterized:
  • presence of the specific aug class + normalization in Compose
  • effect: L2(post_aug, post_norm) > 0 and finite output
- CLI wiring: parse_args populates expected fields

Run:
    pytest -q test_rc_dataloader.py
---------------------------------------------------------------------
History:
    - 2025-10-02:
        First version of unit tests for range-compressed dataloader script.

---------------------------------------------------------------------
Author: Hamideh Kerdegari (HAMK)
E-mail: hkerdegari@indracompany.com
Creation Date: 2025-10-02

© Copyright INDRA DEIMOS, 2025. All rights reserved.
---------------------------------------------------------------------
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pytest
import torch
import rc_dataloader as rc
import data_preprocessing.augmentation.sar_rc_augmentation as aug_mod
import data_preprocessing.normalization.sar_normalization as norm_mod


# ========================= Helpers =========================

def _make_rc_complex(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    real = rng.normal(0, 0.05, size=(h, w)).astype(np.float32)
    imag = rng.normal(0, 0.05, size=(h, w)).astype(np.float32)
    return real + 1j * imag  # complex (H,W)


def _make_rc_hw2(h: int, w: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.normal(0, 0.05, size=(h, w, 2)).astype(np.float32)  # (H,W,2)
    return arr


def _make_rc_2hw(h: int, w: int, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.normal(0, 0.05, size=(2, h, w)).astype(np.float32)  # (2,H,W)
    return arr


def _write_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr, allow_pickle=False)


def _cfg_no_ops() -> dict:
    """Normalization OFF, all augs disabled (deterministic unit tests)."""
    return {
        "device": "cpu",
        "norm": {"method": "none"},
        "augs": {
            "amp_phase": {"enable": False},
            "dropout": {"enable": False},
            "trim": {"enable": False},
            "defocus": {"enable": False},
            "subsample": {"enable": False},
        },
    }


def _cfg_norm_only() -> dict:
    """Normalization enabled, all augmentations disabled."""
    return {
        "device": "cpu",
        "norm": {"method": "rms-shared"},
        "augs": {
            "amp_phase": {"enable": False},
            "dropout": {"enable": False},
            "trim": {"enable": False},
            "defocus": {"enable": False},
            "subsample": {"enable": False},
        },
    }


def _cfg_one_aug(aug_key: str) -> dict:
    """
    Normalization enabled + exactly one augmentation enabled with stable params. It is based on augmentation values for vessel data.
    """
    cfg = _cfg_norm_only()
    augs = cfg["augs"]
    for k in augs:
        augs[k]["enable"] = False
    augs[aug_key]["enable"] = True

    # Add stable/default params to avoid degeneracy
    if aug_key == "amp_phase":
        augs[aug_key].update({
            "amp_scale_range": [0.95, 1.05],
            "max_phase_shift": 0.05,
            "per_pixel": True,
        })
    elif aug_key == "dropout":
        augs[aug_key].update({
            "bands_range": [[0.45, 0.55]],
            "bands_azimuth": [],
            "soft": True,
            "edge_taper": 0.10,
        })
    elif aug_key == "trim":
        augs[aug_key].update({
            "trim_frac_range": 0.95,
            "trim_frac_azimuth": 0.95,
            "window": "hann",
        })
    elif aug_key == "defocus":
        augs[aug_key].update({"kappa": 6.0e-4})
    elif aug_key == "subsample":
        augs[aug_key].update({
            "az_factor": 2,
            "rg_factor": 1,
            "down_mode": "avg",
            "up_mode": "bilinear",
        })
    return cfg


def _l2_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).pow(2).mean().sqrt().item())



@pytest.fixture
def tmp_rc_dir(tmp_path: Path) -> Path:
    """Create a temp directory with three RC files in different formats (and sizes)."""
    d = tmp_path / "rc"
    _write_npy(d / "a_complex.npy", _make_rc_complex(63, 51))
    _write_npy(d / "b_hw2.npy", _make_rc_hw2(64, 50))
    _write_npy(d / "c_2hw.npy", _make_rc_2hw(62, 52))
    return d


@pytest.fixture
def seed_all():
    """Deterministic RNG seeding for any stochastic augs."""
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# ========================= Tests: low-level utils =========================

def test_tensor_from_npy_valid_variants(tmp_path: Path) -> None:
    p_complex = tmp_path / "cplx.npy"
    p_hw2 = tmp_path / "hw2.npy"
    p_2hw = tmp_path / "2hw.npy"

    _write_npy(p_complex, _make_rc_complex(10, 8))
    _write_npy(p_hw2, _make_rc_hw2(11, 7))
    _write_npy(p_2hw, _make_rc_2hw(12, 9))

    t1 = rc.tensor_from_npy(p_complex)
    t2 = rc.tensor_from_npy(p_hw2)
    t3 = rc.tensor_from_npy(p_2hw)

    for t in (t1, t2, t3):
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32
        assert t.ndim == 3 and t.shape[0] == 2  # (2,H,W)


def test_tensor_from_npy_unsupported_shape_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.npy"
    _write_npy(p, np.zeros((3, 3, 3), dtype=np.float32))  # neither complex/(H,W,2)/(2,H,W)
    with pytest.raises(ValueError):
        _ = rc.tensor_from_npy(p)


def test_ensure_channel_first_variants_and_sanitize() -> None:
    # complex ndarray
    arr_c = _make_rc_complex(5, 6)
    t_c = rc.ensure_channel_first(arr_c)
    assert t_c.shape == (2, 5, 6)

    # (H,W,2) ndarray
    arr_hw2 = _make_rc_hw2(4, 7)
    t_hw2 = rc.ensure_channel_first(arr_hw2)
    assert t_hw2.shape == (2, 4, 7)

    # (2,H,W) tensor
    t_in = torch.from_numpy(_make_rc_2hw(3, 8))
    t_2hw = rc.ensure_channel_first(t_in)
    assert t_2hw.shape == (2, 3, 8)

    # NaN/Inf sanitization
    bad = np.zeros((2, 3, 3), dtype=np.float32)
    bad[0, 0, 0] = np.nan
    bad[1, 1, 1] = np.inf
    out = rc.ensure_channel_first(bad)
    assert torch.isfinite(out).all()
    assert out[0, 0, 0] == 0.0 and out[1, 1, 1] == 0.0


def test_magnitude_matches_definition() -> None:
    x = torch.tensor(
        [[[3.0, 4.0], [0.0, 0.0]],
         [[4.0, 3.0], [0.0, 0.0]]],
        dtype=torch.float32
    )  # (2,2,2)
    mag = rc.magnitude(x, eps=0.0)
    expect = torch.tensor([[5.0, 5.0], [0.0, 0.0]])
    assert torch.allclose(mag, expect, atol=1e-6)


def test_stride_helpers() -> None:
    assert rc.ceil_to_multiple(33, 32) == 64
    assert rc.ceil_to_multiple(64, 32) == 64
    assert rc.ceil_to_multiple(0, 32) == 0
    assert "H%32" in rc.check_stride_ready(64, 33, 32)


# ========================= Tests: dataset & dataloader =========================

def test_dataset_len_and_keys(tmp_rc_dir: Path) -> None:
    ds = rc.RCDataset(data_dir=tmp_rc_dir, use_masks=False, mask_resolver=None, transform=None)
    assert len(ds) == 3
    s0 = ds[0]
    assert set(s0.keys()) == {"rc", "mask", "meta"}
    assert isinstance(s0["rc"], torch.Tensor) and s0["rc"].shape[0] == 2
    assert s0["mask"] is None
    assert "path" in s0["meta"] and "basename" in s0["meta"]


def test_build_transforms_noop_and_dataset_post_meta(tmp_rc_dir: Path) -> None:
    cfg = _cfg_no_ops()
    tfm = rc.build_transforms_from_cfg(cfg)  # should effectively be no-op
    ds = rc.RCDataset(data_dir=tmp_rc_dir, use_masks=False, mask_resolver=None, transform=tfm)
    s = ds[1]
    assert "size_pre_transform" in s["meta"] and "size_post_transform" in s["meta"]
    assert tuple(s["meta"]["size_pre_transform"]) == tuple(s["meta"]["size_post_transform"])
    assert isinstance(s["rc"], torch.Tensor) and s["rc"].shape[0] == 2


def test_collate_padding_stride_and_valid_mask(tmp_rc_dir: Path) -> None:
    cfg = _cfg_no_ops()
    transforms = rc.build_transforms_from_cfg(cfg)
    ds = rc.RCDataset(tmp_rc_dir, transform=transforms)
    samples = [ds[i] for i in range(len(ds))]

    stride = 32
    batch = rc.collate_pad_validmask(
        samples,
        enforce_stride_multiple=True,
        stride_multiple=stride,
        bypass_padding=False,
        device=torch.device("cpu"),
    )
    assert isinstance(batch, rc.PaddedBatch)
    B, C, H_pad, W_pad = batch.rc.shape
    assert (B, C) == (3, 2)
    assert H_pad % stride == 0 and W_pad % stride == 0

    for i, meta in enumerate(batch.meta):
        H = int(meta["H"])
        W = int(meta["W"])
        vm = batch.valid_mask[i, 0]  # (H_pad, W_pad)
        assert int(vm.sum().item()) == H * W
        assert vm[H:].sum().item() == 0 or H == H_pad
        assert vm[:, W:].sum().item() == 0 or W == W_pad


def test_collate_bypass_returns_samples(tmp_rc_dir: Path) -> None:
    ds = rc.RCDataset(tmp_rc_dir, transform=None)
    samples = [ds[0], ds[1]]
    out = rc.collate_pad_validmask(samples, bypass_padding=True)
    assert isinstance(out, list) and len(out) == 2
    assert set(out[0].keys()) == {"rc", "mask", "meta"}


def test_normalization_present_and_changes_tensor(tmp_rc_dir: Path, seed_all) -> None:
    # Build normalization-only pipeline
    cfg_norm = _cfg_norm_only()
    tfm_norm = rc.build_transforms_from_cfg(cfg_norm)

    # Presence check
    assert any(isinstance(t, norm_mod.NormalizeRCTransform) for t in getattr(tfm_norm, "transforms", [tfm_norm]))

    # Compare raw vs post-norm
    raw_path = list(sorted(tmp_rc_dir.glob("*.npy")))[0]
    pre = rc.ensure_channel_first(rc.tensor_from_npy(raw_path)).clone()

    ds_norm = rc.RCDataset(data_dir=tmp_rc_dir, transform=tfm_norm)
    post_norm = rc.ensure_channel_first(ds_norm[0]["rc"])
    diff = _l2_diff(post_norm, pre)

    assert diff > 1e-6, f"Normalization produced no observable change (L2={diff:.2e})"
    assert torch.isfinite(post_norm).all()


@pytest.mark.parametrize("aug_key,cls", [
    ("amp_phase", aug_mod.RCAmplitudePhasePerturb),
    ("dropout",   aug_mod.RCNarrowbandSpectralDropout),
    ("trim",      aug_mod.RCBandwidthTrim),
    ("defocus",   aug_mod.RCAzimuthDefocus),
    ("subsample", aug_mod.RCSubsampleAndInterpolate),
])
def test_each_augmentation_present_and_changes_tensor(tmp_rc_dir: Path, seed_all, aug_key: str, cls) -> None:
    # Pre: raw tensor
    raw_path = list(sorted(tmp_rc_dir.glob("*.npy")))[0]
    pre = rc.ensure_channel_first(rc.tensor_from_npy(raw_path)).clone()

    # Post_norm: normalization only
    cfg_norm = _cfg_norm_only()
    tfm_norm = rc.build_transforms_from_cfg(cfg_norm)
    ds_norm = rc.RCDataset(data_dir=tmp_rc_dir, transform=tfm_norm)
    post_norm = rc.ensure_channel_first(ds_norm[0]["rc"]).clone()

    # Post_aug: normalization + exactly one augmentation
    cfg_aug = _cfg_one_aug(aug_key)
    tfm_aug = rc.build_transforms_from_cfg(cfg_aug)

    # Presence check (both norm and the specific aug should be present)
    names: List[str] = [t.__class__.__name__ for t in getattr(tfm_aug, "transforms", [tfm_aug])]
    assert any(isinstance(t, norm_mod.NormalizeRCTransform) for t in tfm_aug.transforms), "Normalization missing"
    assert any(isinstance(t, cls) for t in tfm_aug.transforms), f"{aug_key} transform missing; got {names}"

    ds_aug = rc.RCDataset(data_dir=tmp_rc_dir, transform=tfm_aug)
    post_aug = rc.ensure_channel_first(ds_aug[0]["rc"])

    # The augmentation should introduce a further change vs post_norm
    aug_delta = _l2_diff(post_aug, post_norm)
    assert aug_delta > 1e-6, f"{aug_key} produced no observable change over normalization (L2={aug_delta:.2e})"

    # Also sanity: overall differs from raw
    overall = _l2_diff(post_aug, pre)
    assert overall > 1e-6
    assert torch.isfinite(post_aug).all()


def test_cli_parse_args(monkeypatch) -> None:
    fake_argv = [
        "rc_dataloader.py",
        "--data_dir", "/tmp/nowhere",
        "--config", "/tmp/nowhere.yaml",
        "--batch_size", "2",
        "--num_workers", "0",
        "--enforce_stride_multiple", "true",
        "--bypass_padding", "false",
        "--stride_multiple", "32",
        "--use_masks", "false",
        "--max_batches", "1",
        "--device", "cpu",
    ]
    monkeypatch.setattr("sys.argv", fake_argv)
    ns = rc.parse_args()
    assert ns.data_dir == "/tmp/nowhere"
    assert ns.config.endswith(".yaml")
    assert ns.batch_size == 2
    assert ns.device == "cpu"
