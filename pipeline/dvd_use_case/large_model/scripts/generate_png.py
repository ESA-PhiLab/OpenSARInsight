# type: ignore
"""
---------------------------------------------------------------------
Project: OpenSAR Insight
---------------------------------------------------------------------
generate_png.py

Tool: Convert RC .npy patches to 3-channel PNG images
    All channels: intensity (clipped 2-98 percentile, normalised)
    Output: RGB image where R=G=B=intensity

Author: Abdulhameed Yunusa (ABHY)
E-mail: ayunusa@indracompany.com
Creation Date: 2026-02-26

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
from utilities.read_yaml import read_yaml


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Scale [0, 1] float array to uint8 [0, 255]."""
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def generate_pngs(dataset_root: Path, png_root: Path):
    """
    Convert all .npy RC patches to 3-channel PNGs:
        Ch1, Ch2, Ch3: intensity — clipped to 2-98 percentile then normalised to [0,1]
        Output is a greyscale image stacked into RGB (R=G=B=intensity)

    Saved as DB_OPENSAR_<POL>_VD_<PATCH>.png
    """
    logging.basicConfig(level=logging.INFO, filename="png.log", filemode="w")

    files = [
        f for split in ["train", "val", "test"]
        for f in (dataset_root / split / "range_compressed_rescaled").glob("*.npy")
    ]

    for f in tqdm(files, desc="Converting"):
        try:
            arr  = np.load(f).astype(np.complex64)


            # ── Intensity — clip 2-98 percentile, normalise ──────────────
            intensity = np.abs(arr) ** 2
            p2        = np.percentile(intensity, 2)
            p98       = np.percentile(intensity, 98)
            clipped   = np.clip(intensity, p2, p98)
            norm_int  = (clipped - p2) / (p98 - p2 + 1e-12)   # [0, 1]
            ch       = to_uint8(norm_int)

            # ── Stack intensity 3 times for RGB (R=G=B=intensity) ───────
            rgb    = np.stack([ch, ch, ch], axis=-1)

            pol    = "VV" if "-vv-" in f.stem else "VH"
            patch  = f.stem.split("VD_")[-1].split("_")[0]

            # f is at dataset_root/split/range_compressed_rescaled/file.npy
            split_name = f.parts[-3]  # extract split name (train/val/test)
            out_dir = png_root / split_name
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / f"DB_OPENSAR_{pol}_VD_{patch}.png"
            Image.fromarray(rgb).save(str(out_path))

        except Exception as e:
            logging.error(f"{f.name}: {e}")


if __name__ == "__main__":
    config = read_yaml(Path("config.yaml"))
    dataset_root = Path(config["datapaths"]["dataset_root"])
    png          = Path(config["datapaths"]["png_rc_patches_path"])
    generate_pngs(dataset_root, png)