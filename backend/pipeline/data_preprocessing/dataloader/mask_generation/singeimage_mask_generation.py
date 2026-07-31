"""
Create a binary PNG mask for ONE .npy image from an XML.

- Flood polygons:  ProcessingData/List_of_FloodEvents/Event/Polygon/(Scene_Sample, Scene_Line)
- Water polygons:  ProcessingData/List_of_WaterBodys/Body/Polygon/(Scene_Sample, Scene_Line)
- Canvas (tile-local): ProcessingData/Corner_Coord/(Scene_Sample, Scene_Line)  typically 1..512 (inclusive)
- Uses edge-aligned mapping (min->0, max->W-1/H-1) to avoid 1-based off-by-one.

Outputs:
- A binary PNG mask (0 background, 255 foreground) including layers per --layers flag.
- Optionally a separate water-only mask via --water-out.

Example:
    python make_mask_opensar_one.py \
        --npy /path/tile.npy \
        --xml /path/labels.xml \
        --out /path/mask.png \
        --layers both \
        --water-out /path/water_only.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET


# ----------------------- Utilities -----------------------

def infer_hw(arr: np.ndarray) -> Tuple[int, int]:
    """Infer (H, W) from (H,W), (C,H,W), or (H,W,C)."""
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim == 3:
        # Prefer last two as spatial if plausible
        if arr.shape[0] in (1, 2, 3, 4) and arr.shape[1] >= 8 and arr.shape[2] >= 8:
            return int(arr.shape[1]), int(arr.shape[2])  # (C, H, W)
        if arr.shape[2] in (1, 2, 3, 4) and arr.shape[0] >= 8 and arr.shape[1] >= 8:
            return int(arr.shape[0]), int(arr.shape[1])  # (H, W, C)
        # Fallback: largest two
        h, w = sorted(arr.shape)[-2:]
        logging.warning("Ambiguous array shape %s; using two largest dims as H=%d W=%d.", arr.shape, h, w)
        return int(h), int(w)
    raise ValueError(f"Unsupported array ndim={arr.ndim}; expected 2D or 3D.")


def _parse_float_series(text: Optional[str]) -> List[float]:
    """
    Parse a whitespace-separated series of numbers (possibly spanning lines),
    ignoring tokens that cannot be parsed (e.g., 'NaN' trailer).
    """
    if not text:
        return []
    out: List[float] = []
    for tok in text.replace(",", " ").split():
        try:
            val = float(tok)
            if np.isfinite(val):
                out.append(val)
        except Exception:
            continue
    return out


# ----------------------- Canvas (scene frame) -----------------------

def parse_scene_canvas_minmax(root: ET.Element) -> Optional[Tuple[float, float, float, float]]:
    """
    From Corner_Coord, read Scene_Sample and Scene_Line arrays and return
    (x_min, x_max, y_min, y_max). Typically 1..512 inclusive.
    """
    cc = root.find(".//ProcessingData/Corner_Coord")
    if cc is None:
        return None
    xs = _parse_float_series(cc.findtext("Scene_Sample"))
    ys = _parse_float_series(cc.findtext("Scene_Line"))
    if len(xs) < 2 or len(ys) < 2:
        return None
    return (min(xs), max(xs), min(ys), max(ys))


def edge_aligned_map_xy(
    x_scene: float, y_scene: float,
    x_min: float, x_max: float, y_min: float, y_max: float,
    w_img: int, h_img: int
) -> Tuple[float, float]:
    """
    Map scene coords to image pixel coords using edge-aligned mapping:
      x_min -> 0, x_max -> W-1;  y_min -> 0, y_max -> H-1
    """
    if not (x_max > x_min and y_max > y_min):
        raise ValueError(f"Degenerate scene extents: x[{x_min},{x_max}] y[{y_min},{y_max}]")
    x_norm = (x_scene - x_min) / (x_max - x_min)
    y_norm = (y_scene - y_min) / (y_max - y_min)
    x_img = x_norm * (w_img - 1)
    y_img = y_norm * (h_img - 1)
    # clip
    x_img = min(max(x_img, 0.0), w_img - 1.0)
    y_img = min(max(y_img, 0.0), h_img - 1.0)
    return x_img, y_img


def map_polygon(
    poly_xy: Iterable[Tuple[float, float]],
    x_min: float, x_max: float, y_min: float, y_max: float,
    w_img: int, h_img: int
) -> List[Tuple[float, float]]:
    return [edge_aligned_map_xy(x, y, x_min, x_max, y_min, y_max, w_img, h_img) for x, y in poly_xy]


# ----------------------- Shape extraction (schema-specific) -----------------------

def _extract_polygons_at(parent: ET.Element, poly_tag_path: str) -> List[List[Tuple[float, float]]]:
    """
    Extract polygons at a given path, pairing Scene_Sample & Scene_Line series.
    poly_tag_path examples:
      ".//ProcessingData/List_of_FloodEvents/Event/Polygon"
      ".//ProcessingData/List_of_WaterBodys/Body/Polygon"
    """
    polys: List[List[Tuple[float, float]]] = []
    for poly in parent.findall(poly_tag_path):
        xs = _parse_float_series(poly.findtext("Scene_Sample"))
        ys = _parse_float_series(poly.findtext("Scene_Line"))
        if not xs or not ys or len(xs) != len(ys):
            # try to trim to min length if mismatched but > 2
            n = min(len(xs), len(ys))
            xs, ys = xs[:n], ys[:n]
        if len(xs) >= 3:
            pts = list(zip(xs, ys))
            polys.append(pts)
    return polys


def extract_flood_polygons(root: ET.Element) -> List[List[Tuple[float, float]]]:
    """Flood polygons from ProcessingData/List_of_FloodEvents/Event/Polygon/Scene_*."""
    return _extract_polygons_at(root, ".//ProcessingData/List_of_FloodEvents/Event/Polygon")


def extract_water_polygons(root: ET.Element) -> List[List[Tuple[float, float]]]:
    """Water-body polygons from ProcessingData/List_of_WaterBodys/Body/Polygon/Scene_*."""
    return _extract_polygons_at(root, ".//ProcessingData/List_of_WaterBodys/Body/Polygon")


# ----------------------- Rasterization -----------------------

def rasterize(polys_img: List[List[Tuple[float, float]]], h_img: int, w_img: int) -> Image.Image:
    """Fill polygons into a single-channel ('L') mask with 0/255 values."""
    mask = Image.new("L", (w_img, h_img), 0)
    drw = ImageDraw.Draw(mask, "L")
    for poly in polys_img:
        if len(poly) >= 3:
            drw.polygon(poly, fill=255)
    return mask


def combine_union(masks: List[Image.Image]) -> Image.Image:
    """Union multiple 0/255 masks (logical OR)."""
    if not masks:
        raise ValueError("No masks to combine.")
    base = masks[0].copy()
    for m in masks[1:]:
        base = ImageChops.lighter(base, m)  # OR for 0/255 images
    return base


# ----------------------- Main -----------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Create a binary PNG mask from one .npy and its OpenSAR XML labels.")
    parser.add_argument("--npy", required=True, help="Path to the image .npy")
    parser.add_argument("--xml", required=True, help="Path to the XML labels")
    parser.add_argument("--out", required=True, help="Output PNG path (binary mask)")
    parser.add_argument("--layers", choices=["flood", "water", "both"], default="flood",
                        help="Which layers to include in the output binary mask (default: flood)")
    parser.add_argument("--water-out", default=None,
                        help="Optional: save a separate water-only mask PNG (0/255)")
    parser.add_argument("--flip-x", action="store_true", help="Flip mask horizontally before saving")
    parser.add_argument("--flip-y", action="store_true", help="Flip mask vertically before saving")
    parser.add_argument("--transpose", action="store_true", help="Transpose mask (swap H/W) before saving")
    parser.add_argument("--log", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format="%(levelname)s: %(message)s")

    # 1) Load image dims
    arr = np.load(args.npy, mmap_mode="r")
    h_img, w_img = infer_hw(arr)
    logging.info("Image size (H,W)=(%d,%d)", h_img, w_img)

    # 2) Parse XML
    root = ET.parse(args.xml).getroot()

    # 3) Scene canvas extents (tile-local)
    extents = parse_scene_canvas_minmax(root)
    if not extents:
        # Fallback to typical 1..512 inclusive
        logging.warning("No Corner_Coord/Scene_* extents found; defaulting to x[1,512], y[1,512].")
        x_min, x_max, y_min, y_max = 1.0, 512.0, 1.0, 512.0
    else:
        x_min, x_max, y_min, y_max = extents
    logging.info("Scene extents: x[%.3f..%.3f] y[%.3f..%.3f]", x_min, x_max, y_min, y_max)

    # 4) Extract polygons
    flood_polys_scene = extract_flood_polygons(root)
    water_polys_scene = extract_water_polygons(root)
    logging.info("Found %d flood polygon(s), %d water polygon(s).",
                 len(flood_polys_scene), len(water_polys_scene))

    # 5) Map to image frame
    flood_polys_img = [map_polygon(p, x_min, x_max, y_min, y_max, w_img, h_img) for p in flood_polys_scene]
    water_polys_img = [map_polygon(p, x_min, x_max, y_min, y_max, w_img, h_img) for p in water_polys_scene]

    # 6) Rasterize
    from PIL import ImageChops  # local import to avoid unused warning if not combining
    layers_to_include: List[Image.Image] = []
    water_mask_opt: Optional[Image.Image] = None

    if args.layers in ("flood", "both"):
        flood_mask = rasterize(flood_polys_img, h_img, w_img)
        layers_to_include.append(flood_mask)

    if args.layers in ("water", "both"):
        water_mask = rasterize(water_polys_img, h_img, w_img)
        layers_to_include.append(water_mask)
        water_mask_opt = water_mask

    if not layers_to_include:
        logging.warning("No layers selected or no polygons found; producing all-zero mask.")
        out_mask = Image.new("L", (w_img, h_img), 0)
    else:
        # Union all selected layers
        out_mask = layers_to_include[0]
        for m in layers_to_include[1:]:
            out_mask = ImageChops.lighter(out_mask, m)  # OR for 0/255 masks

    # 7) Optional orientation fixes (if your .npy was flipped/rotated)
    if args.flip_x:
        out_mask = out_mask.transpose(Image.FLIP_LEFT_RIGHT)
        if water_mask_opt is not None:
            water_mask_opt = water_mask_opt.transpose(Image.FLIP_LEFT_RIGHT)
    if args.flip_y:
        out_mask = out_mask.transpose(Image.FLIP_TOP_BOTTOM)
        if water_mask_opt is not None:
            water_mask_opt = water_mask_opt.transpose(Image.FLIP_TOP_BOTTOM)
    if args.transpose:
        out_mask = out_mask.transpose(Image.TRANSPOSE)
        if water_mask_opt is not None:
            water_mask_opt = water_mask_opt.transpose(Image.TRANSPOSE)

    # 8) Save outputs
    out_mask.save(args.out)
    logging.info("Saved mask: %s", args.out)

    if args.water_out:
        (water_mask_opt or Image.new("L", (w_img, h_img), 0)).save(args.water_out)
        logging.info("Saved water-only mask: %s", args.water_out)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("Failed: %s", e)
        sys.exit(1)
