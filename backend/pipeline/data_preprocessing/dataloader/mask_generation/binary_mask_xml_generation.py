"""
Read an OpenSAR-style XML file and save a binary PNG mask (no .npy needed).

Parses polygons from:
- ProcessingData/List_of_FloodEvents/Event/Polygon/(Scene_Sample, Scene_Line)
- ProcessingData/List_of_WaterBodys/Body/Polygon/(Scene_Sample, Scene_Line)

Canvas:
- Derived from ProcessingData/Corner_Coord/(Scene_Sample, Scene_Line).
  If extents look like 1..512 inclusive, output mask is 512x512.
  Otherwise, width/height are computed from the inclusive span.

Output:
- Single-channel PNG (0 background, 255 foreground) containing the union of the chosen layers.

Usage:
    python xml_to_mask.py \
        --xml /path/labels.xml \
        --out /path/mask.png \
        --layers both

Layers:
    --layers flood | water | both    (default: both)
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET


# ----------------------- Parsing helpers -----------------------

def _parse_float_series(text: Optional[str]) -> List[float]:
    """Parse whitespace/comma-separated numbers; ignore non-finite (e.g., 'NaN')."""
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


def parse_scene_extents(root: ET.Element) -> Tuple[float, float, float, float]:
    """
    Read Corner_Coord Scene_Sample/Scene_Line and return (x_min, x_max, y_min, y_max).
    Falls back to (1, 512, 1, 512) if unavailable.
    """
    cc = root.find(".//ProcessingData/Corner_Coord")
    if cc is None:
        logging.warning("Corner_Coord not found; defaulting to x[1..512], y[1..512].")
        return 1.0, 512.0, 1.0, 512.0

    xs = _parse_float_series(cc.findtext("Scene_Sample"))
    ys = _parse_float_series(cc.findtext("Scene_Line"))
    if len(xs) < 2 or len(ys) < 2:
        logging.warning("Corner_Coord incomplete; defaulting to x[1..512], y[1..512].")
        return 1.0, 512.0, 1.0, 512.0

    return min(xs), max(xs), min(ys), max(ys)


def _extract_polygons_at(root: ET.Element, path: str) -> List[List[Tuple[float, float]]]:
    """
    Extract polygons by pairing Scene_Sample & Scene_Line arrays under a given path.
    Paths:
      - ".//ProcessingData/List_of_FloodEvents/Event/Polygon"
      - ".//ProcessingData/List_of_WaterBodys/Body/Polygon"
    """
    polys: List[List[Tuple[float, float]]] = []
    for poly in root.findall(path):
        xs = _parse_float_series(poly.findtext("Scene_Sample"))
        ys = _parse_float_series(poly.findtext("Scene_Line"))
        n = min(len(xs), len(ys))
        if n >= 3:
            polys.append(list(zip(xs[:n], ys[:n])))
    return polys


def extract_flood_polygons(root: ET.Element) -> List[List[Tuple[float, float]]]:
    return _extract_polygons_at(root, ".//ProcessingData/List_of_FloodEvents/Event/Polygon")


def extract_water_polygons(root: ET.Element) -> List[List[Tuple[float, float]]]:
    return _extract_polygons_at(root, ".//ProcessingData/List_of_WaterBodys/Body/Polygon")


# ----------------------- Coordinate mapping -----------------------

def determine_canvas_size(x_min: float, x_max: float, y_min: float, y_max: float) -> Tuple[int, int]:
    """
    Decide output (W, H). If extents closely match 1..512 inclusive, use 512x512.
    Else compute inclusive span: round(x_max - x_min) + 1, same for y.
    """
    if abs(x_min - 1.0) < 1e-6 and abs(x_max - 512.0) < 1e-6 \
       and abs(y_min - 1.0) < 1e-6 and abs(y_max - 512.0) < 1e-6:
        return 512, 512

    w = int(round(x_max - x_min)) + 1
    h = int(round(y_max - y_min)) + 1
    w = max(w, 8)
    h = max(h, 8)
    return w, h


def edge_map_xy(x_scene: float, y_scene: float,
                x_min: float, x_max: float, y_min: float, y_max: float,
                w_img: int, h_img: int) -> Tuple[float, float]:
    """
    Edge-aligned mapping: x_min->0, x_max->W-1; y_min->0, y_max->H-1. Clips to bounds.
    """
    if not (x_max > x_min and y_max > y_min):
        raise ValueError(f"Degenerate scene extents: x[{x_min},{x_max}], y[{y_min},{y_max}]")

    x_norm = (x_scene - x_min) / (x_max - x_min)
    y_norm = (y_scene - y_min) / (y_max - y_min)

    xi = float(np.clip(x_norm * (w_img - 1), 0.0, w_img - 1.0))
    yi = float(np.clip(y_norm * (h_img - 1), 0.0, h_img - 1.0))
    return xi, yi


def map_polygon_to_img(poly: List[Tuple[float, float]],
                       x_min: float, x_max: float, y_min: float, y_max: float,
                       w: int, h: int) -> List[Tuple[float, float]]:
    return [edge_map_xy(x, y, x_min, x_max, y_min, y_max, w, h) for x, y in poly]


# ----------------------- Rasterization -----------------------

def rasterize(polys_img: List[List[Tuple[float, float]]], h: int, w: int) -> Image.Image:
    """
    Fill polygons into a single-channel ('L') mask with 0 (bg) / 255 (fg).
    """
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask, "L")
    for poly in polys_img:
        if len(poly) >= 3:
            draw.polygon(poly, fill=255)
    return mask


# ----------------------- CLI -----------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OpenSAR XML to a binary PNG mask (no image needed).")
    parser.add_argument("--xml", required=True, help="Path to the XML labels")
    parser.add_argument("--out", required=True, help="Output PNG path (binary mask)")
    parser.add_argument("--layers", choices=["flood", "water", "both"], default="both",
                        help="Which layers to include in the mask (default: both)")
    parser.add_argument("--log", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format="%(levelname)s: %(message)s")

    # Parse XML
    root = ET.parse(args.xml).getroot()

    # Scene extents & target canvas size
    x_min, x_max, y_min, y_max = parse_scene_extents(root)
    w, h = determine_canvas_size(x_min, x_max, y_min, y_max)
    logging.info("Scene extents: x[%.3f..%.3f], y[%.3f..%.3f] -> canvas (W,H)=(%d,%d)",
                 x_min, x_max, y_min, y_max, w, h)

    # Extract polygons in scene coords
    flood_scene = extract_flood_polygons(root)
    water_scene = extract_water_polygons(root)
    logging.info("Found %d flood polygon(s), %d water polygon(s).", len(flood_scene), len(water_scene))

    # Map to image coords of (w,h)
    flood_img = [map_polygon_to_img(p, x_min, x_max, y_min, y_max, w, h) for p in flood_scene]
    water_img = [map_polygon_to_img(p, x_min, x_max, y_min, y_max, w, h) for p in water_scene]

    # Rasterize selected layers
    if args.layers == "flood":
        mask = rasterize(flood_img, h, w)
    elif args.layers == "water":
        mask = rasterize(water_img, h, w)
    else:  # both
        mask_f = rasterize(flood_img, h, w)
        mask_w = rasterize(water_img, h, w)
        # Union (logical OR) for 0/255 images: pixel-wise max
        mask = Image.eval(mask_f, lambda v: v)  # copy
        mask = Image.composite(mask_f, mask_w, Image.fromarray(np.maximum(np.array(mask_f), np.array(mask_w)).astype(np.uint8)))

        # Simpler & faster union:
        mask = Image.fromarray(np.maximum(np.array(mask_f), np.array(mask_w)).astype(np.uint8))

    # Save
    mask.save(args.out)
    logging.info("Saved binary mask: %s", args.out)


if __name__ == "__main__":
    main()


