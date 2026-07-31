"""
Batch-create binary PNG masks from XML labels for all .npy tiles
under the 'train', 'val', and 'test' folders.

Key points
----------
- One XML per .npy. XMLs may live in different folders, possibly
  separate per split (train/val/test). We search all provided XML roots.
- Matching is by trailing numeric ID:  <...>_41512.npy  <->  DB_OPENSAR_FD_10_41512.xml

Output layout
-------------
<DATA_ROOT>/
  train/
    mask/
      <tile_stem>_mask.png
  val/
    mask/
      <tile_stem>_mask.png
  test/
    mask/
      <tile_stem>_mask.png

Usage examples
--------------
# XML roots per split:
python mask_generation.py \
  --data-root /path/to/flood/dataset \
  --xml-root-train /path/to/flood/new_kurosiwo_validation_sample_split/train/labels/xml_labelling \
  --xml-root-val   /path/to/flood/new_kurosiwo_validation_sample_split/val/labels/xml_labelling \
  --xml-root-test  /path/to/flood/new_kurosiwo_validation_sample_split/test/labels/xml_labelling \
  --layers both
  --workers 8 

Notes
-----
- Edge-aligned mapping: x_min->0, x_max->W-1; y_min->0, y_max->H-1.
- If no polygons found, emits a valid all-zero mask (0 background).
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageChops
import xml.etree.ElementTree as ET


# ----------------------- Core utilities -----------------------

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
    """Parse a whitespace-separated series of numbers, ignoring non-finite tokens."""
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
    """Edge-aligned mapping: x_min->0, x_max->W-1; y_min->0, y_max->H-1, with clipping."""
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


def rasterize(polys_img: List[List[Tuple[float, float]]], h_img: int, w_img: int) -> Image.Image:
    """Fill polygons into a single-channel ('L') mask with 0/255 values."""
    mask = Image.new("L", (w_img, h_img), 0)
    drw = ImageDraw.Draw(mask, "L")
    for poly in polys_img:
        if len(poly) >= 3:
            drw.polygon(poly, fill=255)
    return mask


# ----------------------- XML indexing & matching -----------------------

ID_XML_RE = re.compile(r"_(\d+)\.xml$", re.IGNORECASE)
ID_NPY_RE = re.compile(r"_(\d+)(?:_[A-Za-z0-9-]+)?\.npy$", re.IGNORECASE)


@dataclass(frozen=True)
class XmlRoot:
    path: Path
    split_hint: Optional[str]  # 'train' | 'val' | 'test' | None


def build_xml_index(xml_roots: List[XmlRoot]) -> Dict[str, List[Path]]:
    """
    Scan all provided XML roots and build: id_str -> [xml_paths...].
    """
    index: Dict[str, List[Path]] = {}
    seen_dirs: List[Path] = []
    for xr in xml_roots:
        if xr.path in seen_dirs:
            continue
        seen_dirs.append(xr.path)
        if not xr.path.exists():
            logging.warning("XML root does not exist: %s", xr.path)
            continue
        for p in xr.path.rglob("*.xml"):
            m = ID_XML_RE.search(p.name)
            if not m:
                continue
            id_str = m.group(1)
            index.setdefault(id_str, []).append(p)
    logging.info("Indexed %d unique tile IDs across %d XML roots.", len(index), len(xml_roots))
    return index


def detect_split_from_path(p: Path) -> Optional[str]:
    """
    Infer split from path parts: 'train', 'val', or 'test' (case-insensitive).
    Returns None if no split found.
    """
    parts = [s.lower() for s in p.parts]
    for s in ("train", "val", "test"):
        if s in parts:
            return s
    return None


def prefer_by_split(candidates: List[Path], desired_split: Optional[str]) -> Path:
    """
    If desired_split is given, prefer candidates whose path contains that split token.
    Otherwise or if none match, return lexicographically smallest path for determinism.
    """
    if desired_split:
        filtered = [c for c in candidates if desired_split in "/".join([q.lower() for q in c.parts])]
        if filtered:
            return sorted(filtered)[0]
    return sorted(candidates)[0]


# ----------------------- Batch processing -----------------------

@dataclass
class TileJob:
    npy_path: Path
    split_dir: Path
    out_path: Path
    xml_path: Path
    layers: str
    flip_x: bool
    flip_y: bool
    transpose: bool
    force: bool


def process_one_tile(job: TileJob) -> Tuple[Path, bool, Optional[str]]:
    """Process a single tile. Returns (npy_path, success, error_message_if_any)."""
    try:
        if job.out_path.exists() and not job.force:
            return (job.npy_path, True, None)

        # 1) Load dims
        arr = np.load(job.npy_path, mmap_mode="r")
        h_img, w_img = infer_hw(arr)

        # ---- ZERO-SIZE GUARD (skip tiles with H==0 or W==0) ----
        if h_img <= 0 or w_img <= 0:
            logging.warning("SKIPPED zero-sized image (H=%s, W=%s): %s", h_img, w_img, job.npy_path)
            # Treat as success so pipeline doesn't count as failure.
            return (job.npy_path, True, None)
        # --------------------------------------------------------

        # 2) Parse XML
        root = ET.parse(job.xml_path).getroot()

        # 3) Scene canvas extents
        extents = parse_scene_canvas_minmax(root)
        if not extents:
            x_min, x_max, y_min, y_max = 1.0, 512.0, 1.0, 512.0
        else:
            x_min, x_max, y_min, y_max = extents

        # 4) Extract polygons
        flood_polys_scene = extract_flood_polygons(root)
        water_polys_scene = extract_water_polygons(root)

        # 5) Map to image (edge-aligned)
        flood_polys_img = [
            [edge_aligned_map_xy(x, y, x_min, x_max, y_min, y_max, w_img, h_img) for (x, y) in p]
            for p in flood_polys_scene
        ]
        water_polys_img = [
            [edge_aligned_map_xy(x, y, x_min, x_max, y_min, y_max, w_img, h_img) for (x, y) in p]
            for p in water_polys_scene
        ]

        # 6) Rasterize selected layers (0/255)
        layers_to_include: List[Image.Image] = []
        if job.layers in ("flood", "both"):
            layers_to_include.append(rasterize(flood_polys_img, h_img, w_img))
        if job.layers in ("water", "both"):
            layers_to_include.append(rasterize(water_polys_img, h_img, w_img))

        if not layers_to_include:
            out_mask = Image.new("L", (w_img, h_img), 0)
        else:
            out_mask = layers_to_include[0]
            for m in layers_to_include[1:]:
                out_mask = ImageChops.lighter(out_mask, m)

        # 7) Optional orientation adjustments
        if job.flip_x:
            out_mask = out_mask.transpose(Image.FLIP_LEFT_RIGHT)
        if job.flip_y:
            out_mask = out_mask.transpose(Image.FLIP_TOP_BOTTOM)
        if job.transpose:
            out_mask = out_mask.transpose(Image.TRANSPOSE)

        # 8) Save
        job.out_path.parent.mkdir(parents=True, exist_ok=True)
        out_mask.save(job.out_path)

        return (job.npy_path, True, None)

    except Exception as e:
        return (job.npy_path, False, str(e))


def discover_npy_files(split_dir: Path, recursive: bool) -> List[Path]:
    if not split_dir.exists():
        return []
    pattern = "**/*.npy" if recursive else "*.npy"
    return sorted(split_dir.glob(pattern))


def make_out_path(npy_path: Path, split_dir: Path) -> Path:
    """
    Write masks under a flat <split_dir>/mask/ folder:
      <split_dir>/mask/<stem>_mask.png
    """
    stem = npy_path.stem
    mask_dir = split_dir / "mask"
    return mask_dir / f"{stem}_mask.png"


# ----------------------- CLI & driver -----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-create binary PNG masks for all .npy tiles in train/val/test.")

    # Data & splits
    p.add_argument("--data-root", required=True, type=Path,
                   help="Root directory containing train/, val/, and test/ subfolders with .npy files.")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   help="Which split folders to process (default: train val test).")
    p.add_argument("--recursive", action="store_true",
                   help="Recurse into split subdirectories to find .npy files.")

    # XML roots (per-split + extra buckets)
    p.add_argument("--xml-root-train", type=Path, default=None, help="XML root for train split.")
    p.add_argument("--xml-root-val",   type=Path, default=None, help="XML root for val split.")
    p.add_argument("--xml-root-test",  type=Path, default=None, help="XML root for test split.")
    p.add_argument("--xml-roots", nargs="*", type=Path, default=[],
                   help="Additional XML roots (searched for any split).")

    # Behavior
    p.add_argument("--layers", choices=["flood", "water", "both"], default="flood",
                   help="Layers included in output mask (default: flood).")
    p.add_argument("--flip-x", action="store_true", help="Flip mask horizontally before saving.")
    p.add_argument("--flip-y", action="store_true", help="Flip mask vertically before saving.")
    p.add_argument("--transpose", action="store_true", help="Transpose mask (swap H/W) before saving.")
    p.add_argument("--workers", type=int, default=0, help="Parallel workers (0/1 = single-threaded).")
    p.add_argument("--force", action="store_true", help="Overwrite existing masks.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip tiles whose masks already exist (ignored if --force).")
    p.add_argument("--log", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(levelname)s: %(message)s"
    )

    data_root: Path = args.data_root
    splits = [s.lower() for s in args.splits]

    # Build XML roots list with split hints
    xml_roots: List[XmlRoot] = []
    if args.xml_root_train:
        xml_roots.append(XmlRoot(args.xml_root_train, "train"))
    if args.xml_root_val:
        xml_roots.append(XmlRoot(args.xml_root_val, "val"))
    if args.xml_root_test:
        xml_roots.append(XmlRoot(args.xml_root_test, "test"))
    for extra in args.xml_roots:
        xml_roots.append(XmlRoot(extra, None))

    if not xml_roots:
        logging.error("No XML roots provided. Use --xml-root-train/val/test and/or --xml-roots.")
        sys.exit(2)

    # Build ID -> [xml_paths] index
    id_to_xmls = build_xml_index(xml_roots)
    if not id_to_xmls:
        logging.error("No XML files indexed from the provided roots.")
        sys.exit(2)

    # Build jobs
    jobs: List[TileJob] = []
    total_tiles = 0

    for split in splits:
        split_dir = data_root / split
        npy_files = discover_npy_files(split_dir, recursive=args.recursive)
        if not npy_files:
            logging.warning("No .npy files found under split '%s' (%s)", split, split_dir)

        for npy_path in npy_files:
            total_tiles += 1

            # Output path (flat mask/ inside split)
            out_path = make_out_path(npy_path, split_dir)
            if out_path.exists() and args.skip_existing and not args.force:
                continue

            # Extract numeric ID from npy filename
            m = ID_NPY_RE.search(npy_path.name)
            if not m:
                logging.error("Cannot extract ID from npy filename: %s", npy_path)
                continue
            id_str = m.group(1)

            candidates = id_to_xmls.get(id_str, [])
            if not candidates:
                logging.error("No XML found for ID %s (npy: %s). Checked all provided XML roots.", id_str, npy_path)
                continue

            # Prefer XML from same split (if any), else choose deterministically
            xml_path = prefer_by_split(candidates, split)

            jobs.append(TileJob(
                npy_path=npy_path,
                split_dir=split_dir,
                out_path=out_path,
                xml_path=xml_path,
                layers=args.layers,
                flip_x=bool(args.flip_x),
                flip_y=bool(args.flip_y),
                transpose=bool(args.transpose),
                force=bool(args.force)
            ))

    logging.info("Discovered %d tiles total; %d jobs to run (after skip/force).", total_tiles, len(jobs))

    # Execute
    n_ok = 0
    n_fail = 0
    failures: List[Tuple[Path, str]] = []

    if args.workers and args.workers > 1:
        with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for npy_path, ok, err in ex.map(process_one_tile, jobs):
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
                    failures.append((npy_path, err or "Unknown error"))
    else:
        for job in jobs:
            npy_path, ok, err = process_one_tile(job)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                failures.append((npy_path, err or "Unknown error"))

    logging.info("Done. Success (incl. skips): %d, Failed: %d.", n_ok, n_fail)
    if failures:
        logging.error("Some tiles failed:")
        for p, msg in failures[:50]:
            logging.error(" - %s :: %s", p, msg)
        if len(failures) > 50:
            logging.error(" ... and %d more.", len(failures) - 50)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("Failed: %s", e)
        sys.exit(1)


