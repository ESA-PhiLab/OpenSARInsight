"""
Flood dataset train/val/test splitter

Output layout:
split/
  ├─ train/
  │    ├─ labels/xml_labelling/*.xml
  │    ├─ patches/*
  │    └─ raw/*.dat
  ├─ val/
  │    ├─ labels/xml_labelling/*.xml
  │    ├─ patches/*
  │    └─ raw/*.dat
  └─ test/
       ├─ labels/xml_labelling/*.xml
       ├─ patches/*
       └─ raw/*.dat

How to run this code: python split_flood.py

Scene grouping is determined from normalized <SARProduct> in the XMLs.
"""

from __future__ import annotations
import concurrent.futures as cf
import logging
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Tuple
import xml.etree.ElementTree as ET
from dataset_generation_scripts.utils import get_config
cfg = get_config("DATA_PREPROC_PATH")

# ====================== PATHS ======================
LABEL_DIR   = Path(cfg["split_flood"]["label_dir"])
PATCHES_DIR = Path(cfg["split_flood"]["patches_dir"])
RAW_DIR     = Path(cfg["split_flood"]["raw_dir"])
OUTPUT_ROOT = Path(cfg["split_flood"]["output_root"])

TRAIN_RATIO: float = 0.70
VAL_RATIO:   float = 0.15
TEST_RATIO:  float = 0.15
SEED:        int   = 42

CPU                 = os.cpu_count() or 8
THREADS:     int    = max(32, CPU * 8)  # I/O-bound: high thread count is fine
USE_HARDLINK: bool  = False              # fastest when source & dest on same filesystem
USE_SYMLINK:  bool  = False             # instant references across filesystems
VERBOSITY:   int    = 1                 # 0=WARNING, 1=INFO, 2=DEBUG
# ==========================================================================

RAW_EXTS      = (".dat",)
IMG_EXTS      = (".tif", ".tiff", ".png")
PATCH_TOKS    = ("VV", "VH", "SLC", "GRD")

# Namespace-tolerant <SARProduct> extraction
SAR_RE = re.compile(
    rb"<\s*(?:[\w.:-]+:)?SARProduct\s*>\s*([^<]+)\s*<\s*/\s*(?:[\w.:-]+:)?SARProduct\s*>",
    re.IGNORECASE,
)
SAFE_SUFFIX_RE = re.compile(r"\.SAFE/?$", re.IGNORECASE)


def setup_logger(verbosity: int) -> None:
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")


def normalize_scene_name(scene: str) -> str:
    """Normalize product name; drop trailing '.SAFE' and prefer S1A_/S1B_ segments."""
    s = scene.strip().replace("\\", "/")
    parts = [p for p in s.split("/") if p] or [s]
    candidate = next((p for p in parts if p.startswith(("S1A_", "S1B_"))), max(parts, key=len))
    return SAFE_SUFFIX_RE.sub("", candidate)


def extract_sarproduct(xml_path: Path) -> str | None:
    """Extract normalized <SARProduct> quickly, fallback to streaming parse."""
    try:
        with open(xml_path, "rb") as f:
            chunk = f.read(131072)
        m = SAR_RE.search(chunk)
        if m:
            raw = m.group(1).decode(errors="ignore").strip()
            return normalize_scene_name(raw)
    except Exception as e:
        logging.debug("Fast scan failed for %s: %s", xml_path, e)

    try:
        for _, elem in ET.iterparse(xml_path, events=("end",)):
            if isinstance(elem.tag, str) and elem.tag.endswith("SARProduct") and elem.text:
                val = normalize_scene_name(elem.text)
                elem.clear()
                return val
            elem.clear()
    except ET.ParseError as e:
        logging.warning("XML parse error in %s: %s", xml_path, e)
        return None
    return None


def get_id(name: str) -> str | None:
    """Return the last numeric token in a filename like ..._00123.tif → '00123'."""
    stem = name.rsplit(".", 1)[0]
    for token in reversed(stem.split("_")):
        if token.isdigit():
            return token
    return None


def index_folder(folder: Path, exts: tuple[str, ...], require_tokens: tuple[str, ...] | None = None) -> Dict[str, List[Path]]:
    """One scan → {id: [paths]}."""
    mapping: Dict[str, List[Path]] = defaultdict(list)
    need = tuple(t.upper() for t in require_tokens or ())
    exts_low = tuple(x.lower() for x in exts)

    with os.scandir(folder) as it:
        for e in it:
            if not e.is_file():
                continue
            name = e.name
            if not name.lower().endswith(exts_low):
                continue
            if need and not any(f"_{tok}" in name.upper() for tok in need):
                continue
            id_ = get_id(name)
            if id_:
                mapping[id_].append(Path(e.path))
    logging.info("Indexed %d IDs from %s", len(mapping), folder)
    return mapping


def build_id_to_scene(xml_map: Dict[str, List[Path]]) -> tuple[Dict[str, str], Dict[str, List[Path]]]:
    """Return (id→scene, scene→xmls) by reading <SARProduct> from sample XMLs."""
    id_to_scene: Dict[str, str] = {}
    scene_to_xmls: Dict[str, List[Path]] = defaultdict(list)

    for id_, xml_list in xml_map.items():
        scene = None
        for xp in xml_list[:2]:  # usually one is enough
            scene = extract_sarproduct(xp)
            if scene:
                break
        if not scene:
            logging.warning("No <SARProduct> for id %s (sample: %s) -> skipping this id", id_, xml_list[0])
            continue
        id_to_scene[id_] = scene
        scene_to_xmls[scene].extend(xml_list)

    logging.info("Resolved %d scenes from XMLs", len(scene_to_xmls))
    return id_to_scene, scene_to_xmls


def remap_by_scene(id_map: Dict[str, List[Path]], id_to_scene: Dict[str, str]) -> Dict[str, List[Path]]:
    """Remap {id: [paths]} → {scene: [paths]} using id_to_scene. Unmapped IDs are dropped (warned earlier)."""
    scene_map: Dict[str, List[Path]] = defaultdict(list)
    dropped = 0
    for id_, paths in id_map.items():
        scene = id_to_scene.get(id_)
        if scene:
            scene_map[scene].extend(paths)
        else:
            dropped += len(paths)
    if dropped:
        logging.warning("Dropped %d files whose IDs were not in XML scene map", dropped)
    return scene_map


def ensure_dirs(base: Path) -> tuple[Path, Path, Path]:
    """Create split subdirs and return destination roots (dst_xml, dst_pt, dst_rw)."""
    dst_xml = base / "labels" / "xml_labelling"
    dst_pt  = base / "patches"
    dst_rw  = base / "raw"
    dst_xml.mkdir(parents=True, exist_ok=True)
    dst_pt.mkdir(parents=True, exist_ok=True)
    dst_rw.mkdir(parents=True, exist_ok=True)
    return dst_xml, dst_pt, dst_rw


def copy_one(job: Tuple[Path, Path, bool, bool]) -> None:
    """Fast copy with fallbacks: hardlink → symlink → copyfile."""
    src, dst, hardlink, symlink = job
    try:
        if dst.exists():
            return
        if hardlink:
            try:
                os.link(src, dst)
                return
            except OSError:
                pass
        if symlink:
            try:
                os.symlink(src, dst)
                return
            except OSError:
                pass
        shutil.copyfile(src, dst)
    except FileExistsError:
        pass
    except Exception as e:
        logging.warning("Copy failed %s -> %s : %s", src, dst, e)


def job_iter(
    split_scenes: dict[str, list[str]],
    scene_to_xmls: Dict[str, List[Path]],
    scene_to_patches: Dict[str, List[Path]],
    scene_to_raw: Dict[str, List[Path]],
    out_root: Path,
    hardlink: bool,
    symlink: bool,
) -> Iterator[tuple[Path, Path, bool, bool]]:
    """Yield (src, dst, hardlink, symlink) pairs lazily to avoid huge in-memory job lists."""
    for split, scene_list in split_scenes.items():
        base = out_root / split
        dst_xml, dst_pt, dst_rw = ensure_dirs(base)

        for scene in scene_list:
            for f in scene_to_xmls.get(scene, ()):
                yield (f, dst_xml / f.name, hardlink, symlink)
            for f in scene_to_patches.get(scene, ()):
                yield (f, dst_pt / f.name,  hardlink, symlink)
            for f in scene_to_raw.get(scene, ()):
                yield (f, dst_rw / f.name,  hardlink, symlink)


def run() -> None:
    """Pipeline using the constants defined at the top."""
    setup_logger(VERBOSITY)

    # Validate inputs
    for p in (LABEL_DIR, PATCHES_DIR, RAW_DIR):
        if not p.is_dir():
            sys.exit(f"[ERROR] folder not found: {p}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # LABEL_DIR must be the xml_labelling folder (your structure already has labels/xml_labelling on input)
    if LABEL_DIR.name != "xml_labelling":
        sys.exit(f"[ERROR] LABEL_DIR must be the xml_labelling folder (got: {LABEL_DIR})")

    # 1) index
    xml_map   = index_folder(LABEL_DIR, (".xml",))
    patch_map = index_folder(PATCHES_DIR, IMG_EXTS, PATCH_TOKS)
    raw_map   = index_folder(RAW_DIR,   RAW_EXTS)

    # 2) scene mapping via <SARProduct>
    id_to_scene, scene_to_xmls = build_id_to_scene(xml_map)
    scenes = sorted(scene_to_xmls)
    if not scenes:
        sys.exit("[ERROR] No scenes could be resolved from XML <SARProduct> tags")

    scene_to_patches = remap_by_scene(patch_map, id_to_scene)
    scene_to_raw     = remap_by_scene(raw_map,   id_to_scene)

    # 3) split scenes
    if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-6:
        sys.exit(f"[ERROR] Ratios must sum to 1.0 (got {TRAIN_RATIO + VAL_RATIO + TEST_RATIO:.4f})")

    random.seed(SEED)
    random.shuffle(scenes)
    n = len(scenes)
    n_tr, n_val = int(TRAIN_RATIO * n), int(VAL_RATIO * n)
    split_scenes = {
        "train": scenes[:n_tr],
        "val":   scenes[n_tr:n_tr + n_val],
        "test":  scenes[n_tr + n_val:],
    }
    logging.info("Scenes %d → train %d, val %d, test %d", n, len(split_scenes["train"]), len(split_scenes["val"]), len(split_scenes["test"]))

    # 4) threaded copy – stream jobs
    jobs = job_iter(split_scenes, scene_to_xmls, scene_to_patches, scene_to_raw, OUTPUT_ROOT, USE_HARDLINK, USE_SYMLINK)
    chunksize = 512

    total = 0
    with cf.ThreadPoolExecutor(max_workers=THREADS) as pool:
        for i, _ in enumerate(pool.map(copy_one, jobs, chunksize=chunksize), 1):
            total = i
            if i % 5000 == 0:
                logging.info("copied %d files...", i)

    print(f"Done. Copied ~{total} files. Output: {OUTPUT_ROOT.resolve()}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
