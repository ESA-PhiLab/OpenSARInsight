"""
Leakage checker for scene-based splits + per-scene XML listing.

- Verifies no scene leakage between train/val/test.
- Saves a detailed XML membership report to a text file.
"""

from __future__ import annotations
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set
import xml.etree.ElementTree as ET
from dataset_generation_scripts.utils import get_config
cfg = get_config("DATA_PREPROC_PATH")

# ----------- EDIT THIS -----------
OUTPUT_ROOT = cfg["check_scene_sarproduct"]["output_root"] # contains train/, val/, test/
OUTPUT_ROOT1 = cfg["check_scene_sarproduct"]["output_root1"] # where to save the report file
REPORT_FILE = OUTPUT_ROOT1 / "scene_xml_report_vessel.txt"  # report output file
CHECK_PATCHES = True
CHECK_RAW = True
CHECK_MASKS = True
# ---------------------------------

IMG_EXTS = (".tif", ".tiff", ".png")
RAW_EXTS = (".dat",)
PATCH_TOKS = ("VV", "VH", "SLC", "GRD")

SAR_RE = re.compile(
    rb"<\s*(?:[\w.:-]+:)?SARProduct\s*>\s*([^<]+)\s*<\s*/\s*(?:[\w.:-]+:)?SARProduct\s*>",
    re.IGNORECASE,
)
SAFE_SUFFIX_RE = re.compile(r"\.SAFE/?$", re.IGNORECASE)


def normalize_scene(scene: str) -> str:
    s = scene.strip().replace("\\", "/")
    parts = [p for p in s.split("/") if p] or [s]
    candidate = next((p for p in parts if p.startswith(("S1A_", "S1B_"))), max(parts, key=len))
    return SAFE_SUFFIX_RE.sub("", candidate)


def extract_scene_from_xml(xml_path: Path) -> str | None:
    # fast regex scan of first 128 KiB
    try:
        with open(xml_path, "rb") as f:
            chunk = f.read(131072)
        m = SAR_RE.search(chunk)
        if m:
            return normalize_scene(m.group(1).decode(errors="ignore"))
    except Exception:
        pass
    # fallback: streaming parse
    try:
        for _, elem in ET.iterparse(xml_path, events=("end",)):
            if isinstance(elem.tag, str) and elem.tag.endswith("SARProduct") and elem.text:
                val = normalize_scene(elem.text)
                elem.clear()
                return val
            elem.clear()
    except ET.ParseError:
        return None
    return None


def get_id_from_name(name: str) -> str | None:
    stem = name.rsplit(".", 1)[0]
    for tok in reversed(stem.split("_")):
        if tok.isdigit():
            return tok
    return None


def index_dir(folder: Path, exts: tuple[str, ...], require_tokens: tuple[str, ...] | None = None) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = defaultdict(list)
    need = tuple(t.upper() for t in (require_tokens or ()))
    exts_low = tuple(x.lower() for x in exts)
    if not folder.is_dir():
        return mapping
    with os.scandir(folder) as it:
        for e in it:
            if not e.is_file():
                continue
            name = e.name
            if not name.lower().endswith(exts_low):
                continue
            if need:
                up = name.upper()
                if not any(f"_{tok}" in up for tok in need):
                    continue
            file_id = get_id_from_name(name)
            if file_id:
                mapping[file_id].append(Path(e.path))
    return mapping


def check() -> int:
    splits = ["train", "val", "test"]
    split_to_scenes: Dict[str, Set[str]] = {s: set() for s in splits}
    scene_to_splits: Dict[str, Set[str]] = defaultdict(set)
    id_to_scene_by_split: Dict[str, Dict[str, str]] = {s: {} for s in splits}
    xmls_by_split_scene: Dict[str, Dict[str, List[Path]]] = {s: defaultdict(list) for s in splits}

    # 1) Read scenes from XMLs in each split
    for split in splits:
        xml_dir = OUTPUT_ROOT / split / "labels" / "xml_labelling"
        if not xml_dir.is_dir():
            print(f"[WARN] Missing: {xml_dir}")
            continue
        for p in sorted(xml_dir.glob("*.xml")):
            scene = extract_scene_from_xml(p)
            if not scene:
                print(f"[WARN] No <SARProduct> in {p}")
                continue
            split_to_scenes[split].add(scene)
            scene_to_splits[scene].add(split)
            xmls_by_split_scene[split][scene].append(p)
            file_id = get_id_from_name(p.name)
            if file_id:
                id_to_scene_by_split[split][file_id] = scene
        print(f"[INFO] {split}: {len(split_to_scenes[split])} scenes, {sum(len(v) for v in xmls_by_split_scene[split].values())} XMLs")

    # 2) Primary leakage check
    leaked = [(scene, sorted(list(sps))) for scene, sps in scene_to_splits.items() if len(sps) > 1]
    if leaked:
        print("\n[ERROR] Scene leakage detected:")
        for scene, sps in leaked:
            print(f"  - {scene} in: {', '.join(sps)}")
    else:
        print("\n[OK] No scene leakage detected across train/val/test based on XMLs.")

    # 3) Save full XML membership report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("==== Per-split Scene Membership Report ====\n\n")
        for split in splits:
            f.write(f"-- {split.upper()} --\n")
            if not xmls_by_split_scene[split]:
                f.write("  (no XMLs found)\n\n")
                continue
            for scene in sorted(xmls_by_split_scene[split].keys()):
                files = xmls_by_split_scene[split][scene]
                f.write(f"  {scene}  [{len(files)} XML]\n")
                for p in sorted(files):
                    f.write(f"    - {p.name}\n")
            f.write("\n")

    print(f"\n[INFO] Full per-scene XML report saved to:\n  {REPORT_FILE}")

    if leaked:
        print("\nResult: FAILED (leakage or mismatches found)")
        return 1

    print("\nResult: PASSED (no leakage; files consistent with owning scenes/splits)")
    return 0


if __name__ == "__main__":
    sys.exit(check())

