"""
Splitting code for the flood “kurosiwo_validation_sample” dataset

• 70/15/15 train/val/test split
• Copies into identical split/{train,val,test}/ trees
    ├─ xml_labelling/*.xml
    ├─ patches/* containing _SLC,_GRD,_VV,_VH
    └─ raw/*.dat

• How to run the code: python dataset-splitting-flood.py
"""

from __future__ import annotations
import concurrent.futures as cf
import os, random, re, shutil, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from dataset_generation_scripts.utils import get_config
cfg = get_config("DGS_CONFIG_PATH")

######################### paths & split ratios ##########################
LABEL_DIR   = Path(cfg["dataset_splitting"]["flood"]["label_dir"])
PATCHES_DIR = Path(cfg["dataset_splitting"]["flood"]["patches_dir"])
RAW_DIR     = Path(cfg["dataset_splitting"]["flood"]["raw_dir"])
OUTPUT_ROOT = Path(cfg["dataset_splitting"]["flood"]["output_dir"])


TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
SEED        = 42
THREADS      = 8          # parallel copy workers
USE_HARDLINK = False      # True ⇒ os.link (same filesystem)
RAW_EXTS   = (".dat",)
IMG_EXTS   = (".tif", ".tiff", ".png")
PATCH_TOKS = ("SLC", "GRD", "VV", "VH")   # keep all variants
###########################################################################

ID_RE = re.compile(r"_(\d+)(?:_|\.|$)")
def get_id(name: str) -> str | None:
    """Return last numeric token before '.' or None."""
    stem = name.rsplit(".", 1)[0]
    for part in reversed(stem.split("_")):
        if part.isdigit():
            return part
    return None


############################ indexing helpers ###########################
def index_by_id(folder: Path,
                exts: tuple[str, ...],
                must_tokens: tuple[str, ...] | None = None) -> Dict[str, List[Path]]:
    """One scan → {id: [paths]}"""
    mapping: Dict[str, List[Path]] = defaultdict(list)
    req = tuple(t.upper() for t in must_tokens or ())
    for e in os.scandir(folder):
        if not e.is_file():
            continue
        if not e.name.lower().endswith(tuple(ext.lower() for ext in exts)):
            continue
        if must_tokens and not any(f"_{tok}" in e.name.upper() for tok in req):
            continue
        if (id_ := get_id(e.name)):
            mapping[id_].append(Path(e.path))
    return mapping

def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        (os.link if USE_HARDLINK else shutil.copy2)(src, dst)
    except FileExistsError:
        pass

############################ main ###########################
def main() -> None:
    for folder in (LABEL_DIR, PATCHES_DIR, RAW_DIR):
        if not folder.is_dir():
            sys.exit(f"[ERROR] Folder not found: {folder}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    xml_map   = index_by_id(LABEL_DIR, (".xml",))
    patch_map = index_by_id(PATCHES_DIR, IMG_EXTS, PATCH_TOKS)
    raw_map   = index_by_id(RAW_DIR,   RAW_EXTS)

    ids = list(xml_map.keys())
    if not ids:
        sys.exit("[ERROR] No XML IDs found")

    random.seed(SEED)
    random.shuffle(ids)
    n = len(ids)
    n_tr, n_val = int(TRAIN_RATIO*n), int(VAL_RATIO*n)
    splits = {
        "train": ids[:n_tr],
        "val":   ids[n_tr:n_tr+n_val],
        "test":  ids[n_tr+n_val:],
    }
    print(f"IDs {n} → train {len(splits['train'])}, "
          f"val {len(splits['val'])}, test {len(splits['test'])}")

    with cf.ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = []
        for split, id_list in splits.items():
            for id_ in id_list:
                dst_xml = OUTPUT_ROOT / split / "xml_labelling"
                for f in xml_map[id_]:
                    futures.append(pool.submit(copy_file, f, dst_xml/f.name))

                dst_pt = OUTPUT_ROOT / split / "patches"
                for f in patch_map.get(id_, []):
                    futures.append(pool.submit(copy_file, f, dst_pt/f.name))

                dst_rw = OUTPUT_ROOT / split / "raw"
                for f in raw_map.get(id_, []):
                    futures.append(pool.submit(copy_file, f, dst_rw/f.name))

        for i, _ in enumerate(cf.as_completed(futures), 1):
            if i % 1000 == 0:
                print(f"copied {i}/{len(futures)} files")

    print("Finished.  Output:", OUTPUT_ROOT.resolve())

if __name__ == "__main__":
    main()
