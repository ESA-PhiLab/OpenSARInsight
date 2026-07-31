"""
Splitting code for the RFI "aresys_full" dataset

• 70/15/15 train/val/test split
• Copies into identical split/{train,val,test}/ trees
    ├─ labels: xml_labelling/*.xml, binary_masks/*.png
    ├─ patches/* containing _SLC,_GRD,_VV,_VH
    └─ raw/*.dat

• How to run the code: python dataset-splitting-rfi.py
"""

from __future__ import annotations
import concurrent.futures as cf
import os, random, re, shutil, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from dataset_generation_scripts.utils import get_config
cfg = get_config("DGS_CONFIG_PATH")

######################### paths & split ratios ############################
LABEL_DIR   = Path(cfg["dataset_splitting"]["rfi"]["label_dir"])
MASKS_DIR   = Path(cfg["dataset_splitting"]["rfi"]["masks_dir"])
PATCHES_DIR = Path(cfg["dataset_splitting"]["rfi"]["patches_dir"])
RAW_DIR     = Path(cfg["dataset_splitting"]["rfi"]["raw_dir"])
OUTPUT_ROOT = Path(cfg["dataset_splitting"]["rfi"]["output_dir"])

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42

THREADS      = 8        # parallel copy workers
USE_HARDLINK = False    # True ⇒ os.link (fast, same filesystem)
RAW_EXTS   = (".dat",)
IMG_EXTS   = (".tif", ".tiff", ".png")
PATCH_TOKS = ("VV", "VH", "SLC", "GRD")

##########################################################################

ID_RE = re.compile(r"_(\d+)(?:_|$)")
def get_id(name: str) -> str | None:
    """Return last numeric token before '.' or None."""
    stem = name.rsplit(".", 1)[0]
    for token in reversed(stem.split("_")):
        if token.isdigit():
            return token
    return None

############################ indexing ###########################
def index_folder(folder: Path,
                 exts: tuple[str, ...],
                 require_tokens: tuple[str, ...] | None = None) -> Dict[str, List[Path]]:
    """
    Walk folder once; build {id: [paths]}.
    If `require_tokens` set, filename must contain at least one of those tokens.
    """
    mapping: Dict[str, List[Path]] = defaultdict(list)
    need = tuple(t.upper() for t in require_tokens or ())
    for e in os.scandir(folder):
        if not e.is_file():
            continue
        if not e.name.lower().endswith(tuple(ext.lower() for ext in exts)):
            continue
        if need and not any(f"_{tok}" in e.name.upper() for tok in need):
            continue
        if (id_ := get_id(e.name)):
            mapping[id_].append(Path(e.path))
    return mapping

########################## copy helpers ##########################
def copy_one(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst) if USE_HARDLINK else shutil.copy2(src, dst)
    except FileExistsError:
        pass  

########################## main ##########################
def main() -> None:
    for p in (LABEL_DIR, MASKS_DIR, PATCHES_DIR, RAW_DIR):
        if not p.is_dir():
            sys.exit(f"[ERROR] folder not found: {p}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 1) build ID→files maps
    xml_map   = index_folder(LABEL_DIR, (".xml",))
    mask_map  = index_folder(MASKS_DIR, (".png",))
    patch_map = index_folder(PATCHES_DIR, IMG_EXTS, PATCH_TOKS)
    raw_map   = index_folder(RAW_DIR,   RAW_EXTS)    

    ids = sorted(xml_map)  # only IDs with XMLs define the split 
    if not ids:
        sys.exit("[ERROR] No XML IDs found")

    # 2) split IDs
    random.seed(SEED)
    random.shuffle(ids)
    n = len(ids)
    n_tr, n_val = int(TRAIN_RATIO*n), int(VAL_RATIO*n)
    split_ids = {
        "train": ids[:n_tr],
        "val":   ids[n_tr:n_tr+n_val],
        "test":  ids[n_tr+n_val:],
    }

    print(f"IDs {n} → train {len(split_ids['train'])}, "
          f"val {len(split_ids['val'])}, test {len(split_ids['test'])}")

    # 3) threaded copy
    with cf.ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = []
        for split, id_list in split_ids.items():
            for id_ in id_list:
                # XMLs
                dst_xml = OUTPUT_ROOT / split / "labels/xml_labelling"
                for f in xml_map.get(id_, []):
                    futures.append(pool.submit(copy_one, f, dst_xml/f.name))

                # Masks
                dst_mk = OUTPUT_ROOT / split / "labels/binary_masks"
                for f in mask_map.get(id_, []):
                    futures.append(pool.submit(copy_one, f, dst_mk/f.name))

                # Patches
                dst_pt = OUTPUT_ROOT / split / "patches"
                for f in patch_map.get(id_, []):
                    futures.append(pool.submit(copy_one, f, dst_pt/f.name))

                # Raw
                dst_rw = OUTPUT_ROOT / split / "raw"
                for f in raw_map.get(id_, []):
                    futures.append(pool.submit(copy_one, f, dst_rw/f.name))

        for i, _ in enumerate(cf.as_completed(futures), 1):
            if i % 1000 == 0:
                print(f"copied {i}/{len(futures)} files")

    print("Done.  Output:", OUTPUT_ROOT.resolve())

if __name__ == "__main__":
    main()
