"""
Splitting code for the vessel "viewsar3_full" dataset.

• 70/15/15 train/val/test split
• Copies into identical split/{train,val,test}/ trees
    ├─ labels: xml_labelling/*.xml
    ├─ patches/* containing _SLC,_GRD,_VV,_VH
    ├─ patches0/* containing _SLC,_GRD,_VV,_VH
    └─ raw/*.dat

• How to run the code: python dataset-splitting-vessel.py
"""


from __future__ import annotations
import concurrent.futures as cf
import os, random, re, shutil, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from dataset_generation_scripts.utils import get_config
cfg = get_config("DGS_CONFIG_PATH")

######################### paths & split ratios ########################
LABEL_DIR    = Path(cfg["dataset_splitting"]["vessel"]["label_dir"])
RAW_DIR      = Path(cfg["dataset_splitting"]["vessel"]["raw_dir"])
PATCHES_DIR  = Path(cfg["dataset_splitting"]["vessel"]["patches_dir"])
PATCHES0_DIR = Path(cfg["dataset_splitting"]["vessel"]["patches0_dir"])
OUTPUT_ROOT  = Path(cfg["dataset_splitting"]["vessel"]["output_dir"])

TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
SEED         = 42

THREADS      = 8            # parallel copy workers
USE_HARDLINK = False        # True ⇒ os.link instead of shutil.copy2
RAW_EXTS     = (".dat",)
IMG_EXTS     = (".tif", ".tiff", ".png")
PATCH_TOKS   = ("VV", "VH", "SLC", "GRD")
#######################################################################

ID_RE = re.compile(r"_(\d+)(?:_|$)")
def get_id(name: str) -> str | None:
    """Return last numeric token before dot, or None."""
    stem = name.rsplit(".", 1)[0]
    for tok in reversed(stem.split("_")):
        if tok.isdigit():
            return tok
    return None


############################ indexing helpers ###########################
def index_by_id(folder: Path,
                exts: tuple[str, ...],
                require_tokens: tuple[str, ...] | None = None) -> Dict[str, List[Path]]:
    """
    Scan `folder` once and build {id: [file_paths]}.
    If `require_tokens` is given, file basename must contain at least one of them.
    """
    mapping: Dict[str, List[Path]] = defaultdict(list)
    req_upper = tuple(tok.upper() for tok in (require_tokens or ()))
    for entry in os.scandir(folder):
        if not entry.is_file():
            continue
        if not entry.name.lower().endswith(tuple(ext.lower() for ext in exts)):
            continue
        if require_tokens and not any(f"_{tok}" in entry.name.upper() for tok in req_upper):
            continue
        id_ = get_id(entry.name)
        if id_:
            mapping[id_].append(Path(entry.path))
    return mapping


############################### copy util ##################################
def copy_one(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if USE_HARDLINK:
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)
    except FileExistsError:
        pass  # another worker already placed it


################################ main ####################################
def main() -> None:
    for folder in (LABEL_DIR, RAW_DIR, PATCHES_DIR, PATCHES0_DIR):
        if not folder.is_dir():
            sys.exit(f"[ERROR] folder not found: {folder}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. gather label IDs
    label_map: Dict[str, List[Path]] = defaultdict(list)
    for xml in LABEL_DIR.glob("*.xml"):
        if (id_ := get_id(xml.name)):
            label_map[id_].append(xml)
    if not label_map:
        sys.exit("[ERROR] No XML IDs found in label folder")

    # 2. index other folders ONCE
    raw_map      = index_by_id(RAW_DIR,      RAW_EXTS)          
    patches_map  = index_by_id(PATCHES_DIR,  IMG_EXTS, PATCH_TOKS)
    patches0_map = index_by_id(PATCHES0_DIR, IMG_EXTS, PATCH_TOKS)

    # 3. split IDs
    ids = list(label_map.keys())
    random.seed(SEED)
    random.shuffle(ids)
    n = len(ids)
    n_tr = int(TRAIN_RATIO * n)
    n_vl = int(VAL_RATIO * n)
    split_ids = {
        "train": ids[:n_tr],
        "val":   ids[n_tr:n_tr + n_vl],
        "test":  ids[n_tr + n_vl:],
    }
    print(f"IDs {n} → train {len(split_ids['train'])}, "
          f"val {len(split_ids['val'])}, test {len(split_ids['test'])}\n")

    # 4. threaded copy
    with cf.ThreadPoolExecutor(max_workers=THREADS) as pool:
        futs = []

        for split, id_list in split_ids.items():
            for id_ in id_list:
                # XMLs
                dst_lbl = OUTPUT_ROOT / split / "labels" / LABEL_DIR.name
                for xml in label_map[id_]:
                    futs.append(pool.submit(copy_one, xml, dst_lbl / xml.name))

                # RAW
                dst_raw = OUTPUT_ROOT / split / "raw"
                for f in raw_map.get(id_, []):
                    futs.append(pool.submit(copy_one, f, dst_raw / f.name))

                # PATCHES
                dst_p = OUTPUT_ROOT / split / "patches" / PATCHES_DIR.name
                for f in patches_map.get(id_, []):
                    futs.append(pool.submit(copy_one, f, dst_p / f.name))

                # PATCHES0
                dst_p0 = OUTPUT_ROOT / split / "patches" / PATCHES0_DIR.name
                for f in patches0_map.get(id_, []):
                    futs.append(pool.submit(copy_one, f, dst_p0 / f.name))

        # progress print every 1000 copies
        for i, _ in enumerate(cf.as_completed(futs), 1):
            if i % 1000 == 0:
                print(f"copied {i}/{len(futs)} files")

    print("Done.  Output:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()
