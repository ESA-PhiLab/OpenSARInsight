"""
---------------------------------------------------------------------
Project: OpenSAR Insight
---------------------------------------------------------------------
filter_dataset.py

Tool: Filter YOLO Dataset to Verified Vessel Labels

Author: Abdulhameed Yunusa (ABHY)
E-mail: ayunusa@indracompany.com
Creation Date: 2026-04-08

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""

import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.rc_labels import *

from dataset_generation_scripts.utils import get_config
cfg = get_config("MODEL_USECASES_PATH")

SRC = cfg["filter_dataset"]["src"]
DST = cfg["filter_dataset"]["dst"]
SPLITS = cfg["filter_dataset"]["splits"]


def build_allowed_prefixes(labels: list[str]) -> set[str]:
    """
    Convert label IDs like 'VD_9_VV' to the file prefix 'DB_OPENSAR_VV_VD_9'.
    Labels without polarization (e.g. 'VD_8') generate both VV and VH prefixes.
    """
    prefixes = set()
    for lbl in labels:
        # Match VD_{num}_{pol} or VD_{num}
        m = re.match(r"^(VD_\d+)_?(VV|VH|V)?$", lbl)
        if not m:
            print(f"WARNING: could not parse label '{lbl}', skipping")
            continue
        vd_part = m.group(1)  # e.g. VD_9
        pol = m.group(2)      # e.g. VV, VH, V, or None

        if pol and pol in ("VV", "VH"):
            prefixes.add(f"DB_OPENSAR_{pol}_{vd_part}")
        elif pol == "V":
            # Likely typo for VV, include both to be safe
            prefixes.add(f"DB_OPENSAR_VV_{vd_part}")
            prefixes.add(f"DB_OPENSAR_VH_{vd_part}")
        else:
            # No polarization suffix — include both
            prefixes.add(f"DB_OPENSAR_VV_{vd_part}")
            prefixes.add(f"DB_OPENSAR_VH_{vd_part}")

    return prefixes


def matches_allowed(filename: str, allowed_prefixes: set[str]) -> bool:
    """
    Check if a filename (e.g. DB_OPENSAR_VV_VD_9_aug1.png) matches any allowed prefix.
    The prefix is everything before an optional _aug suffix and the extension.
    """
    stem = os.path.splitext(filename)[0]  # strip .png / .txt
    # Remove _aug1, _aug2, etc.
    base = re.sub(r"_aug\d+$", "", stem)
    return base in allowed_prefixes


def main():
    allowed = build_allowed_prefixes(
        labels_visible_on_rc_and_align + labels_visible_on_rc_and_align_partly
    )
    print(f"Allowed prefixes: {len(allowed)}")

    summary_before = {}
    summary_after = {}

    for split in SPLITS:
        src_img_dir = os.path.join(SRC, "images", split)
        src_lbl_dir = os.path.join(SRC, "labels", split)
        dst_img_dir = os.path.join(DST, "images", split)
        dst_lbl_dir = os.path.join(DST, "labels", split)

        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        if not os.path.isdir(src_img_dir):
            print(f"Skipping split '{split}': {src_img_dir} not found")
            continue

        all_images = sorted(os.listdir(src_img_dir))
        kept = 0

        for img_name in all_images:
            if not img_name.endswith(".png"):
                continue

            if matches_allowed(img_name, allowed):
                # Copy image
                shutil.copy2(
                    os.path.join(src_img_dir, img_name),
                    os.path.join(dst_img_dir, img_name),
                )
                # Copy matching label
                lbl_name = img_name.replace(".png", ".txt")
                lbl_src = os.path.join(src_lbl_dir, lbl_name)
                if os.path.exists(lbl_src):
                    shutil.copy2(lbl_src, os.path.join(dst_lbl_dir, lbl_name))
                kept += 1

        total = sum(1 for f in all_images if f.endswith(".png"))
        summary_before[split] = total
        summary_after[split] = kept
        print(f"[{split}] {kept}/{total} images kept")

    # Copy data.yaml with updated path
    src_yaml = os.path.join(SRC, "data.yaml")
    if os.path.exists(src_yaml):
        shutil.copy2(src_yaml, os.path.join(DST, "data.yaml"))
        # Update path in the copied yaml
        yaml_path = os.path.join(DST, "data.yaml")
        with open(yaml_path, "r") as f:
            content = f.read()
        content = content.replace(SRC, DST)
        with open(yaml_path, "w") as f:
            f.write(content)

    # Print summary
    print("\n" + "=" * 60)
    print(f"{'Split':<10} {'Before':>10} {'After':>10} {'Removed':>10} {'% Kept':>10}")
    print("-" * 60)
    total_b, total_a = 0, 0
    for split in SPLITS:
        b = summary_before.get(split, 0)
        a = summary_after.get(split, 0)
        total_b += b
        total_a += a
        pct = (a / b * 100) if b else 0
        print(f"{split:<10} {b:>10} {a:>10} {b - a:>10} {pct:>9.1f}%")
    pct_total = (total_a / total_b * 100) if total_b else 0
    print("-" * 60)
    print(f"{'TOTAL':<10} {total_b:>10} {total_a:>10} {total_b - total_a:>10} {pct_total:>9.1f}%")
    print("=" * 60)
    print(f"\nFiltered dataset saved to: {DST}")


if __name__ == "__main__":
    main()
