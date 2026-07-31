"""
---------------------------------------------------------------------
Project: OpenSAR Insight
---------------------------------------------------------------------
inflate_label.py

Tool: Inflate YOLO bounding boxes by padding pixels

Author: Abdulhameed Yunusa (ABHY)
E-mail: ayunusa@indracompany.com
Creation Date: 2026-02-26

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------

Copies a YOLO dataset directory and inflates every bounding box by PADDING px
on each side (so each box grows by 2*PADDING px in width and height).

Labels are expected in YOLO normalised format:
    class x_center y_center width height   (all values 0–1)

The corresponding image is used to convert px ↔ normalised coords.
Boxes are clamped to [0, 1] so they never exceed image boundaries.

Usage:
    python inflate_labels.py \
        --src  /path/to/yolo_dataset \
        --dst  /path/to/yolo_dataset_inflated \
        --pad  10

Directory structure assumed (mirrors src into dst):
    <root>/
        images/
            train/  val/  test/
        labels/
            train/  val/  test/
"""

import argparse
import shutil
from pathlib import Path
from PIL import Image

# ─── defaults ────────────────────────────────────────────────────────────────
DEFAULT_PADDING = 5          # pixels to add on EACH side
LABEL_EXTS      = {'.txt'}
IMAGE_EXTS      = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
SPLITS          = ['train', 'val', 'test']
# ─────────────────────────────────────────────────────────────────────────────


def find_image(label_path: Path, src_root: Path) -> Path | None:
    """
    Given a label path, locate the matching image by swapping
    'labels' → 'images' in the path and trying every image extension.
    """
    rel      = label_path.relative_to(src_root)          # e.g. labels/train/foo.txt
    parts    = list(rel.parts)
    try:
        parts[parts.index('labels')] = 'images'
    except ValueError:
        return None

    stem = Path(*parts).with_suffix('')                   # drop .txt
    for ext in IMAGE_EXTS:
        candidate = src_root / stem.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def inflate_box(x_c, y_c, w, h, pad_x, pad_y):
    """
    Inflate a normalised YOLO box by pad_x / pad_y (already in normalised units).
    Returns clamped (x_c, y_c, w_new, h_new).
    """
    w_new = min(w + 2 * pad_x, 1.0)
    h_new = min(h + 2 * pad_y, 1.0)

    # Re-centre and clamp centre so the box stays inside the image
    x_c_new = max(w_new / 2, min(x_c, 1.0 - w_new / 2))
    y_c_new = max(h_new / 2, min(y_c, 1.0 - h_new / 2))

    return x_c_new, y_c_new, w_new, h_new


def process_label(label_src: Path, label_dst: Path, img_w: int, img_h: int, padding: int):
    """Read, inflate, and write a single label file."""
    lines_out = []
    with open(label_src) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                lines_out.append(line)          # pass through malformed lines
                continue

            cls              = parts[0]
            x_c, y_c, w, h  = map(float, parts[1:5])
            extra            = parts[5:]        # preserve any extra fields

            pad_x = padding / img_w             # normalise padding
            pad_y = padding / img_h

            x_c, y_c, w, h = inflate_box(x_c, y_c, w, h, pad_x, pad_y)

            row = f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
            if extra:
                row += ' ' + ' '.join(extra)
            lines_out.append(row + '\n')

    label_dst.parent.mkdir(parents=True, exist_ok=True)
    with open(label_dst, 'w') as f:
        f.writelines(lines_out)


def copy_and_inflate(src_root: Path, dst_root: Path, padding: int):
    src_root = src_root.resolve()
    dst_root = dst_root.resolve()

    if dst_root.exists():
        print(f"Removing existing destination: {dst_root}")
        shutil.rmtree(dst_root)

    # ── Copy entire tree first (images, yaml) ──────────────────────────
    print(f"Copying {src_root} → {dst_root} …")
    shutil.copytree(src_root, dst_root)
    print("Copy done. Inflating labels …\n")

    # ── Find and re-write every label file ───────────────────────────────────
    labels_root = dst_root / 'labels'
    if not labels_root.exists():
        # Try to find label files anywhere in the tree
        label_files = list(dst_root.rglob('*.txt'))
    else:
        label_files = list(labels_root.rglob('*.txt'))

    total = skipped = 0
    for label_dst in label_files:
        # Corresponding label in src (to read from; dst is already a copy)
        label_src = src_root / label_dst.relative_to(dst_root)

        # Find matching image in src to get dimensions
        img_path = find_image(label_src, src_root)
        if img_path is None:
            print(f"  [SKIP] No image found for {label_src.name}")
            skipped += 1
            continue

        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception as e:
            print(f"  [SKIP] Could not open image {img_path.name}: {e}")
            skipped += 1
            continue

        process_label(label_src, label_dst, img_w, img_h, padding)
        total += 1

    print(f"\nDone!")
    print(f"  Labels inflated : {total}")
    print(f"  Labels skipped  : {skipped}")
    print(f"  Padding applied : {padding}px per side ({2*padding}px total per axis)")
    print(f"  Output          : {dst_root}")


def main():
    parser = argparse.ArgumentParser(description="Copy a YOLO dataset and inflate all bounding boxes.")
    parser.add_argument('--src', required=True, help="Source YOLO dataset root directory")
    parser.add_argument('--dst', required=True, help="Destination directory (will be created/overwritten)")
    parser.add_argument('--pad', type=int, default=DEFAULT_PADDING,
                        help=f"Pixels to add on each side of every box (default: {DEFAULT_PADDING})")
    args = parser.parse_args()

    copy_and_inflate(Path(args.src), Path(args.dst), args.pad)


if __name__ == '__main__':
    main()