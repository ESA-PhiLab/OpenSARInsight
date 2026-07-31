# type: ignore
"""
---------------------------------------------------------------------
Project: OpenSAR Insight
---------------------------------------------------------------------
visualize_yolo.py

Tool: Visualize YOLO dataset labels on images

Author: Abdulhameed Yunusa (ABHY)
E-mail: ayunusa@indracompany.com
Creation Date: 2026-02-26

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""
# type: ignore
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from dataset_generation_scripts.utils import get_config
cfg = get_config("MODEL_USECASES_PATH")

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


YOLO_DATASET_PATH = Path(cfg["visualize_yolo"]["dataset_path"])
OUTPUT_DIR        = Path(cfg["visualize_yolo"]["output_dir"])
SPLITS            = cfg["visualize_yolo"]["splits"]  # e.g. ["train", "val", "test"]
SAMPLES_PER_SPLIT = cfg["visualize_yolo"]["samples_per_split"] # e.g. 10


OUTPUT_DIR.mkdir(exist_ok=True)


def load_image_as_pil(img_path: Path) -> tuple:
    """
    Load PNG or TIFF as a PIL RGB image for drawing.
    For 4-channel TIFFs (C, H, W), displays the VV+VH sum channel (ch3).

    Returns:
        (PIL.Image in RGB mode, width, height)
    """
    ext = img_path.suffix.lower()

    if ext in ('.tif', '.tiff'):
        if not HAS_TIFFFILE:
            raise ImportError("tifffile not installed — run: pip install tifffile")
        arr = tifffile.imread(str(img_path))  # (C, H, W) uint8

        if arr.ndim == 3 and arr.shape[0] <= 4:
            # (C, H, W) → pick VV+VH sum channel (index 3) for best vessel visibility
            display = arr[3] if arr.shape[0] == 4 else arr[0]
        elif arr.ndim == 2:
            display = arr
        else:
            # (H, W, C) — already channel-last
            display = arr[:, :, 0]

        # Convert grayscale to RGB for coloured box drawing
        rgb = np.stack([display, display, display], axis=-1)
        im  = Image.fromarray(rgb.astype(np.uint8), mode='RGB')

    else:
        # PNG or any other PIL-supported format
        im = Image.open(img_path).convert('RGB')

    w, h = im.size
    return im, w, h


def draw_labels(im: Image.Image, label_path: Path, w: int, h: int) -> Image.Image:
    """Draw YOLO bounding boxes onto a PIL image."""
    draw = ImageDraw.Draw(im)
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x, y, bw, bh = map(float, parts)
            x_c   = x  * w
            y_c   = y  * h
            box_w = bw * w
            box_h = bh * h
            x0 = x_c - box_w / 2
            y0 = y_c - box_h / 2
            x1 = x_c + box_w / 2
            y1 = y_c + box_h / 2
            draw.rectangle([x0, y0, x1, y1], outline='red', width=2)
    return im


for split in SPLITS:
    images_dir = YOLO_DATASET_PATH / 'images' / split
    labels_dir = YOLO_DATASET_PATH / 'labels' / split

    # Collect both PNG and TIFF files
    image_files = list(images_dir.glob('*.png')) + \
                  list(images_dir.glob('*.tif')) + \
                  list(images_dir.glob('*.jpg')) + \
                  list(images_dir.glob('*.jpeg')) + \
                  list(images_dir.glob('*.tiff'))

    if not image_files:
        print(f'No images found for split: {split}')
        continue

    selected = random.sample(image_files, min(SAMPLES_PER_SPLIT, len(image_files)))

    for img_path in selected:
        label_path = labels_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            print(f'Label not found: {img_path.name}')
            continue

        try:
            im, w, h = load_image_as_pil(img_path)
            im = draw_labels(im, label_path, w, h)
            out_path = OUTPUT_DIR / f'{split}_{img_path.stem}_vis.png'
            im.save(out_path)
            print(f'Saved: {out_path}')
        except Exception as e:
            print(f'Error processing {img_path.name}: {e}')