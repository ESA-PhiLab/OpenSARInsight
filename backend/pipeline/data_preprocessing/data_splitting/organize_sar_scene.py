import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# STRICT matcher for tiles: requires "...-<scene_id>-...FD_<tile_id>_scaled.npy"
TILE_REGEX = re.compile(r"-([0-9a-fA-F]{6,8})-.*?FD_(\d+)_scaled\.npy$", re.IGNORECASE)
# Lenient matcher for masks: just needs scene_id and tile_id; ext can vary
MASK_NAME_REGEX = re.compile(r"-([0-9a-fA-F]{6,8})-.*?FD_(\d+)", re.IGNORECASE)

MASK_EXTS = {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_scene_tile(filename: str) -> Optional[Tuple[str, str]]:
    m = TILE_REGEX.search(filename)
    if not m:
        return None
    scene_id, tile_id = m.group(1).lower(), m.group(2)
    return scene_id, tile_id


def parse_scene_tile_from_mask_name(filename: str) -> Optional[Tuple[str, str]]:
    m = MASK_NAME_REGEX.search(filename)
    if not m:
        return None
    scene_id, tile_id = m.group(1).lower(), m.group(2)
    return scene_id, tile_id


def get_polarization(name: str) -> str:
    s = name.lower()
    if "-vv-" in s:
        return "vv"
    if "-vh-" in s:
        return "vh"
    return "unknown"


def count_white_pixels(mask_path: Path) -> Tuple[int, int, float]:
    """Count flood pixels (white) in a mask. Accept 255 or 1 as white; use whichever dominates."""
    ext = mask_path.suffix.lower()
    if ext == ".npy":
        arr = np.load(mask_path)
    else:
        if not PIL_AVAILABLE:
            raise RuntimeError(f"Pillow not available to read non-NPY mask: {mask_path.name}")
        from PIL import Image
        with Image.open(mask_path) as im:
            arr = np.array(im.convert("L"))
    total = int(arr.size)
    white_255 = int((arr == 255).sum())
    white_1 = int((arr == 1).sum())
    white = white_255 if white_255 >= white_1 else white_1
    ratio = (white / total) if total > 0 else 0.0
    return white, total, ratio


def find_mask_dirs_for_split(split_dir: Path, mask_subdir_name: str) -> List[Path]:
    """Return possible mask directories inside a split (handle naming variants)."""
    candidates: List[Path] = []
    exact = split_dir / mask_subdir_name
    if exact.is_dir():
        candidates.append(exact)
    for cand in ["masks", "Mask", "MASK", "mask"]:
        p = split_dir / cand
        if p.is_dir() and p not in candidates:
            candidates.append(p)
    for p in split_dir.iterdir():
        if p.is_dir() and "mask" in p.name.lower() and p not in candidates:
            candidates.append(p)
    return candidates


def build_global_mask_index_multi(
    root: Path, mask_subdir_name: str, verbose: bool
) -> Dict[Tuple[str, str], List[Path]]:
    """
    Build an index of ALL masks across train/val/test.
    Returns: dict keyed by (scene_id, tile_id) -> list of mask Paths (may include both vv & vh).
    """
    index: Dict[Tuple[str, str], List[Path]] = {}
    for split_name in ["train", "val", "test"]:
        split = root / split_name
        if not split.is_dir():
            continue
        for mask_dir in find_mask_dirs_for_split(split, mask_subdir_name):
            for p in mask_dir.rglob("*"):
                if not p.is_file() or p.suffix.lower() not in MASK_EXTS:
                    continue
                parsed = parse_scene_tile_from_mask_name(p.name)
                if not parsed:
                    continue
                key = parsed
                index.setdefault(key, []).append(p)
    if verbose:
        total_masks = sum(len(v) for v in index.values())
        print(f"[INFO] Indexed {total_masks} mask files across all splits "
              f"({len(index)} unique (scene_id, tile_id) keys).")
    return index


def collect_tiles(
    root: Path,
    out_root: Path,
    move: bool = False,
    verbose: bool = False,
    mask_subdir_name: str = "mask",
    dry_run: bool = False,
) -> None:
    split_dirs = [root / "train", root / "val", root / "test"]
    out_root.mkdir(parents=True, exist_ok=True)

    # scenes[scene_id][tile_id] -> list of tile Paths (VV & VH mixed together)
    scenes: Dict[str, Dict[str, List[Path]]] = {}
    total_files_scanned = 0
    total_tiles_matched = 0

    # 1) Build a global mask index (multi: may contain vv & vh)
    mask_index = build_global_mask_index_multi(root, mask_subdir_name, verbose)

    # 2) Discover tiles, excluding anything under mask folders
    for split in split_dirs:
        if not split.is_dir():
            if verbose:
                print(f"[WARN] Missing split dir: {split}")
            continue
        for npy in split.rglob("*.npy"):
            if any("mask" in part.lower() for part in npy.parts):
                continue
            total_files_scanned += 1
            parsed = parse_scene_tile(npy.name)
            if parsed:
                scene_id, tile_id = parsed
                scenes.setdefault(scene_id, {}).setdefault(tile_id, []).append(npy)
                total_tiles_matched += 1
            elif verbose:
                print(f"[SKIP] No match for {npy.name}")

    print(f"[INFO] Scanned {total_files_scanned} .npy tiles (excluding masks); "
          f"matched {total_tiles_matched} tile files across {len(scenes)} scenes.")
    if not scenes:
        print("[ERROR] No tiles matched the expected filename pattern.")
        return

    if dry_run:
        shown = 0
        for scene_id, tile_map in scenes.items():
            n_files = sum(len(v) for v in tile_map.values())
            n_tiles = len(tile_map)
            masks_available = sum(1 for t in tile_map if (scene_id, t) in mask_index)
            print(f"[DRY-RUN] Scene {scene_id}: {n_files} tile files, {n_tiles} unique tile_ids, "
                  f"{masks_available} tile_ids have ≥1 mask")
            shown += 1
            if shown >= 10:
                break
        print("[DRY-RUN] Stopping before any file operations.")
        return

    # 3) Copy tiles & all available masks (vv/vh) per tile_id
    for scene_id, tile_map in scenes.items():
        scene_dir = out_root / scene_id
        tiles_dir = scene_dir / "tiles"
        masks_dir = scene_dir / "masks"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            total_files = sum(len(v) for v in tile_map.values())
            print(f"[SCENE] {scene_id}: {total_files} tile files, {len(tile_map)} unique tile_ids")

        stats_csv = scene_dir / "mask_stats.csv"
        scene_white_total = 0
        scene_pixels_total = 0
        masks_copied_total = 0

        with stats_csv.open("w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["mask_file", "tile_id", "pol", "white_pixels", "total_pixels", "white_ratio"])

            for tile_id, file_list in tile_map.items():
                # Copy/move all tile files
                for tile_path in file_list:
                    dst_tile = tiles_dir / tile_path.name
                    if move:
                        if verbose:
                            print(f"  [MOVE] {tile_path} -> {dst_tile}")
                        shutil.move(str(tile_path), str(dst_tile))
                    else:
                        if not dst_tile.exists():
                            if verbose:
                                print(f"  [COPY] {tile_path} -> {dst_tile}")
                            shutil.copy2(str(tile_path), str(dst_tile))

                # All masks available for this (scene_id, tile_id)
                key = (scene_id, tile_id)
                mask_paths = mask_index.get(key, [])

                if not mask_paths:
                    print(f"[WARN] No masks found anywhere for scene={scene_id}, tile={tile_id}")
                    continue

                # Deduplicate by polarization: keep at most one vv and one vh; keep one 'unknown' if any
                pol_choice: Dict[str, Path] = {}
                for p in mask_paths:
                    pol = get_polarization(p.name)
                    if pol not in pol_choice:
                        pol_choice[pol] = p
                # Copy and count each chosen mask
                for pol, src_mask in pol_choice.items():
                    dst_mask = masks_dir / src_mask.name
                    if not dst_mask.exists():
                        if verbose:
                            print(f"  [COPY MASK] ({pol}) {src_mask} -> {dst_mask}")
                        shutil.copy2(str(src_mask), str(dst_mask))
                    try:
                        white, total, ratio = count_white_pixels(dst_mask)
                        writer.writerow([dst_mask.name, tile_id, pol, white, total, f"{ratio:.6f}"])
                        scene_white_total += white
                        scene_pixels_total += total
                        masks_copied_total += 1
                    except Exception as e:
                        print(f"[ERROR] Counting white pixels for {dst_mask.name}: {e}")

        # Summary
        with (scene_dir / "summary.txt").open("w") as fs:
            fs.write(f"Scene: {scene_id}\n")
            fs.write(f"Tile files: {sum(len(v) for v in tile_map.values())}\n")
            fs.write(f"Unique tile_ids: {len(tile_map)}\n")
            fs.write(f"Masks copied (vv/vh/unknown): {masks_copied_total}\n")
            if scene_pixels_total > 0:
                fs.write(f"Total white pixels: {scene_white_total}\n")
                fs.write(f"Total pixels: {scene_pixels_total}\n")
                fs.write(f"Overall white ratio: {scene_white_total/scene_pixels_total:.6f}\n")
            else:
                fs.write("No mask pixels counted.\n")

        if verbose:
            print(f"[DONE] {scene_id} -> {scene_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Group SAR tiles by scene, copy tiles into one folder, include BOTH vv & vh masks per tile, and compute per-mask flood-pixel counts."
    )
    parser.add_argument("--root", type=Path, required=True,
                        help="Dataset root with train/, val/, test/ (each with a mask/ subfolder).")
    parser.add_argument("--out", type=Path, default=Path("scenes_out"),
                        help="Output root to create per-scene folders.")
    parser.add_argument("--move", action="store_true",
                        help="Move tiles instead of copying them.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print extra progress information.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen, without copying/moving.")
    parser.add_argument("--mask-subdir", type=str, default="mask",
                        help="Name of the mask subfolder inside each split (default: 'mask').")
    args = parser.parse_args()

    collect_tiles(
        root=args.root,
        out_root=args.out,
        move=args.move,
        verbose=args.verbose,
        mask_subdir_name=args.mask_subdir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
