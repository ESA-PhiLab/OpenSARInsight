"""
Group files by 'stem before the final _ID' and move/copy them into per-stem folders.

Definition
----------
Given a filename like:
    s1b-iw-raw-s-vv-20170311t234307-20170311t234339-004667-008247-IW2-FD_42041.npy
The script defines the STEM as:
    s1b-iw-raw-s-vv-20170311t234307-20170311t234339-004667-008247-IW2-FD
i.e., filename without extension, with the last '_' + trailing token removed.

What it does
------------
1) Scans a directory (optionally recursively) and discovers all unique stems.
2) For each stem, collects all files whose base name starts with f"{stem}_".
3) Moves or copies those files into a destination directory under:
       <dst>/<stem>/
4) Optionally match only specific extensions, or ANY extension.

Usage
-----
Basic (move .npy files):
  python group_all_stems.py --src /data/in --dst /data/grouped --ext .npy --action move

Recursive, any extension, copy, dry-run first:
  python group_all_stems.py --src /data/in --dst /data/grouped --ext ANY --recursive --action copy --dry-run

Only create folders for stems with >= 2 files:
  python group_all_stems.py --src /data/in --dst /data/grouped --min-files 2

Outputs
-------
- A folder per stem inside --dst, containing the stem’s files.
- A summary CSV (optional) mapping each file to its stem if --summary is provided.

Notes
-----
- Files without at least one '_' in their *name without extension* are skipped (unless --loose is used).
- Overwrite behaviour is controlled by --overwrite.
"""

from __future__ import annotations
import argparse
import csv
import logging
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


logger = logging.getLogger("group_all_stems")


# --------------------------- Helpers ---------------------------

def setup_logging(verbosity: int) -> None:
    """Configure root logger."""
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def extract_stem_from_name(name: str, *, loose: bool = False) -> Optional[str]:
    """
    Extract the grouping stem from a filename (no directories).
    Removes the extension, then removes the last '_' + trailing token.

    Returns None if the name has no '_' (unless loose=True, in which case
    the whole name-without-ext is returned).

    Examples
    --------
    'foo_bar_123.npy' -> 'foo_bar'
    'alpha_beta.gamma_42.npz' -> 'alpha_beta.gamma'  (only last extension is removed)
    'no_underscore.ext' -> 'no' (if loose=True) else None
    """
    p = Path(name)
    base = p.stem  # removes only the last extension, keeps the rest intact
    if "_" not in base:
        return base if loose else None
    return base.rsplit("_", 1)[0]


def iter_files(src: Path, recursive: bool) -> Iterable[Path]:
    """Yield files in src (optionally recursively)."""
    if recursive:
        yield from (p for p in src.rglob("*") if p.is_file())
    else:
        yield from (p for p in src.iterdir() if p.is_file())


def matches_extension(path: Path, ext: str) -> bool:
    """Return True if path matches extension filter (case-insensitive), or ANY."""
    if ext.upper() == "ANY":
        return True
    return path.suffix.lower() == ext.lower()


def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def transfer_file(src: Path, dst: Path, action: str, overwrite: bool, dry_run: bool) -> bool:
    """
    Move or copy a single file. Returns True if a transfer would occur/occurred.
    Respects overwrite and dry_run flags.
    """
    if dst.exists():
        if not overwrite:
            logger.debug(f"Skip existing (no overwrite): {dst}")
            return False
        if not dry_run:
            dst.unlink()

    if dry_run:
        logger.info(f"[DRY-RUN] {action.upper()}: {src} -> {dst}")
        return True

    if action == "move":
        shutil.move(str(src), str(dst))
    elif action == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown action: {action}")
    return True


# --------------------------- Core logic ---------------------------

def discover_stems(
    src: Path,
    ext: str,
    recursive: bool,
    loose: bool,
) -> Dict[str, List[Path]]:
    """
    Scan `src` and build a mapping: stem -> list of files that start with '{stem}_' (base name).
    Only includes files matching `ext` (or ANY).

    This uses the *actual* rule to discover candidates: we first try to extract a stem from each
    file name; then we add *all* files whose base name starts with that stem + '_' and match the
    extension. This ensures related variants (e.g., multiple IDs) are grouped together.
    """
    # First pass: get all candidate stems
    stems: Dict[str, List[Path]] = {}
    files = [p for p in iter_files(src, recursive=recursive) if matches_extension(p, ext)]
    logger.info(f"Scanned {len(files)} file(s) matching ext filter.")

    candidate_stems: Dict[str, None] = {}
    for p in files:
        stem = extract_stem_from_name(p.name, loose=loose)
        if stem:
            candidate_stems[stem] = None

    logger.info(f"Discovered {len(candidate_stems)} unique stem(s).")

    # Second pass: assign files to stems by prefix rule
    for stem in candidate_stems.keys():
        prefix = stem + "_"
        grouped = [p for p in files if p.stem.startswith(prefix)]
        if grouped:
            stems[stem] = grouped

    return stems


def write_summary(summary_csv: Path, mapping: Dict[str, List[Path]], dst_root: Path) -> None:
    """Write a CSV summary: stem, file_path, destination_folder."""
    ensure_dir(summary_csv.parent)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "file", "destination"])
        for stem, paths in sorted(mapping.items()):
            dest = dst_root / stem
            for p in paths:
                w.writerow([stem, str(p), str(dest)])


def group_transfer(
    src: Path,
    dst: Path,
    ext: str,
    action: str,
    overwrite: bool,
    recursive: bool,
    min_files: int,
    loose: bool,
    dry_run: bool,
    summary_csv: Optional[Path],
) -> None:
    """Discover stems and move/copy files into <dst>/<stem>/."""
    stems = discover_stems(src=src, ext=ext, recursive=recursive, loose=loose)

    if min_files > 1:
        stems = {k: v for k, v in stems.items() if len(v) >= min_files}
        logger.info(f"{len(stems)} stem(s) remain after min-files>={min_files} filter.")

    if not stems:
        logger.warning("No stems to process.")
        return

    ensure_dir(dst)

    if summary_csv:
        write_summary(summary_csv, stems, dst)

    total_files = 0
    transferred = 0

    for stem, paths in sorted(stems.items()):
        target_dir = dst / stem
        ensure_dir(target_dir)
        logger.info(f"{action.capitalize()} {len(paths)} file(s) -> {target_dir}")

        for p in paths:
            total_files += 1
            out_path = target_dir / p.name
            if transfer_file(p, out_path, action=action, overwrite=overwrite, dry_run=dry_run):
                transferred += 1

    logger.info(
        f"Done. Unique stems: {len(stems)} | Files considered: {total_files} | "
        f"Files {action}d: {transferred}{' (DRY-RUN)' if dry_run else ''}."
    )


# --------------------------- CLI ---------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Group files by 'stem before the final _ID' and move/copy into per-stem folders."
    )
    parser.add_argument("--src", type=Path, required=True, help="Source directory to scan.")
    parser.add_argument("--dst", type=Path, required=True, help="Destination root where per-stem folders are created.")
    parser.add_argument(
        "--ext",
        type=str,
        default=".npy",
        help="Extension filter (e.g., .npy). Use 'ANY' to accept any extension. Default: .npy",
    )
    parser.add_argument(
        "--action",
        choices=["move", "copy"],
        default="move",
        help="Move or copy files into destination. Default: move",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in destination folders.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories of --src.",
    )
    parser.add_argument(
        "--min-files",
        type=int,
        default=1,
        help="Only create a per-stem folder if the stem has at least this many files. Default: 1",
    )
    parser.add_argument(
        "--loose",
        action="store_true",
        help="If a filename lacks '_', use the whole name (without extension) as the stem instead of skipping.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional path to write a CSV summary mapping files to stems.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without moving/copying any files.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (-v for INFO, -vv for DEBUG). Default: INFO",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    if not args.src.is_dir():
        raise NotADirectoryError(f"--src not found or not a directory: {args.src}")

    group_transfer(
        src=args.src,
        dst=args.dst,
        ext=args.ext,
        action=args.action,
        overwrite=args.overwrite,
        recursive=args.recursive,
        min_files=args.min_files,
        loose=args.loose,
        dry_run=args.dry_run,
        summary_csv=args.summary,
    )


if __name__ == "__main__":
    main()
