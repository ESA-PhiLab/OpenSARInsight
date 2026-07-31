import os
import sys
import argparse
import yaml
import pandas as pd
from main import main
import glob
from pathlib import Path


def _normpath(p: str) -> str:
    """Normalize and expand path."""
    if not p:
        return p
    return str(Path(p).expanduser().resolve())


def _expand_scene_entries(entries):
    """Expand scene entries (integers or ranges like '5-10')."""
    if entries is None:
        return None

    seen = set()
    ordered = []

    def add(v: int):
        if v not in seen:
            seen.add(v)
            ordered.append(v)

    for item in entries:
        if isinstance(item, int):
            add(item)
        elif isinstance(item, str):
            s = item.strip()
            if "-" in s:
                try:
                    a_str, b_str = s.split("-", 1)
                    a = int(a_str.strip())
                    b = int(b_str.strip())
                    step = 1 if a <= b else -1
                    for v in range(a, b + step, step):
                        add(v)
                except ValueError:
                    raise ValueError(f"Invalid range entry in scenes_to_process: {item}")
            else:
                try:
                    add(int(s))
                except ValueError:
                    raise ValueError(f"Invalid scene entry in scenes_to_process: {item}")
        else:
            raise TypeError(f"Unsupported scenes_to_process entry type: {type(item)}")
    return ordered


def _validate_mapping_df(df: pd.DataFrame):
    """Validate that mapping CSV has required columns."""
    required = ["num", "grd", "slc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Mapping CSV missing required columns: {missing}")


def _find_safe_folder(base_dir, product_name):
    """
    Find a Sentinel-1 SAFE folder matching product_name.
    
    Args:
        base_dir (str or Path): Directory to search in
        product_name (str): Product name with or without .SAFE extension
        
    Returns:
        str or None: Full path to SAFE folder, or None if not found
    """
    base_dir = Path(base_dir)
    
    # Remove .SAFE if present
    product_base = product_name.replace(".SAFE", "")
    
    # Try exact match first
    exact = base_dir / f"{product_base}.SAFE"
    if exact.is_dir():
        return str(exact)
    
    # Try glob pattern (handles wildcards in product name)
    pattern = f"{product_base}*.SAFE"
    matches = list(base_dir.glob(pattern))
    
    if len(matches) == 1:
        return str(matches[0])
    elif len(matches) > 1:
        # Return the shortest match (most specific)
        shortest = min(matches, key=lambda p: len(p.name))
        print(f"[WARN] Multiple SAFE folders match '{product_name}': {[m.name for m in matches]}")
        print(f"[WARN] Using shortest match: {shortest.name}")
        return str(shortest)
    
    return None


def _select_rows_by_indices(df: pd.DataFrame, indices):
    """Select and order rows by scene indices."""
    if not indices:
        return df.copy()
    
    dfn = df.copy()
    dfn["num"] = pd.to_numeric(dfn["num"], errors="coerce").astype("Int64")
    wanted = set(int(i) for i in indices)
    out = dfn[dfn["num"].isin(wanted)].copy()
    
    # Preserve order from input indices
    order_map = {v: i for i, v in enumerate(indices)}
    out["__order__"] = out["num"].map(lambda v: order_map.get(int(v), 10**9))
    out = out.sort_values("__order__").drop(columns="__order__").reset_index(drop=True)
    
    return out


def main_launcher():
    """Main entry point for RFI processing pipeline."""
    parser = argparse.ArgumentParser(
        description="Run RFI pipeline for selected scenes from config.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config.yaml
  python run.py
  
  # Use custom config
  python run.py --config custom_config.yaml
  
  # Dry run to see what would be processed
  python run.py --dry-run
        """
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (default: config.yaml)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be executed, don't run processing"
    )
    args = parser.parse_args()

    # Determine script directory for relative path resolution
    script_dir = Path(__file__).resolve().parent
    
    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path
    
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"[CONFIG] Loading config from: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg_input = cfg.get("input", {})
    cfg_output = cfg.get("output", {})

    # Get directory names (with defaults)
    slc_dirname = cfg_input.get("slc_dirname", "slc")
    grd_dirname = cfg_input.get("grd_dirname", "grd")
    bin_dirname = cfg_input.get("binarymask_dirname", "binarymask")

    # Get input/output roots
    input_root_str = cfg_input.get("input_root")
    output_root_str = cfg_output.get("output_root")
    
    if not input_root_str:
        raise ValueError("Missing 'input.input_root' in config")
    if not output_root_str:
        raise ValueError("Missing 'output.output_root' in config")
    
    # Resolve paths (make absolute if relative)
    input_root = Path(input_root_str)
    if not input_root.is_absolute():
        input_root = script_dir / input_root
    
    output_root = Path(output_root_str)
    if not output_root.is_absolute():
        output_root = script_dir / output_root
    
    # Validate input_root exists
    if not input_root.exists():
        raise NotADirectoryError(f"input_root does not exist: {input_root}")

    print(f"[CONFIG] input_root  = {input_root}")
    print(f"[CONFIG] output_root = {output_root}")
    print(f"[CONFIG] Subdirectories: {slc_dirname}/, {grd_dirname}/, {bin_dirname}/")

    # Get mapping CSV path
    mapping_rel = cfg_input.get("scene_mapping") or cfg_input.get("mapping_csv", "scene_mapping.csv")
    mapping_path = Path(mapping_rel)
    
    if not mapping_path.is_absolute():
        mapping_path = script_dir / mapping_path
    
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_path}")

    print(f"[CONFIG] scene_mapping = {mapping_path}")

    # Read mapping CSV with BOM handling
    df = pd.read_csv(mapping_path, sep=';', dtype=str, encoding='utf-8-sig')
    df["num"] = df["num"].astype(int)
    _validate_mapping_df(df)

    # Determine which scenes to process
    scenes_cfg = cfg_input.get("scenes_to_process", None)
    indices = _expand_scene_entries(scenes_cfg)
    
    if indices is None:
        df_sel = df
        print(f"[INFO] Processing all {len(df_sel)} scenes from mapping CSV")
    else:
        df_sel = _select_rows_by_indices(df, indices)
        print(f"[INFO] Processing {len(df_sel)} scenes: {sorted(indices)}")
    
    if df_sel.empty:
        raise ValueError("No scenes selected for processing")

    # Validate subdirectories
    input_slc_dir = input_root / slc_dirname
    input_grd_dir = input_root / grd_dirname
    input_bin_dir = input_root / bin_dirname

    print(f"\n[VALIDATE] Checking input directories...")
    dirs_ok = True
    for name, path in [
        ("SLC", input_slc_dir),
        ("GRD", input_grd_dir),
        ("Binary mask", input_bin_dir)
    ]:
        if path.is_dir():
            count = len(list(path.glob("*.SAFE")))
            print(f"  ✓ {name:12s}: {path} ({count} SAFE folders)")
        else:
            print(f"  ✗ {name:12s}: {path} (NOT FOUND)")
            if name in ["SLC", "GRD"]:
                dirs_ok = False
    
    if not dirs_ok:
        raise NotADirectoryError("Required input directories missing")

    # Create output root
    output_root.mkdir(parents=True, exist_ok=True)

    # Process each scene
    print(f"\n{'='*70}")
    print("PROCESSING SCENES")
    print(f"{'='*70}")
    
    success_count = 0
    failed_scenes = []

    for idx, row in df_sel.iterrows():
        num = int(row["num"])
        grd_name = str(row["grd"]).strip()
        slc_name = str(row["slc"]).strip()

        scene_out = output_root / f"scene_{num:03d}"
        
        print(f"\n[SCENE {num:03d}] ({idx + 1}/{len(df_sel)})")
        print(f"  SLC: {slc_name}")
        print(f"  GRD: {grd_name}")
        print(f"  Output: {scene_out}")

        # Find actual SAFE folders
        slc_safe = _find_safe_folder(input_slc_dir, slc_name)
        grd_safe = _find_safe_folder(input_grd_dir, grd_name)

        # Validate products exist
        missing = []
        if not slc_safe or not Path(slc_safe).is_dir():
            missing.append("SLC")
            print(f"  ✗ SLC SAFE not found in {input_slc_dir}")
        else:
            print(f"  ✓ SLC: {Path(slc_safe).name}")
            
        if not grd_safe or not Path(grd_safe).is_dir():
            missing.append("GRD")
            print(f"  ✗ GRD SAFE not found in {input_grd_dir}")
        else:
            print(f"  ✓ GRD: {Path(grd_safe).name}")

        if missing:
            print(f"  ✗ SKIPPED (missing: {', '.join(missing)})")
            failed_scenes.append((num, f"Missing {', '.join(missing)}"))
            continue

        if args.dry_run:
            print(f"  [DRY-RUN] Would process this scene")
            continue

        # Create scene output directory
        scene_out.mkdir(parents=True, exist_ok=True)

        try:
            # Call main() with product names (not full paths)
            # main() will use inputpath + subdirectory structure
            main(
                inputpath=str(input_root),
                grdproduct=grd_name.replace(".SAFE", ""),
                slcproduct=slc_name.replace(".SAFE", ""),
                outputpath=str(scene_out)
            )
            print(f"  ✓ SUCCESS")
            success_count += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed_scenes.append((num, str(e)))

    # Summary
    print(f"\n{'='*70}")
    print("PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total scenes   : {len(df_sel)}")
    print(f"Successful     : {success_count}")
    print(f"Failed         : {len(failed_scenes)}")
    
    if args.dry_run:
        print("\n(DRY RUN - no actual processing performed)")
    
    if failed_scenes:
        print(f"\nFailed scenes:")
        for num, reason in failed_scenes:
            print(f"  - Scene {num:03d}: {reason}")
    
    print("\n[DONE] Processing complete")
    
    return 0 if not failed_scenes else 1


if __name__ == "__main__":
    try:
        sys.exit(main_launcher())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Stopped by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)