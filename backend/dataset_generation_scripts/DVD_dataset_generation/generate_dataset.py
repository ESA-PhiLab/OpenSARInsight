from __future__ import annotations

import argparse
import csv
import os
import subprocess
import shutil
from pathlib import Path
from typing import Any
import yaml


# =========================
# Helpers
# =========================

def check_tool_exists(tool: str) -> bool:
    """Check if a command-line tool is available."""
    return shutil.which(tool) is not None


def run_logged(cmd: list[str], log_path: Path, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"      [CMD] {' '.join(cmd)}")
    print(f"      [LOG] {log_path.absolute()}")
    with log_path.open("w") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        subprocess.run(cmd, check=True, stdout=f, stderr=f, cwd=str(cwd) if cwd else None, env=env)


def expand_scene_entries(entries: Any) -> list[int] | None:
    """
    Same semantics as DVD run.py: entries can be ints or inclusive ranges "a-b".
    Returns unique ordered scene indices.
    """
    if entries is None:
        return None

    seen = set()
    ordered: list[int] = []

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
                a_str, b_str = s.split("-", 1)
                a = int(a_str.strip())
                b = int(b_str.strip())
                step = 1 if a <= b else -1
                v = a
                while True:
                    add(v)
                    if v == b:
                        break
                    v += step
            else:
                add(int(s))
        else:
            raise TypeError(f"Unsupported scenes_to_process entry type: {type(item)}")
    return ordered


def load_cfg(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    print(f"[INIT] Loading config from: {config_path}")
    cfg = yaml.safe_load(config_path.read_text()) or {}
    print(f"[INIT] Config loaded successfully")
    return cfg


def get_products_root(cfg: dict) -> Path:
    p = cfg.get("input", {}).get("products_root")
    if not p:
        raise ValueError("Missing input.products_root in config")
    root = Path(p)
    if not root.exists():
        raise NotADirectoryError(f"products_root does not exist: {root}")
    print(f"[INIT] Products root: {root}")
    return root


def get_output_root(cfg: dict) -> Path:
    p = cfg.get("output", {}).get("output_root")
    if not p:
        raise ValueError("Missing output.output_root in config")
    print(f"[INIT] Output root: {Path(p)}")
    return Path(p)


def get_scene_mapping(cfg: dict) -> Path:
    p = cfg.get("input", {}).get("scene_mapping")
    if not p:
        raise ValueError("Missing input.scene_mapping in config")
    mapping_path = Path(p)
    if not mapping_path.exists():
        raise FileNotFoundError(f"scene_mapping file not found: {mapping_path}")
    print(f"[INIT] Scene mapping: {mapping_path}")
    return mapping_path


def normalize_safe_name(name: str) -> str:
    """Remove .SAFE suffix if present to avoid double-suffix issues."""
    return name.replace(".SAFE", "")


def build_slc_safe_path(products_root: Path, slc_name: str) -> Path:
    slc_dir = products_root / "slc"
    name = normalize_safe_name(slc_name)
    return slc_dir / (name + ".SAFE")


def build_l0_decoded_path(products_root: Path, raw_name: str) -> Path:
    name = normalize_safe_name(raw_name)
    return products_root / "raw" / (name + ".SAFE")


# =========================
# Steps
# =========================

def step1_dvd(scene_num: int, cfg: dict, out_root: Path, dvd_root: Path, mapping_csv: Path) -> Path:
    """
    Runs DVD_Codes_Python/run.py for a single scene by generating a per-scene config.
    Produces: scene_XXX/{slc,grd,label,...}
    """
    print(f"\n  [STEP 1] DVD Processing")
    print(f"  ========================")
    
    scene_out = out_root / f"scene_{scene_num:03d}"
    done = scene_out / "01_step1_DONE.txt"
    log = scene_out / "01_step1_run.log"
    tmp_cfg = scene_out / f"_tmp_config_scene_{scene_num:03d}.yaml"

    if done.exists():
        print(f"  [SKIP] Step 1 already completed (found: {done})")
        return scene_out

    print(f"  [EXEC] Creating output directory: {scene_out}")
    scene_out.mkdir(parents=True, exist_ok=True)

    print(f"  [EXEC] Generating temporary config for scene {scene_num:03d}")
    cfg2 = dict(cfg)
    cfg2.setdefault("input", {})
    cfg2["input"]["scenes_to_process"] = [scene_num]
    tmp_cfg.write_text(yaml.safe_dump(cfg2, sort_keys=False))
    print(f"  [EXEC] Temp config written to: {tmp_cfg}")

    print(f"  [EXEC] Running DVD processing...")
    cmd = ["python", "run.py", "--config", str(tmp_cfg)]
    run_logged(cmd, log_path=log, cwd=dvd_root)

    # Cleanup temp config
    if tmp_cfg.exists():
        print(f"  [EXEC] Cleaning up temporary config")
        tmp_cfg.unlink()

    print(f"  [EXEC] Writing completion marker")
    done.write_text("ok\n")
    print(f"  [DONE] Step 1 completed successfully")
    return scene_out


def step2_slc2raw(scene_out: Path, label_folder: Path, slc_safe: Path) -> Path:
    """
    Runs:
      slc2raw label_folder product_folder -o rawH.npy -l traceability_log.txt
    Returns patch_info file path (rawH.npy).
    """
    print(f"\n  [STEP 2] SLC to RAW Conversion")
    print(f"  ==============================")
    
    out_dir = scene_out / "02_slc2raw"
    out_file = out_dir / "rawH.npy"
    done = out_dir / "_DONE.txt"
    log = out_dir / "02_step2_driver.log"
    tool_log = out_dir / "traceability_log.txt"

    if done.exists() and out_file.exists():
        print(f"  [SKIP] Step 2 already completed (found: {done})")
        print(f"  [INFO] Patch info file: {out_file}")
        return out_file

    print(f"  [EXEC] Creating output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [EXEC] Converting SLC patches to RAW format...")
    print(f"  [INFO] Label folder: {label_folder}")
    print(f"  [INFO] SLC SAFE: {slc_safe}")
    print(f"  [INFO] Output file: {out_file}")
    
    cmd = [
        "slc2raw",
        str(label_folder),
        str(slc_safe),
        "-o", str(out_file),
        "-l", str(tool_log),
    ]
    run_logged(cmd, log_path=log)

    print(f"  [EXEC] Writing completion marker")
    done.write_text("ok\n")
    print(f"  [DONE] Step 2 completed successfully")
    print(f"  [INFO] Patch info created: {out_file}")
    return out_file


def step3_l0patcher(scene_out: Path, l0_decoded: Path, patch_info: Path, base_name: str) -> Path:
    """
    Runs L0Patcher to extract RAW patches.
    """
    print(f"\n  [STEP 3] L0 RAW Patch Extraction")
    print(f"  =================================")
    
    out_dir = scene_out / "03_raw_patches"
    done = out_dir / "_DONE.txt"
    driver_log = out_dir / "03_step3_driver.log"
    tool_log = out_dir / "Logfile.log"

    # Check if already completed (full or partial success)
    if done.exists():
        status = done.read_text().strip()
        if status in ("ok", "partial"):
            dat_count = len(list(out_dir.glob("*.dat")))
            print(f"  [SKIP] Step 3 already completed (status: {status})")
            print(f"  [INFO] Found {dat_count} patch files")
            return out_dir

    print(f"  [EXEC] Creating output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [EXEC] Extracting L0 RAW patches...")
    print(f"  [INFO] L0 decoded: {l0_decoded}")
    print(f"  [INFO] Patch info: {patch_info}")
    print(f"  [INFO] Patch base name: {base_name}")
    print(f"  [INFO] Output directory: {out_dir}")
    
    cmd = [
        "L0Patcher",
        str(l0_decoded),
        str(patch_info),
        "-o", str(out_dir),
        "-l", str(tool_log),
        "-b", base_name,
    ]

    try:
        run_logged(cmd, log_path=driver_log)
        dat_count = len(list(out_dir.glob("*.dat")))
        print(f"  [EXEC] Writing completion marker")
        done.write_text("ok\n")
        print(f"  [DONE] Step 3 completed successfully")
        print(f"  [INFO] Created {dat_count} patch files")
        return out_dir

    except subprocess.CalledProcessError as e:
        # If patches were created, treat as partial success
        dat_files = list(out_dir.glob("*.dat"))
        if dat_files:
            print(f"  [WARN] L0Patcher exited with error (code={e.returncode})")
            print(f"  [WARN] But produced {len(dat_files)} .dat patches")
            msg = (
                f"L0Patcher exited non-zero (code={e.returncode}) but produced {len(dat_files)} .dat patches.\n"
                f"Driver log: {driver_log.absolute()}\n"
                f"Tool log: {tool_log.absolute()}\n"
            )
            (out_dir / "_PARTIAL_SUCCESS.txt").write_text(msg)
            done.write_text("partial\n")
            print(f"  [EXEC] Marked as partial success")
            print(f"  [INFO] See {out_dir / '_PARTIAL_SUCCESS.txt'} for details")
            return out_dir
        print(f"  [ERROR] L0Patcher failed and produced no patches")
        raise


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset pipeline: DVD → SLC2RAW → L0Patcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config.yaml in script directory
  python generate_dataset.py

  # Use custom config
  python generate_dataset.py --config /path/to/config.yaml
        """
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config.yaml (default: ./config.yaml)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("DATASET GENERATION PIPELINE")
    print("="*70)
    
    # Determine script directory
    script_dir = Path(__file__).resolve().parent
    print(f"\n[INIT] Script directory: {script_dir}")
    
    # Set default config relative to script location
    config_path = args.config if args.config else script_dir / "config.yaml"
    
    # Validate config exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Validate environment
    print(f"\n[INIT] Checking required tools...")
    tools_ok = True
    for tool in ["slc2raw", "L0Patcher"]:
        if check_tool_exists(tool):
            print(f"[INIT] ✓ {tool} found")
        else:
            print(f"[INIT] ✗ {tool} NOT FOUND")
            tools_ok = False
    
    if not tools_ok:
        raise EnvironmentError("Required tools not found on PATH")

    print(f"\n[INIT] Loading configuration...")
    cfg = load_cfg(config_path)
    products_root = get_products_root(cfg)
    out_root = get_output_root(cfg)
    mapping_csv = get_scene_mapping(cfg)
    
    print(f"\n[INIT] Creating output directory...")
    out_root.mkdir(parents=True, exist_ok=True)

    # Read wanted scenes from config
    print(f"\n[INIT] Determining scenes to process...")
    scenes_cfg = cfg.get("input", {}).get("scenes_to_process", None)
    wanted_scenes = expand_scene_entries(scenes_cfg)

    # Load mapping.csv into dict by num
    print(f"[INIT] Loading scene mapping...")
    mapping_by_num: dict[int, dict[str, str]] = {}
    with mapping_csv.open(newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            mapping_by_num[int(row["num"])] = row
    print(f"[INIT] Loaded {len(mapping_by_num)} scenes from mapping file")

    # If config doesn't specify scenes -> default to all
    if not wanted_scenes:
        wanted_scenes = sorted(mapping_by_num.keys())
        print(f"[INIT] No scenes specified in config, processing all {len(wanted_scenes)} scenes")
    else:
        print(f"[INIT] Processing {len(wanted_scenes)} scenes: {wanted_scenes}")

    print(f"\n{'='*70}")
    print("CONFIGURATION SUMMARY")
    print(f"{'='*70}")
    print(f"Config file    : {config_path.absolute()}")
    print(f"Scene mapping  : {mapping_csv.absolute()}")
    print(f"DVD root       : {script_dir}")
    print(f"Products root  : {products_root}")
    print(f"Output root    : {out_root}")
    print(f"Scenes         : {wanted_scenes}")
    print(f"{'='*70}")

    failed_scenes = []

    for idx, scene_num in enumerate(wanted_scenes, 1):
        print(f"\n{'#'*70}")
        print(f"# SCENE {scene_num:03d} ({idx}/{len(wanted_scenes)})")
        print(f"{'#'*70}")
        
        row = mapping_by_num.get(scene_num)
        if row is None:
            print(f"[WARN] Scene {scene_num:03d} not found in scene_mapping.csv. Skipping.")
            failed_scenes.append((scene_num, "Not in mapping file"))
            continue

        slc_name = row["slc"].strip()
        raw_name = row.get("raw", "").strip()

        if not slc_name:
            print(f"[WARN] Scene {scene_num:03d} has no SLC name. Skipping.")
            failed_scenes.append((scene_num, "Missing SLC name"))
            continue

        if not raw_name:
            print(f"[WARN] Scene {scene_num:03d} has no RAW name. Skipping.")
            failed_scenes.append((scene_num, "Missing RAW name"))
            continue

        # Extract base name from SLC product (remove .SAFE suffix)
        patch_base_name = normalize_safe_name(slc_name)
        slc_safe = build_slc_safe_path(products_root, slc_name)
        l0_decoded = build_l0_decoded_path(products_root, raw_name)

        print(f"\n[INFO] Scene Details:")
        print(f"  SLC name   : {slc_name}")
        print(f"  RAW name   : {raw_name}")
        print(f"  SLC path   : {slc_safe}")
        print(f"  L0 path    : {l0_decoded}")
        print(f"  Patch base : {patch_base_name}")

        # Validate inputs exist
        print(f"\n[CHECK] Validating input products...")
        validation_failed = False
        
        if not slc_safe.exists():
            print(f"[FAIL] ✗ SLC SAFE not found: {slc_safe}")
            failed_scenes.append((scene_num, "SLC SAFE missing"))
            validation_failed = True
        else:
            print(f"[CHECK] ✓ SLC SAFE exists")

        if not l0_decoded.exists():
            print(f"[FAIL] ✗ L0 decoded not found: {l0_decoded}")
            failed_scenes.append((scene_num, "L0 decoded missing"))
            validation_failed = True
        else:
            print(f"[CHECK] ✓ L0 decoded exists")
        
        if validation_failed:
            print(f"[FAIL] Validation failed, skipping scene {scene_num:03d}")
            continue

        # Step 1: DVD processing
        try:
            scene_out = step1_dvd(scene_num, cfg, out_root, script_dir, mapping_csv)
        except subprocess.CalledProcessError as e:
            log_path = out_root / f"scene_{scene_num:03d}" / "01_step1_run.log"
            print(f"\n[FAIL] Step 1 failed with exit code {e.returncode}")
            print(f"[FAIL] Check log: {log_path.absolute()}")
            failed_scenes.append((scene_num, "step1 failed"))
            continue
        except Exception as e:
            print(f"\n[FAIL] Step 1 failed with exception: {e}")
            failed_scenes.append((scene_num, f"step1 error: {e}"))
            continue

        # Label folder should be created by DVD
        label_folder = scene_out / "label"
        print(f"\n[CHECK] Verifying DVD outputs...")
        if not label_folder.exists():
            print(f"[FAIL] ✗ Label folder not found: {label_folder}")
            failed_scenes.append((scene_num, "label folder missing"))
            continue
        else:
            label_count = len(list(label_folder.glob("*.xml")))
            print(f"[CHECK] ✓ Label folder exists ({label_count} XML files)")

        # Step 2: SLC to RAW conversion
        try:
            patch_info = step2_slc2raw(scene_out, label_folder, slc_safe)
        except subprocess.CalledProcessError as e:
            log_path = scene_out / "02_slc2raw" / "02_step2_driver.log"
            print(f"\n[FAIL] Step 2 failed with exit code {e.returncode}")
            print(f"[FAIL] Check log: {log_path.absolute()}")
            failed_scenes.append((scene_num, "step2 failed"))
            continue
        except Exception as e:
            print(f"\n[FAIL] Step 2 failed with exception: {e}")
            failed_scenes.append((scene_num, f"step2 error: {e}"))
            continue

        # Step 3: L0 patching
        try:
            out_patches = step3_l0patcher(scene_out, l0_decoded, patch_info, patch_base_name)
            dat_count = len(list(out_patches.glob("*.dat")))
            xml_count = len(list(out_patches.glob("*.xml")))
            print(f"\n[SUCCESS] Scene {scene_num:03d} completed!")
            print(f"[SUCCESS] Output: {out_patches}")
            print(f"[SUCCESS] Created {dat_count} .dat files and {xml_count} .xml files")
        except subprocess.CalledProcessError as e:
            log_path = scene_out / "03_raw_patches" / "03_step3_driver.log"
            print(f"\n[FAIL] Step 3 failed with exit code {e.returncode}")
            print(f"[FAIL] Check log: {log_path.absolute()}")
            failed_scenes.append((scene_num, "step3 failed"))
            continue
        except Exception as e:
            print(f"\n[FAIL] Step 3 failed with exception: {e}")
            failed_scenes.append((scene_num, f"step3 error: {e}"))
            continue

    # Summary
    print(f"\n{'='*70}")
    print("PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total scenes requested : {len(wanted_scenes)}")
    print(f"Successfully completed : {len(wanted_scenes) - len(failed_scenes)}")
    print(f"Failed                 : {len(failed_scenes)}")

    if failed_scenes:
        print(f"\n{'='*70}")
        print("FAILED SCENES")
        print(f"{'='*70}")
        for num, reason in failed_scenes:
            print(f"  Scene {num:03d}: {reason}")
    
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETED")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()