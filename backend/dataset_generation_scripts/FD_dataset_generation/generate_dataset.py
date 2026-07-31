from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path
import sys
from typing import Any

import yaml


# ============================================================
# Helpers
# ============================================================

def check_tool_exists(tool: str) -> bool:
    """Check if a command-line tool is available."""
    return shutil.which(tool) is not None


def run_logged(
    cmd: list[str],
    log_path: Path,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"      [CMD] {' '.join(cmd)}")
    print(f"      [LOG] {log_path.absolute()}")
    with log_path.open("w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        subprocess.run(
            cmd,
            check=True,
            stdout=f,
            stderr=f,
            cwd=str(cwd) if cwd else None,
            env=env,
        )


def expand_scene_entries(entries: Any) -> list[int] | None:
    """
    Same semantics as FD run.py: entries can be ints or inclusive ranges "a-b".
    Returns unique ordered logical IDs / scene IDs.
    """
    if entries is None:
        return None

    if isinstance(entries, (int, str)):
        entries = [entries]

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
            if not s:
                continue
            if "-" in s:
                a_str, b_str = s.split("-", 1)
                a = int(a_str.strip())
                b = int(b_str.strip())
                step = 1 if a <= b else -1
                for v in range(a, b + step, step):
                    add(v)
            else:
                add(int(s))
        else:
            raise TypeError(f"Unsupported scenes_to_process entry type: {type(item)}")

    return ordered


def load_cfg(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    print(f"[INIT] Loading config from: {config_path}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    print("[INIT] Config loaded successfully")
    return cfg


def _get_path(cfg: dict, *keys: str, required: bool = True) -> Path | None:
    d = cfg
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            if required:
                raise ValueError(f"Missing config key: {'.'.join(keys)}")
            return None
        d = d[k]
    if d in (None, ""):
        if required:
            raise ValueError(f"Empty config key: {'.'.join(keys)}")
        return None
    return Path(os.path.expanduser(str(d))).resolve()


def get_fd_input_root(cfg: dict) -> Path:
    """
    FD input root, expected by main_fd_v5/main.py:
      input_root/<extrap>/SLC/<product>.SAFE
      input_root/<extrap>/GRD/<product>.SAFE
      input_root/<extrap>/<extrap>_mask/...
    """
    p = _get_path(cfg, "paths", "input_root", required=False)
    if p is None:
        p = _get_path(cfg, "input", "fd_input_root", required=False)
    if p is None:
        p = _get_path(cfg, "input", "input_root", required=True)
    if not p.exists():
        raise NotADirectoryError(f"FD input_root does not exist: {p}")
    print(f"[INIT] FD input root: {p}")
    return p


def get_products_root(cfg: dict) -> Path:
    """
    Product root for slc2raw/L0Patcher validation:
      products_root/slc/<SLC>.SAFE
      products_root/raw/<RAW>.SAFE

    If input.products_root is absent, defaults to FD input root.
    """
    p = _get_path(cfg, "input", "products_root", required=False)
    if p is None:
        p = get_fd_input_root(cfg)
    if not p.exists():
        raise NotADirectoryError(f"products_root does not exist: {p}")
    print(f"[INIT] Products root: {p}")
    return p


def get_output_root(cfg: dict) -> Path:
    p = _get_path(cfg, "output", "output_root", required=False)
    if p is None:
        p = _get_path(cfg, "paths", "output_root", required=True)
    print(f"[INIT] Output root: {p}")
    return p


def get_mapping_csv(cfg: dict) -> Path:
    """Mapping file used by both Step 1 FD and this full pipeline."""
    p = _get_path(cfg, "paths", "floods_mapping_csv", required=False)
    if p is None:
        p = _get_path(cfg, "input", "scene_mapping", required=True)
    if not p.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {p}")
    print(f"[INIT] Mapping CSV: {p}")
    return p


def normalize_safe_base(name: str) -> str:
    """Remove .SAFE suffix if present."""
    name = str(name).strip().strip('"').strip("'")
    return name[:-5] if name.endswith(".SAFE") else name


def normalize_safe_dir(name: str) -> str:
    """Ensure .SAFE suffix is present."""
    base = normalize_safe_base(name)
    return base + ".SAFE"


def build_slc_safe_path(products_root: Path, slc_name: str) -> Path:
    """
    For slc2raw, this should point to the SLC SAFE product folder.

    Supports either:
      products_root/slc/<SLC>.SAFE
    or, if products_root is the FD input root and scene/extrap-specific validation is used,
    the caller may override/construct a scene-specific path.
    """
    return products_root / "slc" / normalize_safe_dir(slc_name)


def build_l0_decoded_path(products_root: Path, raw_name: str) -> Path:
    return products_root / "raw" / normalize_safe_dir(raw_name)


def _clean_cell(v: Any) -> str:
    return str(v or "").strip().strip('"').strip("'")


def load_mapping_rows(mapping_csv: Path) -> dict[int, dict[str, str]]:
    """
    Accepts either of these schemas:

    Full-pipeline schema:
      num ; slc ; raw ; ... optional fields

    FD schema:
      ID ; SCENE ; AOI ; GRD ; SLC ; optional RAW/raw

    RAW is optional. If RAW is missing/empty for a scene, the pipeline runs
    only Step 1: FD label/SLC/GRD patch generation, then skips slc2raw and
    L0Patcher.

    Returns dict keyed by logical ID / num.
    """
    rows: dict[int, dict[str, str]] = {}

    with mapping_csv.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError(f"Mapping CSV has no header: {mapping_csv}")

        fieldnames = [c.strip() for c in reader.fieldnames]

        # Handle hierarchical FD CSV manually: ID/SCENE/AOI can be blank.
        current_id: int | None = None
        current_scene: str | None = None
        current_aoi: str | None = None

        for raw_row in reader:
            row = {str(k).strip(): _clean_cell(v) for k, v in raw_row.items() if k is not None}

            if not any(row.values()):
                continue

            if "num" in fieldnames:
                if not row.get("num"):
                    continue
                num = int(row["num"])
                rows[num] = row
                rows[num]["num"] = str(num)
                continue

            if "ID" not in fieldnames:
                raise ValueError("Mapping CSV must contain either 'num' or 'ID' column")

            if row.get("ID"):
                current_id = int(row["ID"])
            if row.get("SCENE"):
                current_scene = row["SCENE"]
            if row.get("AOI"):
                current_aoi = row["AOI"]

            if current_id is None:
                raise ValueError("Found FD mapping row without ID and no previous ID to propagate")
            if current_scene is None:
                raise ValueError(f"ID {current_id}: missing SCENE and no previous SCENE to propagate")
            if current_aoi is None:
                raise ValueError(f"ID {current_id}: missing AOI and no previous AOI to propagate")

            # If multiple rows have same ID, preserve first for full-pipeline lookup.
            # FD run.py will process all rows for that ID during Step 1.
            if current_id not in rows:
                rows[current_id] = row.copy()
                rows[current_id]["ID"] = str(current_id)
                rows[current_id]["num"] = str(current_id)
                rows[current_id]["SCENE"] = current_scene
                rows[current_id]["AOI"] = current_aoi
            else:
                # Fill missing values from later rows if useful.
                for k, v in row.items():
                    if v and not rows[current_id].get(k):
                        rows[current_id][k] = v

    return rows


def get_row_value(row: dict[str, str], *names: str, required: bool = True) -> str:
    for name in names:
        if name in row and _clean_cell(row[name]):
            return _clean_cell(row[name])
        # also try case variants
        for k, v in row.items():
            if k.lower() == name.lower() and _clean_cell(v):
                return _clean_cell(v)
    if required:
        raise KeyError(f"Missing required mapping column/value among: {names}")
    return ""


def infer_extrap(row: dict[str, str]) -> str | None:
    scene = get_row_value(row, "SCENE", "scene", required=False)
    if scene:
        return scene.split()[0]
    return None


def build_scene_specific_slc_path(fd_input_root: Path, row: dict[str, str], slc_name: str) -> Path | None:
    extrap = infer_extrap(row)
    if not extrap:
        return None
    return fd_input_root / extrap / "SLC" / normalize_safe_dir(slc_name)


def find_fd_scene_output(out_root: Path, scene_num: int, row: dict[str, str], cfg: dict) -> Path:
    """
    Matches corrected FD run.py behavior.

    Preferred corrected FD output:
      output_root/<extrap>/id_XXX

    Legacy full-pipeline wrapper output:
      output_root/scene_XXX
    """
    extrap = infer_extrap(row)
    output_mode = cfg.get("output", {}).get("mode", "per_id")

    candidates: list[Path] = []

    if extrap:
        if output_mode == "per_id":
            candidates.append(out_root / extrap / f"id_{scene_num:03d}")
        candidates.append(out_root / extrap)

    candidates.append(out_root / f"scene_{scene_num:03d}")

    for c in candidates:
        if (c / "label").exists():
            return c

    # Return first expected path even if not created, so the error message is useful.
    return candidates[0]


def make_fd_temp_config(cfg: dict, scene_num: int, scene_out: Path, fd_input_root: Path, out_root: Path, mapping_csv: Path) -> Path:
    """
    Generate a config compatible with corrected FD run.py.
    """
    cfg2 = dict(cfg)

    # Preserve existing sections but guarantee corrected FD run.py keys.
    cfg2.setdefault("processing", {})
    cfg2.setdefault("paths", {})
    cfg2.setdefault("options", {})
    cfg2.setdefault("output", {})

    cfg2["processing"]["scenes_to_process"] = [scene_num]
    cfg2["paths"]["input_root"] = str(fd_input_root)
    cfg2["paths"]["output_root"] = str(out_root)
    cfg2["paths"]["floods_mapping_csv"] = str(mapping_csv)
    cfg2["output"].setdefault("mode", "per_id")

    tmp_cfg = scene_out / f"_tmp_fd_config_scene_{scene_num:03d}.yaml"
    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
    tmp_cfg.write_text(yaml.safe_dump(cfg2, sort_keys=False), encoding="utf-8")
    return tmp_cfg


# ============================================================
# Steps
# ============================================================

def step1_fd(scene_num: int, cfg: dict, out_root: Path, fd_root: Path, mapping_csv: Path, fd_input_root: Path, row: dict[str, str]) -> Path:
    """
    Runs corrected FD run.py for a single logical ID by generating a temporary config.

    Produces labels under corrected FD run.py output, usually:
      out_root/<extrap>/id_XXX/label
    """
    print("\n  [STEP 1] Flood Processing")
    print("  =========================")

    wrapper_out = out_root / f"scene_{scene_num:03d}"
    done = wrapper_out / "01_step1_DONE.txt"
    log = wrapper_out / "01_step1_run.log"

    expected_fd_out = find_fd_scene_output(out_root, scene_num, row, cfg)

    if done.exists() and (expected_fd_out / "label").exists():
        print(f"  [SKIP] Step 1 already completed: {done}")
        print(f"  [INFO] FD output: {expected_fd_out}")
        return expected_fd_out

    print(f"  [EXEC] Creating wrapper output directory: {wrapper_out}")
    wrapper_out.mkdir(parents=True, exist_ok=True)

    print(f"  [EXEC] Generating temporary FD config for ID {scene_num:03d}")
    tmp_cfg = make_fd_temp_config(cfg, scene_num, wrapper_out, fd_input_root, out_root, mapping_csv)
    print(f"  [EXEC] Temp config written to: {tmp_cfg}")

    print("  [EXEC] Running FD processing...")
    cmd = ["python", "run.py", "--config", str(tmp_cfg)]
    run_logged(cmd, log_path=log, cwd=fd_root)

    if tmp_cfg.exists():
        print("  [EXEC] Cleaning up temporary config")
        tmp_cfg.unlink()

    fd_out = find_fd_scene_output(out_root, scene_num, row, cfg)
    if not (fd_out / "label").exists():
        raise FileNotFoundError(f"FD completed but label folder not found at expected output: {fd_out / 'label'}")

    done.write_text(f"ok\nfd_output={fd_out}\n", encoding="utf-8")
    print("  [DONE] Step 1 completed successfully")
    print(f"  [INFO] FD output: {fd_out}")
    return fd_out


def step2_slc2raw(scene_out: Path, label_folder: Path, slc_safe: Path) -> Path:
    """
    Runs:
      slc2raw label_folder product_folder -o rawH.npy -l traceability_log.txt
    Returns patch_info file path (rawH.npy).
    """
    print("\n  [STEP 2] SLC to RAW Conversion")
    print("  ==============================")

    out_dir = scene_out / "02_slc2raw"
    out_file = out_dir / "rawH.npy"
    done = out_dir / "_DONE.txt"
    log = out_dir / "02_step2_driver.log"
    tool_log = out_dir / "traceability_log.txt"

    if done.exists() and out_file.exists():
        print(f"  [SKIP] Step 2 already completed: {done}")
        print(f"  [INFO] Patch info file: {out_file}")
        return out_file

    print(f"  [EXEC] Creating output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("  [EXEC] Converting SLC patches to RAW format...")
    print(f"  [INFO] Label folder: {label_folder}")
    print(f"  [INFO] SLC SAFE: {slc_safe}")
    print(f"  [INFO] Output file: {out_file}")

    cmd = [
        "slc2raw",
        str(label_folder),
        str(slc_safe),
        "-o",
        str(out_file),
        "-l",
        str(tool_log),
    ]
    run_logged(cmd, log_path=log)

    done.write_text("ok\n", encoding="utf-8")
    print("  [DONE] Step 2 completed successfully")
    print(f"  [INFO] Patch info created: {out_file}")
    return out_file


def step3_l0patcher(scene_out: Path, l0_decoded: Path, patch_info: Path, base_name: str) -> Path:
    """Runs L0Patcher to extract RAW patches."""
    print("\n  [STEP 3] L0 RAW Patch Extraction")
    print("  =================================")

    out_dir = scene_out / "03_raw_patches"
    done = out_dir / "_DONE.txt"
    driver_log = out_dir / "03_step3_driver.log"
    tool_log = out_dir / "Logfile.log"

    if done.exists():
        status = done.read_text(encoding="utf-8").strip()
        if status in ("ok", "partial"):
            dat_count = len(list(out_dir.glob("*.dat")))
            print(f"  [SKIP] Step 3 already completed (status: {status})")
            print(f"  [INFO] Found {dat_count} patch files")
            return out_dir

    print(f"  [EXEC] Creating output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("  [EXEC] Extracting L0 RAW patches...")
    print(f"  [INFO] L0 decoded: {l0_decoded}")
    print(f"  [INFO] Patch info: {patch_info}")
    print(f"  [INFO] Patch base name: {base_name}")
    print(f"  [INFO] Output directory: {out_dir}")

    cmd = [
        "L0Patcher",
        str(l0_decoded),
        str(patch_info),
        "-o",
        str(out_dir),
        "-l",
        str(tool_log),
        "-b",
        base_name,
    ]

    try:
        run_logged(cmd, log_path=driver_log)
        dat_count = len(list(out_dir.glob("*.dat")))
        done.write_text("ok\n", encoding="utf-8")
        print("  [DONE] Step 3 completed successfully")
        print(f"  [INFO] Created {dat_count} patch files")
        return out_dir

    except subprocess.CalledProcessError as e:
        dat_files = list(out_dir.glob("*.dat"))
        if dat_files:
            print(f"  [WARN] L0Patcher exited with error (code={e.returncode})")
            print(f"  [WARN] But produced {len(dat_files)} .dat patches")
            msg = (
                f"L0Patcher exited non-zero (code={e.returncode}) but produced {len(dat_files)} .dat patches.\n"
                f"Driver log: {driver_log.absolute()}\n"
                f"Tool log: {tool_log.absolute()}\n"
            )
            (out_dir / "_PARTIAL_SUCCESS.txt").write_text(msg, encoding="utf-8")
            done.write_text("partial\n", encoding="utf-8")
            print("  [EXEC] Marked as partial success")
            return out_dir
        print("  [ERROR] L0Patcher failed and produced no patches")
        raise


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate full Flood dataset pipeline: FD labels → SLC2RAW → L0Patcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_dataset.py
  python generate_dataset.py --config /path/to/config.yaml
        """,
    )
    parser.add_argument("--config", type=Path, help="Path to config.yaml (default: ./config.yaml)")
    args = parser.parse_args()

    print("=" * 70)
    print("FULL FLOOD DATASET GENERATION PIPELINE")
    print("=" * 70)

    script_dir = Path(__file__).resolve().parent
    print(f"\n[INIT] Script directory: {script_dir}")

    config_path = args.config if args.config else script_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print("\n[INIT] Checking required tools...")
    tools_ok = True
    for tool in ["slc2raw", "L0Patcher"]:
        if check_tool_exists(tool):
            print(f"[INIT] ✓ {tool} found")
        else:
            print(f"[INIT] ✗ {tool} NOT FOUND")
            tools_ok = False

    if not tools_ok:
        raise EnvironmentError("Required tools not found on PATH")

    print("\n[INIT] Loading configuration...")
    cfg = load_cfg(config_path)

    fd_input_root = get_fd_input_root(cfg)
    products_root = get_products_root(cfg)
    out_root = get_output_root(cfg)
    mapping_csv = get_mapping_csv(cfg)

    print("\n[INIT] Creating output directory...")
    out_root.mkdir(parents=True, exist_ok=True)

    print("\n[INIT] Determining scenes / IDs to process...")
    scenes_cfg = (
        cfg.get("processing", {}).get("scenes_to_process", None)
        or cfg.get("input", {}).get("scenes_to_process", None)
    )
    wanted_scenes = expand_scene_entries(scenes_cfg)

    print("[INIT] Loading mapping...")
    mapping_by_num = load_mapping_rows(mapping_csv)
    print(f"[INIT] Loaded {len(mapping_by_num)} logical scenes from mapping file")

    if not wanted_scenes:
        wanted_scenes = sorted(mapping_by_num.keys())
        print(f"[INIT] No scenes specified, processing all {len(wanted_scenes)} scenes")
    else:
        print(f"[INIT] Processing {len(wanted_scenes)} scenes: {wanted_scenes}")

    print(f"\n{'=' * 70}")
    print("CONFIGURATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Config file   : {config_path.absolute()}")
    print(f"Mapping CSV   : {mapping_csv.absolute()}")
    print(f"FD root       : {script_dir}")
    print(f"FD input root : {fd_input_root}")
    print(f"Products root : {products_root}")
    print(f"Output root   : {out_root}")
    print(f"Scenes / IDs  : {wanted_scenes}")
    print(f"{'=' * 70}")

    failed_scenes: list[tuple[int, str]] = []

    for idx, scene_num in enumerate(wanted_scenes, 1):
        print(f"\n{'#' * 70}")
        print(f"# SCENE / ID {scene_num:03d} ({idx}/{len(wanted_scenes)})")
        print(f"{'#' * 70}")

        row = mapping_by_num.get(scene_num)
        if row is None:
            print(f"[WARN] Scene {scene_num:03d} not found in mapping CSV. Skipping.")
            failed_scenes.append((scene_num, "not in mapping file"))
            continue

        try:
            slc_name = get_row_value(row, "slc", "SLC")
        except KeyError as e:
            print(f"[WARN] Scene {scene_num:03d}: {e}. Skipping.")
            failed_scenes.append((scene_num, str(e)))
            continue

        # RAW is optional. If absent, we generate only FD labels/SLC/GRD patches.
        raw_name = get_row_value(row, "raw", "RAW", required=False)
        has_raw = bool(raw_name)

        patch_base_name = normalize_safe_base(slc_name)

        # Prefer scene-specific FD SLC if available; otherwise use products_root/slc.
        scene_specific_slc = build_scene_specific_slc_path(fd_input_root, row, slc_name)
        generic_slc = build_slc_safe_path(products_root, slc_name)
        slc_safe = scene_specific_slc if scene_specific_slc and scene_specific_slc.exists() else generic_slc

        l0_decoded = build_l0_decoded_path(products_root, raw_name) if has_raw else None

        print("\n[INFO] Scene Details:")
        print(f"  SLC name   : {slc_name}")
        print(f"  RAW name   : {raw_name if has_raw else '[not provided - Step 2/3 will be skipped]'}")
        print(f"  SLC path   : {slc_safe}")
        if has_raw:
            print(f"  L0 path    : {l0_decoded}")
        print(f"  Patch base : {patch_base_name}")

        print("\n[CHECK] Validating input products...")
        validation_failed = False

        if not slc_safe.exists():
            print(f"[FAIL] ✗ SLC SAFE not found: {slc_safe}")
            failed_scenes.append((scene_num, "SLC SAFE missing"))
            validation_failed = True
        else:
            print("[CHECK] ✓ SLC SAFE exists")

        if has_raw:
            if l0_decoded is None or not l0_decoded.exists():
                print(f"[FAIL] ✗ L0 decoded not found: {l0_decoded}")
                failed_scenes.append((scene_num, "L0 decoded missing"))
                validation_failed = True
            else:
                print("[CHECK] ✓ L0 decoded exists")
        else:
            print("[CHECK] RAW not provided: Step 2 slc2raw and Step 3 L0Patcher will be skipped")

        if validation_failed:
            print(f"[FAIL] Validation failed, skipping scene {scene_num:03d}")
            continue

        # Step 1: FD processing
        try:
            scene_out = step1_fd(scene_num, cfg, out_root, script_dir, mapping_csv, fd_input_root, row)
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

        label_folder = scene_out / "label"
        print("\n[CHECK] Verifying FD outputs...")
        if not label_folder.exists():
            print(f"[FAIL] ✗ Label folder not found: {label_folder}")
            failed_scenes.append((scene_num, "label folder missing"))
            continue

        label_count = len(list(label_folder.glob("*.xml")))
        print(f"[CHECK] ✓ Label folder exists ({label_count} XML files)")
        if label_count == 0:
            print(f"[FAIL] ✗ No XML labels found in: {label_folder}")
            failed_scenes.append((scene_num, "no labels generated"))
            continue

        # If RAW is not available, stop here successfully after FD generation.
        if not has_raw:
            print(f"\n[SUCCESS] Scene {scene_num:03d} completed without RAW stage.")
            print(f"[SUCCESS] Generated FD outputs only: {scene_out}")
            print(f"[SUCCESS] Labels: {label_count} XML files")
            continue

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

    print(f"\n{'=' * 70}")
    print("PROCESSING SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total scenes requested : {len(wanted_scenes)}")
    print(f"Successfully completed : {len(wanted_scenes) - len(failed_scenes)}")
    print(f"Failed                 : {len(failed_scenes)}")

    if failed_scenes:
        print(f"\n{'=' * 70}")
        print("FAILED SCENES")
        print(f"{'=' * 70}")
        for num, reason in failed_scenes:
            print(f"  Scene {num:03d}: {reason}")

    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETED")
    print(f"{'=' * 70}\n")

    if failed_scenes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
