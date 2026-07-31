from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

from main import main


# ============================================================
# UTILITIES
# ============================================================

def _normpath(p: str | None) -> str | None:
    return os.path.normpath(os.path.expanduser(str(p))) if p else p


def _expand_id_entries(entries):
    """
    Converts scenes_to_process / IDs to a list of 1-based logical IDs.

    Supports:
        None        -> all
        1           -> [1]
        "1"         -> [1]
        "1-5"       -> [1, 2, 3, 4, 5]
        [1, "3-4"]  -> [1, 3, 4]
    """
    if entries is None:
        return None

    if isinstance(entries, int):
        entries = [entries]

    if isinstance(entries, str):
        entries = [entries]

    out = []
    for item in entries:
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, str):
            s = item.strip()
            if not s:
                continue
            if "-" in s:
                a, b = s.split("-", 1)
                a, b = int(a.strip()), int(b.strip())
                step = 1 if a <= b else -1
                out.extend(range(a, b + step, step))
            else:
                out.append(int(s))
        else:
            raise TypeError("scenes_to_process must be int/str/list/None")

    # remove duplicates preserving order
    res, seen = [], set()
    for v in out:
        if v not in seen:
            seen.add(v)
            res.append(v)
    return res


def _read_csv_mapping(path_csv: str) -> pd.DataFrame:
    """
    CSV with columns:
        ID ; SCENE ; AOI ; GRD ; SLC

    Supports hierarchy-style CSV where ID/SCENE/AOI can be blank and inherited
    from previous rows.
    """
    if not os.path.isfile(path_csv):
        raise FileNotFoundError(f"Mapping CSV not found: {path_csv}")

    from io import StringIO

    with open(path_csv, "r", encoding="utf-8-sig", errors="ignore") as f:
        text = f.read()

    df = pd.read_csv(
        StringIO(text),
        sep=";",
        dtype=str,
        engine="python",
        quoting=csv.QUOTE_NONE,
        on_bad_lines="warn",
    )

    # normalize column names and string values
    df.columns = [c.strip() for c in df.columns]
    df = df.fillna("").astype(str)
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.strip().strip('"').strip("'"))

    # remove fully empty rows
    df = df[df.apply(lambda r: any(str(v).strip() for v in r.values), axis=1)].reset_index(drop=True)

    required = ["ID", "SCENE", "AOI", "GRD", "SLC"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # propagate hierarchy
    df["ID"] = df["ID"].replace("", pd.NA).ffill()
    if df["ID"].isna().any():
        raise ValueError("There are rows without ID before propagation is possible.")
    df["ID"] = df["ID"].astype(int)

    df["SCENE"] = df["SCENE"].replace("", pd.NA).ffill()
    if df["SCENE"].isna().any():
        raise ValueError("There are rows without SCENE before propagation is possible.")

    # propagate AOI only inside each ID
    df["AOI"] = df["AOI"].replace("", pd.NA)
    df["AOI"] = df.groupby("ID")["AOI"].ffill()

    if df["AOI"].isna().any():
        raise ValueError("There are rows without AOI after propagation.")

    # products cannot be empty on actual processing rows
    bad_grd = df["GRD"].astype(str).str.strip() == ""
    bad_slc = df["SLC"].astype(str).str.strip() == ""
    if bad_grd.any() or bad_slc.any():
        bad_rows = df[bad_grd | bad_slc][["ID", "SCENE", "AOI", "GRD", "SLC"]]
        raise ValueError(f"Rows with missing GRD/SLC products:\n{bad_rows}")

    return df


def _ensure_output_dirs(base_out: str):
    paths = {
        "root": base_out,
        "label": os.path.join(base_out, "label"),
        "cp": os.path.join(base_out, "cp"),
        "grd": os.path.join(base_out, "grd"),
        "logs": os.path.join(base_out, "logs"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def _normalize_safe_name(name: str) -> str:
    name = str(name).strip().strip('"').strip("'")
    if not name.endswith(".SAFE"):
        name += ".SAFE"
    return name


def _parse_aoi_list(aoi_text: str) -> list[str]:
    """
    Converts:
        "1"      -> ["01"]
        "01"     -> ["01"]
        "1,2,10" -> ["01", "02", "10"]
    """
    out = []
    for item in str(aoi_text).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(f"{int(item):02d}")
    if not out:
        raise ValueError(f"Invalid AOI field: {aoi_text!r}")
    return out


def _validate_inputs(input_root: str, extrap: str, aoi_list: list[str], slc: str, grd: str) -> list[str]:
    errors = []

    scene_root = os.path.join(input_root, extrap)
    slc_path = os.path.join(scene_root, "SLC", slc)
    grd_path = os.path.join(scene_root, "GRD", grd)
    mask_root = os.path.join(scene_root, f"{extrap}_mask")
    ne_land = os.path.join(input_root, "shared_data", "natural_earth", "ne_110m_land.shp")

    if not os.path.isdir(scene_root):
        errors.append(f"Scene folder not found: {scene_root}")

    if not os.path.isdir(slc_path):
        errors.append(f"SLC SAFE not found: {slc_path}")

    if not os.path.isdir(grd_path):
        errors.append(f"GRD SAFE not found: {grd_path}")

    if not os.path.isdir(mask_root):
        errors.append(f"Mask folder not found: {mask_root}")

    if not os.path.isfile(ne_land):
        errors.append(f"Natural Earth land shapefile not found: {ne_land}")

    for aoi in aoi_list:
        aoi_shp = os.path.join(mask_root, aoi, "aoi", "aoi.shp")
        if not os.path.isfile(aoi_shp):
            errors.append(f"AOI shapefile not found for AOI {aoi}: {aoi_shp}")

    return errors


# ============================================================
# MAIN LAUNCHER
# ============================================================

def main_launcher():
    parser = argparse.ArgumentParser(description="Launcher for Flood Detection YAML + floods_mapping.csv")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if a DONE marker exists.",
    )
    args = parser.parse_args()

    cfg_path = _normpath(args.config)
    if not cfg_path or not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    try:
        input_root = _normpath(cfg["paths"]["input_root"])
        output_root = _normpath(cfg["paths"]["output_root"])
        mapping_csv = _normpath(cfg["paths"]["floods_mapping_csv"])
        sizepatch = int(cfg["options"].get("SIZEPATCH", 512))
        use_aoi_exact_polygon = bool(cfg["options"].get("USE_AOI_EXACT_POLYGON", False))
        ids_cfg = cfg.get("processing", {}).get("scenes_to_process")
    except Exception as e:
        raise KeyError(f"Malformed YAML or missing keys: {e}") from e

    # Safer default: each logical ID gets its own output folder.
    # This avoids overwrite because main() restarts patch numbering from 1 per call.
    output_mode = cfg.get("output", {}).get("mode", "per_id")
    if output_mode not in {"per_id", "per_scene"}:
        raise ValueError("output.mode must be either 'per_id' or 'per_scene'")

    ids_to_run = _expand_id_entries(ids_cfg)

    print("\n========== CONFIG ==========")
    print("input_root            :", input_root)
    print("output_root           :", output_root)
    print("mapping_csv           :", mapping_csv)
    print("sizepatch             :", sizepatch)
    print("USE_AOI_EXACT_POLYGON :", use_aoi_exact_polygon)
    print("output_mode           :", output_mode)
    print("IDs to run            :", ids_to_run if ids_to_run else "ALL")
    print("============================\n")

    if not input_root or not os.path.isdir(input_root):
        raise NotADirectoryError(f"input_root does not exist: {input_root}")

    if not output_root:
        raise ValueError("output_root is empty")
    os.makedirs(output_root, exist_ok=True)

    df = _read_csv_mapping(mapping_csv)

    if ids_to_run:
        df = df[df["ID"].isin(ids_to_run)]

    if df.empty:
        raise ValueError("No matching IDs found in mapping CSV.")

    failed = []

    # ----------------------------
    # PROCESS BY LOGICAL ID
    # ----------------------------
    for id_value, block in df.groupby("ID", sort=True):
        scene = block["SCENE"].iloc[0]
        extrap = scene.split()[0]

        print(f"\n{'=' * 70}")
        print(f"ID {int(id_value):03d} | SCENE {scene}")
        print(f"{'=' * 70}")

        # Usually one block should be processed in one call if all rows share same SLC/GRD.
        # If there are multiple rows under one ID, each row is processed separately.
        for row_idx, row in block.iterrows():
            aoi_list = _parse_aoi_list(row["AOI"])
            slc = _normalize_safe_name(row["SLC"])
            grd = _normalize_safe_name(row["GRD"])

            if output_mode == "per_id":
                out_scene = os.path.join(output_root, extrap, f"id_{int(id_value):03d}")
            else:
                # Use only if you are sure each scene is processed once.
                out_scene = os.path.join(output_root, extrap)

            _ensure_output_dirs(out_scene)

            logs_dir = os.path.join(out_scene, "logs")
            done_marker = os.path.join(logs_dir, f"DONE_id_{int(id_value):03d}_row_{row_idx}.txt")

            print(f"\n[RUN]")
            print(f"  AOIs   : {aoi_list}")
            print(f"  SLC    : {slc}")
            print(f"  GRD    : {grd}")
            print(f"  Output : {out_scene}")

            if os.path.isfile(done_marker) and not args.force:
                print(f"  [SKIP] Done marker exists: {done_marker}")
                continue

            errors = _validate_inputs(
                input_root=input_root,
                extrap=extrap,
                aoi_list=aoi_list,
                slc=slc,
                grd=grd,
            )

            if errors:
                print("  [FAIL] Input validation failed:")
                for err in errors:
                    print(f"    - {err}")
                failed.append((int(id_value), f"input validation failed for row {row_idx}"))
                continue

            try:
                main(
                    inputpath=input_root,
                    SLC_product=slc,
                    GRD_product=grd,
                    extrap=extrap,
                    subextrap=aoi_list,
                    outputpath=out_scene,
                    SIZEPATCH=sizepatch,
                    USE_AOI_EXACT_POLYGON=use_aoi_exact_polygon,
                )

                label_count = len(list(Path(out_scene, "label").glob("*.xml")))
                slc_vv_count = len(list(Path(out_scene, "cp").glob("*SLC_VV.tiff")))
                slc_vh_count = len(list(Path(out_scene, "cp").glob("*SLC_VH.tiff")))
                grd_count = len(list(Path(out_scene, "grd").glob("*.tiff")))

                with open(done_marker, "w", encoding="utf-8") as f:
                    f.write("ok\n")
                    f.write(f"ID={id_value}\n")
                    f.write(f"SCENE={scene}\n")
                    f.write(f"AOI={aoi_list}\n")
                    f.write(f"SLC={slc}\n")
                    f.write(f"GRD={grd}\n")
                    f.write(f"labels={label_count}\n")
                    f.write(f"slc_vv={slc_vv_count}\n")
                    f.write(f"slc_vh={slc_vh_count}\n")
                    f.write(f"grd={grd_count}\n")

                print("  [DONE]")
                print(f"    labels : {label_count}")
                print(f"    SLC VV : {slc_vv_count}")
                print(f"    SLC VH : {slc_vh_count}")
                print(f"    GRD    : {grd_count}")

            except Exception as e:
                print(f"  [FAIL] ID {id_value} row {row_idx}: {e}")
                failed.append((int(id_value), str(e)))
                continue

    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Requested IDs : {ids_to_run if ids_to_run else 'ALL'}")
    print(f"Failed        : {len(failed)}")

    if failed:
        print("\nFAILED:")
        for id_value, reason in failed:
            print(f"  ID {id_value:03d}: {reason}")
        sys.exit(1)

    print("\n[DONE] All IDs processed successfully.")


if __name__ == "__main__":
    try:
        main_launcher()
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
