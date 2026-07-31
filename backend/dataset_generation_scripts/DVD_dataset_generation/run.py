import os
import sys
import argparse
import yaml
import pandas as pd
from main import *
import glob

def _normpath(p: str) -> str:
    """Normalize a filesystem path (handles Windows/Unix)."""
    return os.path.normpath(os.path.expanduser(p)) if p else p

def _expand_scene_entries(entries):
    """
    Expand a list of entries that can be integers or inclusive ranges ("a-b")
    into a sorted list of unique integers, preserving the original order.
    """
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
                    if a <= b:
                        for v in range(a, b + 1):
                            add(v)
                    else:
                        for v in range(a, b - 1, -1):
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
    """Ensure mapping.csv has the required columns."""
    required = ["num", "grd", "slc", "ocn", "raw"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"mapping.csv is missing required columns: {missing}")



def _find_safe_folder(base_dir, product_name):
    """
    Find a Sentinel-1 SAFE folder matching product_name.
    Accepts product_name with or without .SAFE.
    """
    if product_name.endswith(".SAFE"):
        candidate = os.path.join(base_dir, product_name)
        return candidate if os.path.isdir(candidate) else None

    pattern = os.path.join(base_dir, product_name + "*.SAFE")
    matches = glob.glob(pattern)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple SAFE folders match {product_name}: {matches}")

    return None

def _select_rows_by_indices(df: pd.DataFrame, indices):
    """
    Select rows whose df['num'] is in indices.
    If indices is None or empty, return the full DataFrame.
    """
    if not indices:
        return df.copy()
    # Make sure 'num' is numeric and comparable
    dfn = df.copy()
    dfn["num"] = pd.to_numeric(dfn["num"], errors="coerce").astype("Int64")
    wanted = set(int(i) for i in indices)
    out = dfn[dfn["num"].isin(wanted)].copy()
    # Sort by the order of indices provided
    order_map = {v: i for i, v in enumerate(indices)}
    out["__order__"] = out["num"].map(lambda v: order_map.get(int(v), 10**9))
    out = out.sort_values(["__order__", "num"]).drop(columns="__order__")
    return out


def main_launcher():
    parser = argparse.ArgumentParser(
        description="Run DVD processing on selected scenes.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)"
    )
    
    args = parser.parse_args()
    
    # Determine script directory
    script_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(script_dir, config_path)
    
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    
    # Extract configuration values
    input_cfg = cfg.get("input", {})
    output_cfg = cfg.get("output", {})
    
    # Get scene mapping path from config with fallback
    scene_mapping = input_cfg.get("scene_mapping")
    if scene_mapping:
        # Use scene_mapping from config
        if not os.path.isabs(scene_mapping):
            scene_mapping = os.path.join(script_dir, scene_mapping)
    else:
        # Fallback: try common names in script directory
        for fallback_name in ["scene_mapping.csv", "mapping.csv"]:
            fallback_path = os.path.join(script_dir, fallback_name)
            if os.path.isfile(fallback_path):
                scene_mapping = fallback_path
                print(f"[INFO] Using fallback mapping file: {fallback_name}")
                break
    
    if not scene_mapping or not os.path.isfile(scene_mapping):
        raise FileNotFoundError(
            f"Scene mapping file not found. "
            f"Specify 'input.scene_mapping' in config or place 'scene_mapping.csv' or 'mapping.csv' in script directory."
        )
    
    slcvalidation = input_cfg.get("slcvalidation")
    if not slcvalidation:
        raise ValueError("Missing input.slcvalidation in config")
    if not os.path.isabs(slcvalidation):
        slcvalidation = os.path.join(script_dir, slcvalidation)
    if not os.path.isfile(slcvalidation):
        raise FileNotFoundError(f"slcvalidation file not found: {slcvalidation}")
    
    products_root = input_cfg.get("products_root")
    if not products_root:
        raise ValueError("Missing input.products_root in config")
    if not os.path.isabs(products_root):
        products_root = os.path.join(script_dir, products_root)
    
    output_root = output_cfg.get("output_root")
    if not output_root:
        raise ValueError("Missing output.output_root in config")
    if not os.path.isabs(output_root):
        output_root = os.path.join(script_dir, output_root)
    
    sizepatch = input_cfg.get("sizepatch", 512)
    
    print(f"[CONFIG] config file   = {config_path}")
    print(f"[CONFIG] scene_mapping = {scene_mapping}")
    print(f"[CONFIG] products_root = {products_root}")
    print(f"[CONFIG] output_root   = {output_root}")
    print(f"[CONFIG] slcvalidation = {slcvalidation}")
    print(f"[CONFIG] sizepatch     = {sizepatch}")
    
    # Validate products_root structure
    products_slc_dir = os.path.join(products_root, "slc")
    products_grd_dir = os.path.join(products_root, "grd")
    products_ocn_dir = os.path.join(products_root, "ocn")
    
    if not os.path.isdir(products_slc_dir):
        raise NotADirectoryError(f"SLC products directory not found: {products_slc_dir}")
    if not os.path.isdir(products_grd_dir):
        raise NotADirectoryError(f"GRD products directory not found: {products_grd_dir}")
    if not os.path.isdir(products_ocn_dir):
        raise NotADirectoryError(f"OCN products directory not found: {products_ocn_dir}")
    
    # Read scene mapping CSV
    df = pd.read_csv(scene_mapping, sep=";", dtype=str)
    df["num"] = df["num"].astype(int)
    
    # Determine which scenes to process
    scenes_cfg = input_cfg.get("scenes_to_process", None)
    wanted = _expand_scene_entries(scenes_cfg)
    
    if wanted is None:
        df_sel = df
        print(f"[INFO] Processing all {len(df_sel)} scenes from mapping file")
    else:
        df_sel = df[df["num"].isin(wanted)].copy()
        print(f"[INFO] Processing {len(df_sel)} scenes: {sorted(wanted)}")
    
    if df_sel.empty:
        print("[WARN] No scenes selected for processing.")
        return
    
    os.makedirs(output_root, exist_ok=True)
    
    for _, row in df_sel.iterrows():
        num = int(row["num"])
        grd_name = str(row["grd"]).strip()
        slc_name = str(row["slc"]).strip()
        ocn_name = str(row["ocn"]).strip()
        
        # Remove .SAFE suffix if present (main.py will add it)
        grd_name = grd_name.replace(".SAFE", "")
        slc_name = slc_name.replace(".SAFE", "")
        ocn_name = ocn_name.replace(".SAFE", "")
        
        scene_out = os.path.join(output_root, f"scene_{num:03d}")
        
        # Advisory checks before processing
        exp_slc = _find_safe_folder(products_slc_dir, slc_name)
        exp_grd = _find_safe_folder(products_grd_dir, grd_name)
        exp_ocn = _find_safe_folder(products_ocn_dir, ocn_name)
        
        missing = []
        if not exp_slc or not os.path.isdir(exp_slc):
            missing.append(f"SLC: {slc_name}")
        if not exp_grd or not os.path.isdir(exp_grd):
            missing.append(f"GRD: {grd_name}")
        if not exp_ocn or not os.path.isdir(exp_ocn):
            missing.append(f"OCN: {ocn_name}")
        
        if missing:
            print(f"[ERROR] Scene {num} - Missing products: {', '.join(missing)}")
            continue
        
        print(f"\n[RUN] Scene num={num}")
        print(f"      SLC={slc_name}")
        print(f"      GRD={grd_name}")
        print(f"      OCN={ocn_name}")
        print(f"      Output: {scene_out}")
        print(f"      sizepatch={sizepatch}")
        
        try:
            main(
                grdproduct=grd_name,
                slcproduct=slc_name,
                ocnproduct=ocn_name,
                outputpath=scene_out,
                slcvalidation=slcvalidation,
                sizepatch=sizepatch
            )
            print(f"[SUCCESS] Scene {num} completed")
        except Exception as e:
            print(f"[ERROR] Scene {num} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n[DONE] All selected scenes processed.")



if __name__ == "__main__":
    try:
        main_launcher()
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)