# type: ignore
"""
---------------------------------------------------------------------
Project: OpenSAR Insight
---------------------------------------------------------------------
extract_patch_metadata.py

Tool: Extract patch-level and vessel-level metadata from XML annotations
      and export to CSV with summary statistics.

Author: Abdulhameed Yunusa (ABHY)
E-mail: ayunusa@indracompany.com
Creation Date: 2026-05-05

© Copyright INDRA DEIMOS, 2026. All rights reserved.
---------------------------------------------------------------------
"""
import csv
import logging
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utilities.read_yaml import read_yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# XML Parsing
# ═══════════════════════════════════════════════════════════════

def _safe_float(text):
    """Return float or None for missing / NaN values."""
    if text is None:
        return None
    text = text.strip()
    if text.lower() == "nan" or text == "":
        return None
    return float(text)


def _extract_polarization(product_name: str) -> str:
    """Infer polarization from the SAR product name (e.g. 1SDV -> DV, 1SSV -> SV)."""
    if not product_name:
        return "unknown"
    # Sentinel-1 convention: ...1SDV... = Dual VV+VH, ...1SSV... = Single VV
    if "1SDV" in product_name:
        return "DV"
    if "1SSH" in product_name:
        return "SH"
    if "1SSV" in product_name:
        return "SV"
    if "1SDH" in product_name:
        return "DH"
    return "unknown"


def parse_patch_xml(xml_path: Path) -> dict:
    """
    Parse a single XML annotation file and return patch + vessel metadata.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    scene_id = root.findtext(".//Scene_ID", default="")
    sar_product = root.findtext(".//SARProduct", default="")
    swath = root.findtext(".//SLCSwath", default="")
    wind_speed = _safe_float(root.findtext(".//WindData/Speed"))
    wind_dir = _safe_float(root.findtext(".//WindData/Direction"))
    polarization = _extract_polarization(sar_product)
    num_ships = int(root.findtext(".//Number_of_ships", default="0"))

    vessels = []
    for ship in root.findall(".//Ship"):
        name = ship.findtext("Name", default="")
        size = _safe_float(ship.findtext("Size"))
        is_vessel = ship.findtext("Is_Vessel")
        confidence = ship.findtext("Confidence", default="")
        dist_shore = _safe_float(ship.findtext("DistanceToShore"))

        vessels.append({
            "vessel_name": name,
            "vessel_length": size,
            "is_vessel": is_vessel,
            "confidence": confidence,
            "distance_to_shore": dist_shore,
        })

    return {
        "xml_file": xml_path.name,
        "scene_id": scene_id,
        "sar_product": sar_product,
        "polarization": polarization,
        "swath": swath,
        "wind_speed": wind_speed,
        "wind_direction": wind_dir,
        "num_ships": num_ships,
        "vessels": vessels,
    }


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _extract_id_from_image(filename: str) -> str:
    """Extract the numeric ID from an image filename like DB_OPENSAR_VH_VD_104.png -> 104."""
    return Path(filename).stem.split("_")[-1]


def _extract_id_from_xml(filename: str) -> str:
    """Extract the numeric ID from an XML filename like DB_OPENSAR_VD_104.xml -> 104."""
    return Path(filename).stem.split("_")[-1]


def _collect_split_ids(images_dir: Path) -> set:
    """Return set of numeric IDs present in an image directory."""
    ids = set()
    if images_dir.exists():
        for f in images_dir.iterdir():
            if f.suffix == ".png":
                ids.add(_extract_id_from_image(f.name))
    return ids


def _log_split_summary(label: str, counters: dict):
    """Log summary stats for a split."""
    tp = counters["patches"]
    tv = counters["vessels"]
    log.info("  Patches                           : %d", tp)
    log.info("  Vessels                            : %d", tv)
    log.info("  Patches with wind speed            : %d / %d", counters["patches_wind"], tp)
    log.info("  Patches with ≥1 vessel length      : %d / %d", counters["patches_length"], tp)
    log.info("  Vessels with vessel length          : %d / %d", counters["vessels_length"], tv)
    log.info("  Vessels with BOTH wind + length     : %d / %d", counters["vessels_wind_length"], tv)


def _new_counters():
    return {
        "patches": 0,
        "vessels": 0,
        "patches_wind": 0,
        "patches_length": 0,
        "vessels_length": 0,
        "vessels_wind_length": 0,
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

FIELDNAMES = [
    "split",
    "xml_file",
    "scene_id",
    "sar_product",
    "polarization",
    "swath",
    "wind_speed",
    "wind_direction",
    "num_ships",
    "vessel_name",
    "vessel_length",
    "is_vessel",
    "confidence",
    "distance_to_shore",
]


def main():
    config = read_yaml(Path("config.yaml"))
    dataset_root = Path(config["datapaths"]["dataset_root"])
    yolo_dataset = Path(config["datapaths"]["yolo_dataset_filtered_rc"])
    output_csv = Path("patch_vessel_metadata.csv")

    # Collect XML files per split from the unified dataset structure
    split_names = ["train", "val", "test"]
    split_xml_files = {}  # split -> list of xml paths
    for split in split_names:
        split_xml_dir = dataset_root / split / "labels"
        xmls = sorted(split_xml_dir.glob("*.xml")) if split_xml_dir.exists() else []
        split_xml_files[split] = xmls
        log.info("Split %-5s : %d XML files in %s", split, len(xmls), split_xml_dir)

    # Per-split + total counters
    counters = {s: _new_counters() for s in split_names}
    counters["total"] = _new_counters()

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for split in split_names:
            for xml_path in split_xml_files[split]:
                try:
                    patch = parse_patch_xml(xml_path)
                except ET.ParseError:
                    log.warning("Skipping malformed XML: %s", xml_path.name)
                    continue

                has_wind = patch["wind_speed"] is not None
                patch_has_length = False

                for bucket in (split, "total"):
                    counters[bucket]["patches"] += 1
                    if has_wind:
                        counters[bucket]["patches_wind"] += 1

                for v in patch["vessels"]:
                    has_length = v["vessel_length"] is not None

                    for bucket in (split, "total"):
                        counters[bucket]["vessels"] += 1
                        if has_length:
                            counters[bucket]["vessels_length"] += 1
                        if has_wind and has_length:
                            counters[bucket]["vessels_wind_length"] += 1

                    if has_length:
                        patch_has_length = True

                    row = {
                        "split": split,
                        "xml_file": patch["xml_file"],
                        "scene_id": patch["scene_id"],
                        "sar_product": patch["sar_product"],
                        "polarization": patch["polarization"],
                        "swath": patch["swath"],
                        "wind_speed": patch["wind_speed"] if patch["wind_speed"] is not None else "",
                        "wind_direction": patch["wind_direction"] if patch["wind_direction"] is not None else "",
                        "num_ships": patch["num_ships"],
                        "vessel_name": v["vessel_name"],
                        "vessel_length": v["vessel_length"] if v["vessel_length"] is not None else "",
                        "is_vessel": v["is_vessel"],
                        "confidence": v["confidence"],
                        "distance_to_shore": v["distance_to_shore"] if v["distance_to_shore"] is not None else "",
                    }
                    writer.writerow(row)

                if patch_has_length:
                    for bucket in (split, "total"):
                        counters[bucket]["patches_length"] += 1

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("PATCH / VESSEL METADATA SUMMARY  (filtered dataset)")
    log.info("=" * 60)

    for split in split_names:
        log.info("── %s ──", split.upper())
        _log_split_summary(split, counters[split])
        log.info("-" * 60)

    log.info("── TOTAL ──")
    _log_split_summary("total", counters["total"])
    log.info("=" * 60)
    log.info("CSV written to: %s", output_csv.resolve())


if __name__ == "__main__":
    main()
