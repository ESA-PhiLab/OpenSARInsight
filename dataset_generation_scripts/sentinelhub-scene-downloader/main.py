"""
---------------------------------------------------------------------
Project: OpenSAR Insight
---------------------------------------------------------------------
Batch downloader for Sentinel-1 scenes from different categories (DARK Vessels, FLOOD, RFI).

Tool: SentinelHub Downloader
Purpose: Automate batch downloads of Sentinel-1 scenes for OpenSAR Insight.
Customer: ESA

Author: Abdulhameed Yunusa (ABHY)
E-mail: ayunusa@indracompany.com
Creation Date: 2025-08-20

© Copyright INDRA DEIMOS, 2025. All rights reserved.
---------------------------------------------------------------------
"""
import os
import sys
import time

from scenes import dvd, fld, rfi
from savepaths import dvd_path, fld_path, rfi_path
# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.sentinelHubAPI.batchdownloader import batch_download


def main() -> None:
    """Run batch downloads for all scene categories."""

    # Download DVD scenes
    print(f"Starting batch download of DARK Vessels, {len(dvd)} scenes")
    print("=" * 50)
    print()
    batch_download(scene_list=dvd, download_dir=dvd_path)
    print("Waiting 10 minutes before next download...")
    time.sleep(600)  # 10 minutes

    # Download FLOOD scenes
    print(f"Starting batch download of FLOOD, {len(fld)} scenes")
    print("=" * 50)
    print()
    batch_download(scene_list=fld, download_dir=fld_path)
    print("Waiting 10 minutes before next download...")
    time.sleep(600)  # 10 minutes

    # Download RFI scenes
    print(f"Starting batch download of RFI, {len(rfi)} scenes")
    print("=" * 50)
    print()
    batch_download(scene_list=rfi, download_dir=rfi_path)


if __name__ == "__main__":
    main()