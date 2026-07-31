import os
import sys
import time
from dotenv import dotenv_values  # type: ignore

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.sentinelHubAPI import CopernicusDataSpace

# Load config
script_dir = os.path.dirname(os.path.abspath(__file__))  # /src/sentinelHubAPI/
src_dir = os.path.dirname(script_dir)                    # /src/
project_root = os.path.dirname(src_dir)                  # /project_root/
env_path = os.path.join(project_root, ".env")

def batch_download(scene_list: list[str], download_dir: str):
    """
    Batch download scenes using CopernicusDataSpace client.

    Args:
        scene_list (list): List of scene names to be downloaded.
        download_dir (str): Directory path where downloaded scenes will be saved.

    Returns:
        None: This function does not return anything. It prints status messages to the console.

    Notes:
        - Loads configuration from environment file for each scene.
        - Waits 2 minutes between downloads to avoid rate limiting.
        - Prints success or error messages for each scene.
    """
    for i, scene_name in enumerate(scene_list, 1):
        # Print progress for current scene
        print(f"\n[{i}/{len(scene_list)}] Processing: {scene_name}")

        try:
            # Load configuration from .env file
            config = dotenv_values(env_path)  # type: ignore
            print(f"Config loaded successfully")

            # Initialize CopernicusDataSpace client with credentials
            client = CopernicusDataSpace(
                base_url=config['base_url'],  # type: ignore
                username=config['username'],  # type: ignore
                password=config['password']  # type: ignore
            )

            # Search for the scene and download it
            success = client.search_and_download(scene_name, download_dir)

            if success:
                print(f"✅ Successfully downloaded: {scene_name}")
            else:
                print(f"❌ Failed to download: {scene_name}")

        except Exception as e:
            # Print error message if any exception occurs
            print(f"❌ Error processing {scene_name}: {e}")

        # Wait 2 minutes before next download, except for the last scene
        if i < len(scene_list):  # type: ignore
            print(f"⏰ Waiting 2 minutes before next download...")
            time.sleep(120)  # 2 minutes = 120 seconds

    # Print completion message
    print("\n" + "=" * 50)
    print("Batch download completed")