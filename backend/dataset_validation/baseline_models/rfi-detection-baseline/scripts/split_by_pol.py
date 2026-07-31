import os
import shutil

def copy_vh_slc_files(source_directory='/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/rfi/aresys_full/patches', destination_folder='dataset/patches/patches1_VH'):
    """
    Copy files that contain both 'vh' and 'slc' in their filename to a new folder.
    
    Args:
        source_directory (str): Directory to search for files
        destination_folder (str): Name of the destination folder
    """
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created directory: {destination_folder}")
    else:
        print(f"Directory already exists: {destination_folder}")
    
    # Get all files in source directory
    files_copied = 0
    
    for filename in os.listdir(source_directory):
        # Skip directories
        if os.path.isdir(os.path.join(source_directory, filename)):
            continue
            
        # Check if filename contains both 'vh' and 'slc' (case insensitive)
        filename_lower = filename.lower()
        if 'vh' in filename_lower and 'slc' in filename_lower:
            source_path = os.path.join(source_directory, filename)
            dest_path = os.path.join(destination_folder, filename)
            
            try:
                shutil.copy2(source_path, dest_path)
                print(f"Copied: {filename}")
                files_copied += 1
            except Exception as e:
                print(f"Error copying {filename}: {e}")
    
    print(f"\nTotal files copied: {files_copied}")

# Run the script
if __name__ == "__main__":
    copy_vh_slc_files()
