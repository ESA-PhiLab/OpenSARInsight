import os
import MissionConstantsPath_def as OBJ_MIS_PATH

# ========================================================================================
# =================================== LOCAL FUNCTIONS ====================================
# ========================================================================================
def remove_trailing_backslash(path: str) -> str:
    if path.endswith("\\"):
        return path[:-1]
    return path

def fcn_VerificationExistFilesInFolder(folder_path):
    return os.path.isdir(folder_path) and any(
        os.path.isfile(os.path.join(folder_path, f))
        for f in os.listdir(folder_path)
    )

# Iterates through a folder and returns file names one at a time
def fcn_GetFileNamesInFolder(folder_path, verbosity=None):
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        if verbosity:
            print(f"Info: The directory '{folder_path}' does not exist.")
        return
    # Check if folder is empty
    if not os.listdir(folder_path):
        if verbosity:
            print(f"Info: The directory '{folder_path}' is empty.")
        return
    # Iterate over the files in the directory
    found = False
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path):
            found = True
            yield filename  # Yield file names one at a time
    if not found and verbosity:
        print(f"Info: No files found in '{folder_path}'.")

import os

import os

# def fcn_CountFilesInFolder(ReferenceCase, countRAW, countXML, countL1, countL1_0, countMask):
#     try:
#         if ReferenceCase == "Flood":
#             FD_path_XML = OBJ_MIS_PATH.PATH_FLOODS_XML
#             FD_path_RAW = OBJ_MIS_PATH.PATH_FLOODS_RAW
#             FD_path_L1  = OBJ_MIS_PATH.PATH_FLOODS
#             FD_path_L1_0 = OBJ_MIS_PATH.PATH_FLOODS_0
#             FD_path_MASK = OBJ_MIS_PATH.PATH_RFI_MASK

#         elif ReferenceCase == "Vessel":
#             FD_path_XML = OBJ_MIS_PATH.PATH_VESSELS_XML
#             FD_path_RAW = OBJ_MIS_PATH.PATH_VESSELS_RAW
#             FD_path_L1  = OBJ_MIS_PATH.PATH_VESSELS
#             FD_path_L1_0 = OBJ_MIS_PATH.PATH_VESSELS_0
#             FD_path_MASK = OBJ_MIS_PATH.PATH_RFI_MASK

#         elif ReferenceCase == "RFI":
#             FD_path_XML = OBJ_MIS_PATH.PATH_RFI_XML
#             FD_path_RAW = OBJ_MIS_PATH.PATH_RFI_RAW
#             FD_path_L1  = OBJ_MIS_PATH.PATH_RFI
#             FD_path_L1_0 = OBJ_MIS_PATH.PATH_RFI_0
#             FD_path_MASK = OBJ_MIS_PATH.PATH_RFI_MASK

#         else:
#             print(f"[ERROR] Unknown ReferenceCase: {ReferenceCase}")
#             return 0, 0, 0, 0, 0

#         filesRAW = [f for f in os.listdir(FD_path_RAW) if os.path.isfile(os.path.join(FD_path_RAW, f))]
#         filesXML = [f for f in os.listdir(FD_path_XML) if os.path.isfile(os.path.join(FD_path_XML, f))]
#         filesL1  = [f for f in os.listdir(FD_path_L1)  if os.path.isfile(os.path.join(FD_path_L1, f))]
#         filesL1_0  = [f for f in os.listdir(FD_path_L1_0)  if os.path.isfile(os.path.join(FD_path_L1_0, f))]
#         filesMask = [f for f in os.listdir(FD_path_MASK)  if os.path.isfile(os.path.join(FD_path_MASK, f))]
        
#         countRAW = len(filesRAW)
#         countXML = len(filesXML)
#         countL1 = len(filesL1)
#         countL1_0 = len(filesL1_0)
#         countMask = len(filesMask)

#         return countRAW, countXML, countL1, countL1_0, countMask

#     except Exception as e:
#         print(f"[ERROR] Exception while counting files: {e}")
#         return 0, 0, 0,  0

def fcn_CountFilesInFolder(ReferenceCase, countRAW, countXML, countL1, countL1_0, countMask,
                           Path_Tiles, Path_Tiles_0, Path_Raw, Path_Labels, Path_Mask):
    try:
        if Path_Tiles.endswith("\\"):
            Path_Tiles = Path_Tiles[:-1]
        if Path_Tiles_0.endswith("\\"):
            Path_Tiles_0 = Path_Tiles_0[:-1]
        if Path_Raw.endswith("\\"):
            Path_Raw = Path_Raw[:-1]
        if Path_Labels.endswith("\\"):
            PathPath_Labels = Path_Labels[:-1]
        if Path_Mask.endswith("\\"):
            Path_Mask = Path_Mask[:-1]
        if ReferenceCase == "Flood" or ReferenceCase == "Vessel" or ReferenceCase == "RFI":
            FD_path_XML = Path_Labels
            FD_path_RAW = Path_Raw
            FD_path_L1  = Path_Tiles
            FD_path_L1_0 = Path_Tiles_0
            FD_path_MASK = Path_Mask
        else:
            print(f"[ERROR] Unknown ReferenceCase: {ReferenceCase}")
            return 0, 0, 0, 0, 0

        filesRAW = [f for f in os.listdir(FD_path_RAW) if os.path.isfile(os.path.join(FD_path_RAW, f))]
        filesXML = [f for f in os.listdir(FD_path_XML) if os.path.isfile(os.path.join(FD_path_XML, f))]
        filesL1  = [f for f in os.listdir(FD_path_L1)  if os.path.isfile(os.path.join(FD_path_L1, f))]
        if ReferenceCase == "RFI":
            filesL1_0 = []
        else:
            filesL1_0 = [f for f in os.listdir(FD_path_L1_0) if os.path.isfile(os.path.join(FD_path_L1_0, f))] if os.path.isdir(FD_path_L1_0) else []
        if ReferenceCase == "RFI":
            filesMask = [f for f in os.listdir(FD_path_MASK)  if os.path.isfile(os.path.join(FD_path_MASK, f))]
        
        countRAW = len(filesRAW)
        countXML = len(filesXML)
        countL1 = len(filesL1)
        if ReferenceCase == "RFI":
            countL1_0 = 0
        else:
            countL1_0 = len(filesL1_0)
        if ReferenceCase == "RFI":
            countMask = len(filesMask)
        else:
            countMask = 0
        return countRAW, countXML, countL1, countL1_0, countMask

    except Exception as e:
        print(f"[ERROR] Exception while counting files: {e}")
        return 0, 0, 0, 0, 0
