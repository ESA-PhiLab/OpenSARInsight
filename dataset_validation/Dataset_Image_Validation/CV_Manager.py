# standard libraries
import sys
import os
import time

import xml.etree.ElementTree as ET

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Recursively add all subdirectories
for root, dirs, _ in os.walk(current_dir):
    for dir_name in dirs:
        sys.path.append(os.path.join(root, dir_name))

# user-defined core libraries
import CoreConstants_def as OBJ_CORE_CONSTANT
import CoreEnumeration_def as OBJ_CORE_ENUMERATION
import CoreImageProcessing as OBJ_CORE_IP
import CoreFileProcessing as OBJ_CORE_FILE
import CoreLog_lib as OBJ_CORE_LOG

# user-defined mission libraries
import SARImage_ver as OBJ_MIS_IMAGE
import MissionConstantsPath_def as OBJ_MIS_PATH
import MissionEnumaration_def as OBJ_MIS_ENUMERATION
import MissionRequirementVerification as OBJ_MIS_REQUIREMENTS
import MissionRequirementExtraction as OBJ_MIS_REQEXTRACTION
import MissionConsoleMsg as OBJ_MIS_CONSOLE
import UseCaseConstants_def as OBJ_MIS_UC

import XMLProcessing as OBJ_XML
            
# ========================================================================================
# ==================================== MAIN FUNCTION =====================================
# ========================================================================================
def fcn_mainProcessFolderContent(
    TypeOfImages,
    extractedStrings,
    ReferenceCase,
    Path_Tiles,  # Path to 'Tiles'
    Path_Tiles_0,
    Path_Raw,
    Path_Labels,
    Path_Mask,
    verbosity: OBJ_CORE_ENUMERATION.Verbosity,  # Verbosity level for logging/output
    imgFormat: OBJ_CORE_ENUMERATION.ImageFormat,  # Image format constant (for filtering)
    logSave: OBJ_CORE_ENUMERATION.LoggingValidity,  # Logging validity (whether or not to save logs)
    logFlag: OBJ_MIS_ENUMERATION.LogDetailLevel  # Log detail level (flag for detailed logs)
):   
    if TypeOfImages == 'L1':
        folderTestPath = Path_Tiles
    if TypeOfImages == 'Raw':
        folderTestPath = Path_Raw
    if TypeOfImages == 'PNG':
        folderTestPath = Path_Mask
    #Remove backslash - it is added by the system when calling *.py from *.sh
    folderTestPath = OBJ_CORE_FILE.remove_trailing_backslash(folderTestPath)
        
    # log file name initialization    
    log_filename = ""
    # asume the folder contains valid content for the use case
    ValidUseCasePerFolder = True
    
    log_filename = OBJ_CORE_LOG.fcn_GenerateLogFile("LogsTestFolder", folderTestPath, f"{ReferenceCase}", verbosity)
    OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"\n\t{ReferenceCase}\n", verbosity)       
    # flags for requirement verification
    UR_GEN_DAT_01_status = False
    SR_DAT_FUN_03_status = False
    SR_DAT_FUN_02_status = False
    SR_DAT_FUN_09_status = False
    IndexFailed_UR_GEN_DAT_01 = 0
    IndexFailed_SR_DAT_FUN_03 = 0 
    IndexFailed_SR_DAT_FUN_09 = 0
    IndexValid_UR_GEN_DAT_01 = 0
    IndexValid_SR_DAT_FUN_03 = 0 
    IndexValid_SR_DAT_FUN_09 = 0
    ErrMsg_UR_GEN_DAT_01 = ""
    ErrMsg_SR_DAT_FUN_09 = ""
    ErrMsg_SR_DAT_FUN_03 = ""
    
    RAW_FORMAT_status = False
    IndexFailed_RAW_FORMAT = 0
    IndexValid_RAW_FORMAT = 0
    ErrMsg_RAW_FORMAT = ""
    RAW_CORRESPONDENT_status = False
    IndexFailed_RAW_CORRESPONDENT = 0
    IndexValid_RAW_CORRESPONDENT = 0
    ErrMsg_RAW_CORRESPONDENT = ""
    
    NumberOfSARProducts = 0
    IndexSLC = 0
    countRAW = 0 
    countXML = 0
    countL1 = 0
    countL1_0 = 0
    countMask = 0

    num_samples = 0 # number of identified images in folder
    extra_samples = 0 # number of identified images in folder that are not following the convention
    
    NumOfMatchRawXML = 0
    NumOfRaw = 0
    FormatValidity = True
    
    # print folder path
    if verbosity.value > OBJ_CORE_ENUMERATION.Verbosity.NONE.value:  # Fixed indentation here
        print(f'Processing Path: {folderTestPath}')
    
    # identify SAR images in the folder
    if OBJ_CORE_FILE.fcn_VerificationExistFilesInFolder(folderTestPath):
        # process each file in folder
        
        for filename in OBJ_CORE_FILE.fcn_GetFileNamesInFolder(folderTestPath):
            # process L1 images
            if TypeOfImages == 'L1':
                
                [FcnStatus, ImgType, ImgFormatExtracted] = OBJ_MIS_IMAGE.fcn_AssertImageName(filename, OBJ_MIS_PATH.PATH_SAR_IMG_NAME_DEFINITION, ReferenceCase, verbosity)
                if FcnStatus == 1:
                    num_samples = num_samples + 1
                    [Width, Height] = OBJ_MIS_IMAGE.fcn_ProcessSAR(TypeOfImages, folderTestPath, filename, ImgFormatExtracted, verbosity)
                    
                    # Requirement verification   
                    [UR_GEN_DAT_01_status, ErrMsg_UR_GEN_DAT_01, Case] = OBJ_MIS_REQUIREMENTS.fcn_VerReq_UR_GEN_DAT_01(filename,ReferenceCase, OBJ_CORE_ENUMERATION.Verbosity.LOW)  
                    [SR_DAT_FUN_03_status, ErrMsg_SR_DAT_FUN_03] = OBJ_MIS_REQUIREMENTS.fcn_VerReq_SR_DAT_FUN_03(TypeOfImages, ReferenceCase, ImgType, Width, Height, OBJ_CORE_ENUMERATION.Verbosity.LOW)
                    [SR_DAT_FUN_09_status, ErrMsg_SR_DAT_FUN_09] = OBJ_MIS_REQUIREMENTS.fcn_VerReq_SR_DAT_FUN_09(ImgType, filename)      
                    
                    [IndexValid_UR_GEN_DAT_01, IndexFailed_UR_GEN_DAT_01,IndexValid_SR_DAT_FUN_03, IndexFailed_SR_DAT_FUN_03,IndexValid_SR_DAT_FUN_09, IndexFailed_SR_DAT_FUN_09, IndexSLC] = OBJ_MIS_REQUIREMENTS.fcn_LogErrors(
                        TypeOfImages,
                        filename, 
                        log_filename, Case, ReferenceCase,
                        UR_GEN_DAT_01_status, ErrMsg_UR_GEN_DAT_01, IndexValid_UR_GEN_DAT_01, IndexFailed_UR_GEN_DAT_01,
                        SR_DAT_FUN_03_status, ErrMsg_SR_DAT_FUN_03, IndexValid_SR_DAT_FUN_03, IndexFailed_SR_DAT_FUN_03, IndexSLC,
                        SR_DAT_FUN_09_status, ErrMsg_SR_DAT_FUN_09, IndexValid_SR_DAT_FUN_09, IndexFailed_SR_DAT_FUN_09,
                        verbosity)
                                
                    # Display the image
                    if verbosity.value > OBJ_CORE_ENUMERATION.Verbosity.HIGH.value:  # Fixed indentation here
                        OBJ_CORE_IP.fcn_ReadSARImagePillow(os.path.join(folderTestPath, filename), verbosity, OBJ_CORE_CONSTANT.DISPLAY_INTERVAL)
                else:
                    extra_samples = extra_samples+1  
                    ValidUseCasePerFolder = False
                    IndexFailed_UR_GEN_DAT_01 = IndexFailed_UR_GEN_DAT_01 + 1
                    IndexFailed_SR_DAT_FUN_03 = IndexFailed_SR_DAT_FUN_03 + 1
                    IndexFailed_SR_DAT_FUN_09 = IndexFailed_SR_DAT_FUN_09 + 1
                    OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"- Error in {filename}: ['Invalid use-case identified in folder.']", verbosity)                    
                    print(f"- {OBJ_CORE_ENUMERATION.Colors.RED.value}Error{OBJ_CORE_ENUMERATION.Colors.RESET.value}: ['{OBJ_CORE_ENUMERATION.Colors.YELLOW.value}Invalid use-case identified in folder{OBJ_CORE_ENUMERATION.Colors.RESET.value}']")
                [SR_DAT_FUN_02_status, ErrMsg_SR_DAT_FUN_02] = OBJ_MIS_REQUIREMENTS.fcn_VerReq_SR_DAT_FUN_02(num_samples+extra_samples, verbosity)         
    
            # process Raw images
            if TypeOfImages == 'Raw':
                if ReferenceCase == 'Flood':
                    Case = 'FD'
                if ReferenceCase == 'Vessel':
                    Case = 'VD'
                if ReferenceCase == 'RFI':   
                    Case = 'RFI'
                
                prevVal = NumOfMatchRawXML
                [FormatValidity, NumOfRaw, NumOfMatchRawXML] = OBJ_MIS_IMAGE.fcn_AssertDat(filename, extractedStrings, NumOfMatchRawXML, NumOfRaw, FormatValidity)
                num_samples = num_samples + 1
                
                [RAW_FORMAT_status, ErrMsg_RAW_FORMAT] = OBJ_MIS_REQUIREMENTS.fcn_VerReq_UR_GEN_DAT_01_RawFormat(filename, FormatValidity, OBJ_CORE_ENUMERATION.Verbosity.LOW) 
                [RAW_CORRESPONDENT_status, ErrMsg_RAW_CORRESPONDENT] = OBJ_MIS_REQUIREMENTS.fcn_VerReq_UR_GEN_DAT_01_RawCorrespondent(filename, NumOfMatchRawXML-prevVal, OBJ_CORE_ENUMERATION.Verbosity.LOW) 
                
                [IndexValid_RAW_FORMAT, IndexFailed_RAW_FORMAT,IndexValid_RAW_CORRESPONDENT, IndexFailed_RAW_CORRESPONDENT] = OBJ_MIS_REQUIREMENTS.fcn_LogErrorsRaw(
                        filename, log_filename, Case, ReferenceCase,
                        RAW_FORMAT_status, IndexFailed_RAW_FORMAT, IndexValid_RAW_FORMAT, ErrMsg_RAW_FORMAT,
                        RAW_CORRESPONDENT_status, IndexFailed_RAW_CORRESPONDENT, IndexValid_RAW_CORRESPONDENT, ErrMsg_RAW_CORRESPONDENT,
                        verbosity)
            
            # process Mask (RFI)
            if TypeOfImages == 'PNG':
                [FcnStatus, ImgType, ImgFormatExtracted] = OBJ_MIS_IMAGE.fcn_AssertImageName(filename, OBJ_MIS_PATH.PATH_SAR_IMG_NAME_DEFINITION, ReferenceCase, verbosity)
                if FcnStatus == 1:
                    num_samples = num_samples + 1
                    [Width, Height] = OBJ_MIS_IMAGE.fcn_ProcessSAR(TypeOfImages, folderTestPath, filename, ImgFormatExtracted, verbosity)
                    
                    # Requirement verification   
                    [UR_GEN_DAT_01_status, ErrMsg_UR_GEN_DAT_01, Case] = OBJ_MIS_REQUIREMENTS.fcn_VerReq_UR_GEN_DAT_01(filename,ReferenceCase, OBJ_CORE_ENUMERATION.Verbosity.LOW)   
                    
                    [IndexValid_UR_GEN_DAT_01, IndexFailed_UR_GEN_DAT_01,IndexValid_SR_DAT_FUN_03, IndexFailed_SR_DAT_FUN_03,IndexValid_SR_DAT_FUN_09, IndexFailed_SR_DAT_FUN_09, IndexSLC] = OBJ_MIS_REQUIREMENTS.fcn_LogErrors(
                        TypeOfImages, filename, 
                        log_filename, Case, ReferenceCase,
                        UR_GEN_DAT_01_status, ErrMsg_UR_GEN_DAT_01, IndexValid_UR_GEN_DAT_01, IndexFailed_UR_GEN_DAT_01,
                        SR_DAT_FUN_03_status, ErrMsg_SR_DAT_FUN_03, IndexValid_SR_DAT_FUN_03, IndexFailed_SR_DAT_FUN_03, IndexSLC,
                        SR_DAT_FUN_09_status, ErrMsg_SR_DAT_FUN_09, IndexValid_SR_DAT_FUN_09, IndexFailed_SR_DAT_FUN_09,
                        verbosity)
                                
                    # Display the image
                    if verbosity.value > OBJ_CORE_ENUMERATION.Verbosity.HIGH.value:  # Fixed indentation here
                        OBJ_CORE_IP.fcn_ReadSARImagePillow(os.path.join(folderTestPath, filename), verbosity, OBJ_CORE_CONSTANT.DISPLAY_INTERVAL)
                else:
                    extra_samples = extra_samples+1  
                    ValidUseCasePerFolder = False
                    IndexFailed_UR_GEN_DAT_01 = IndexFailed_UR_GEN_DAT_01 + 1
                    IndexFailed_SR_DAT_FUN_03 = IndexFailed_SR_DAT_FUN_03 + 1
                    IndexFailed_SR_DAT_FUN_09 = IndexFailed_SR_DAT_FUN_09 + 1
                    OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"- Error in {filename}: ['Invalid use-case identified in folder.']", verbosity)                    
                    print(f"- {OBJ_CORE_ENUMERATION.Colors.RED.value}Error{OBJ_CORE_ENUMERATION.Colors.RESET.value}: ['{OBJ_CORE_ENUMERATION.Colors.YELLOW.value}Invalid use-case identified in folder{OBJ_CORE_ENUMERATION.Colors.RESET.value}']")
                [SR_DAT_FUN_02_status, ErrMsg_SR_DAT_FUN_02] = OBJ_MIS_REQUIREMENTS.fcn_VerReq_SR_DAT_FUN_02(num_samples+extra_samples, verbosity)
            
        if TypeOfImages == 'Raw':    
            NumberOfSARProducts = OBJ_MIS_REQUIREMENTS.fcn_VerReq_NumberOfSARProducts(extractedStrings)
        if TypeOfImages == 'Raw' or TypeOfImages == 'PNG':    
            if Path_Labels.endswith("\\"):
                Path_Labels = Path_Labels[:-1]

            [countRAW, countXML, countL1, countL1_0, countMask] = OBJ_CORE_FILE.fcn_CountFilesInFolder(
                    ReferenceCase, countRAW, countXML, countL1, countL1_0, countMask,
                    Path_Tiles, Path_Tiles_0, Path_Raw, Path_Labels, Path_Mask)
                           
        OBJ_MIS_REQUIREMENTS.fcn_LogValidFolder(TypeOfImages, FormatValidity, NumOfRaw, NumOfMatchRawXML,
            log_filename, num_samples, extra_samples, IndexFailed_UR_GEN_DAT_01, IndexValid_SR_DAT_FUN_03,
            IndexFailed_SR_DAT_FUN_03, IndexFailed_SR_DAT_FUN_09, SR_DAT_FUN_02_status, NumberOfSARProducts, 
            IndexSLC, countRAW, countXML, countL1, countL1_0, countMask,
            verbosity) 
    else:
        # Requirement verification
        OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"- Error: folder is missing.", verbosity)                    
        print(f"- {OBJ_CORE_ENUMERATION.Colors.RED.value}Error{OBJ_CORE_ENUMERATION.Colors.RESET.value}: ['{OBJ_CORE_ENUMERATION.Colors.YELLOW.value}Folder is missing{OBJ_CORE_ENUMERATION.Colors.RESET.value}']")
        
        OBJ_MIS_REQUIREMENTS.fcn_LogInvalidFolder(TypeOfImages, log_filename, verbosity)  
        
    # processing complete   
    if verbosity.value > OBJ_CORE_ENUMERATION.Verbosity.NONE.value:  # Fixed indentation here         
        print(f'Processing: {OBJ_CORE_ENUMERATION.Colors.GREEN.value}Complete{OBJ_CORE_ENUMERATION.Colors.RESET.value}.')     
# ========================================================================================
# ====================================== EXECUTION =======================================
# ========================================================================================
if __name__ == "__main__":   
    if len(sys.argv) == 2:
        TypeOfImages = sys.argv[1]
        
    if len(sys.argv) < 2:
        TypeOfImages = 'Raw'
        
    if len(sys.argv) > 2:
        TypeOfImages = sys.argv[1]
        selectionFromArguments = True
        UCselected = int(sys.argv[2])
        VERselected = int(sys.argv[3])
        LOGselected = int(sys.argv[4])
        PATH__Tiles = str(sys.argv[5]) + "\\"
        PATH__Tiles_0 = str(sys.argv[6]) + "\\"
        PATH__Raw = str(sys.argv[7]) + "\\"
        PATH__Labels = str(sys.argv[8]) + "\\"
        PATH__Mask = str(sys.argv[9])  + "\\" # applicable only for RFI
    else: 
        selectionFromArguments = False
        UCselected = int(-1)
        VERselected = int(-1)
        LOGselected = int(-1)
        PATH__Tiles = str("")
        PATH__Tiles_0 = str("") # applicable only for VD
        PATH__Raw = str("")
        PATH__Labels = str("")
        PATH__Mask = str("") # applicable only for RFI
        
    # Define the Excel file path
    path_generation_scripts = os.path.join(current_dir, "Mission")  # Get absolute path
    req_excel_file_path = os.path.join(path_generation_scripts, OBJ_MIS_PATH.PATH_REQUIREMENTS_EXCEL)

    # Extract requirements
    requirements = OBJ_MIS_REQEXTRACTION.fcn_ExtractDatasetRequirements(req_excel_file_path)

    # Check if the DataFrame is not empty
    if not requirements.empty:
        # Extract IDs as a string array
        ReqIDs = requirements["ID"].astype(str).tolist()
    else:
        print("No requirements extracted.")

    OBJ_CORE_LOG.fcn_PrintBanner()
    # Print the content of ReqIDs
    print("Applicable requirements for dataset validation:")
    for req_id in ReqIDs:
        print(req_id)
    print(f'\n')
    
    # use select cases
    if OBJ_MIS_UC.Automatic_UC_Selection_in_Console is False:
        extractedStrings = ''
        [UC_Selected, VER_Selected, LOG_Selected] = OBJ_MIS_CONSOLE.fcn_ConsoleSelection(selectionFromArguments, UCselected, VERselected, LOGselected)
        # if all selected
        if UC_Selected < OBJ_MIS_ENUMERATION.UseCaseSelector.UC_ALL.value:
            if TypeOfImages == 'L1':          
                # clear log folder (delete old log files)
                if LOG_Selected == OBJ_CORE_ENUMERATION.LoggingValidity.DELETE_LOG.value:
                    OBJ_CORE_LOG.fcn_DeleteAllLogFiles(OBJ_MIS_PATH.PATH_LOG)
                if UC_Selected == OBJ_MIS_ENUMERATION.UseCaseSelector.UV_FLOOD.value:
                    # run Floods detection
                    fcn_mainProcessFolderContent(TypeOfImages, extractedStrings, OBJ_MIS_PATH.FLOOD_USECASE, 
                                                 PATH__Tiles, PATH__Tiles_0, PATH__Raw, PATH__Labels, PATH__Mask,
                                                 OBJ_CORE_ENUMERATION.Verbosity.LOW, 
                                                 OBJ_CORE_ENUMERATION.ImageFormat.TIFF.value,
                                                 OBJ_CORE_ENUMERATION.LoggingValidity.SAVE_LOG,
                                                 OBJ_MIS_ENUMERATION.LogDetailLevel.DETAILED)
                if UC_Selected == OBJ_MIS_ENUMERATION.UseCaseSelector.UC_VESSEL.value:
                    time.sleep(OBJ_CORE_CONSTANT.DELAY_LOG)  # Pauses the program for 1 second
                    # run Vessel detection
                    fcn_mainProcessFolderContent(TypeOfImages, extractedStrings, OBJ_MIS_PATH.VESSELS_USECASE, 
                                                 PATH__Tiles, PATH__Tiles_0, PATH__Raw, PATH__Labels, PATH__Mask,
                                                 OBJ_CORE_ENUMERATION.Verbosity.LOW, 
                                                 OBJ_CORE_ENUMERATION.ImageFormat.TIFF.value,
                                                 OBJ_CORE_ENUMERATION.LoggingValidity.SAVE_LOG,
                                                 OBJ_MIS_ENUMERATION.LogDetailLevel.DETAILED)
                    
                if UC_Selected == OBJ_MIS_ENUMERATION.UseCaseSelector.UC_RFI.value:
                    time.sleep(OBJ_CORE_CONSTANT.DELAY_LOG)  # Pauses the program for 1 second
                    # run RFI
                    fcn_mainProcessFolderContent(TypeOfImages, extractedStrings, OBJ_MIS_PATH.RFI_USECASE, 
                                                 PATH__Tiles, PATH__Tiles_0, PATH__Raw, PATH__Labels, PATH__Mask,
                                                 OBJ_CORE_ENUMERATION.Verbosity.LOW, 
                                                 OBJ_CORE_ENUMERATION.ImageFormat.TIFF.value,
                                                 OBJ_CORE_ENUMERATION.LoggingValidity.SAVE_LOG,
                                                 OBJ_MIS_ENUMERATION.LogDetailLevel.DETAILED)
                    
            if TypeOfImages == 'Raw':
                # clear log folder (delete old log files)
                if LOG_Selected == OBJ_CORE_ENUMERATION.LoggingValidity.DELETE_LOG.value:
                    #print(LOG_Selected)
                    OBJ_CORE_LOG.fcn_DeleteAllLogFiles(OBJ_MIS_PATH.PATH_LOG)
                if UC_Selected == OBJ_MIS_ENUMERATION.UseCaseSelector.UV_FLOOD.value:
                    # run Floods detection
                    extractedStrings = OBJ_XML.fcn_XMLextract(PATH__Labels)
                    fcn_mainProcessFolderContent(TypeOfImages, extractedStrings, OBJ_MIS_PATH.FLOOD_USECASE, 
                                                 PATH__Tiles, PATH__Tiles_0, PATH__Raw, PATH__Labels, PATH__Mask,
                                                 OBJ_CORE_ENUMERATION.Verbosity.LOW, 
                                                 OBJ_CORE_ENUMERATION.ImageFormat.TIFF.value,
                                                 OBJ_CORE_ENUMERATION.LoggingValidity.SAVE_LOG,
                                                 OBJ_MIS_ENUMERATION.LogDetailLevel.DETAILED)
                    
                if UC_Selected == OBJ_MIS_ENUMERATION.UseCaseSelector.UC_VESSEL.value:
                    time.sleep(OBJ_CORE_CONSTANT.DELAY_LOG)  # Pauses the program for 1 second
                    # run Vessel detection
                    extractedStrings = OBJ_XML.fcn_XMLextract(PATH__Labels)
                    fcn_mainProcessFolderContent(TypeOfImages, extractedStrings, OBJ_MIS_PATH.VESSELS_USECASE, 
                                                 PATH__Tiles, PATH__Tiles_0, PATH__Raw, PATH__Labels, PATH__Mask,
                                                 OBJ_CORE_ENUMERATION.Verbosity.LOW, 
                                                 OBJ_CORE_ENUMERATION.ImageFormat.TIFF.value,
                                                 OBJ_CORE_ENUMERATION.LoggingValidity.SAVE_LOG,
                                                 OBJ_MIS_ENUMERATION.LogDetailLevel.DETAILED)
                    
                if UC_Selected == OBJ_MIS_ENUMERATION.UseCaseSelector.UC_RFI.value:
                    time.sleep(OBJ_CORE_CONSTANT.DELAY_LOG)  # Pauses the program for 1 second
                    # run RFI
                    extractedStrings = OBJ_XML.fcn_XMLextract(PATH__Labels)
                    fcn_mainProcessFolderContent(TypeOfImages, extractedStrings, OBJ_MIS_PATH.RFI_USECASE, 
                                                 PATH__Tiles, PATH__Tiles_0, PATH__Raw, PATH__Labels, PATH__Mask,
                                                 OBJ_CORE_ENUMERATION.Verbosity.LOW, 
                                                 OBJ_CORE_ENUMERATION.ImageFormat.TIFF.value,
                                                 OBJ_CORE_ENUMERATION.LoggingValidity.SAVE_LOG,
                                                 OBJ_MIS_ENUMERATION.LogDetailLevel.DETAILED)
            
            if TypeOfImages == 'PNG':
                # clear log folder (delete old log files)
                if LOG_Selected == OBJ_CORE_ENUMERATION.LoggingValidity.DELETE_LOG.value:
                    #print(LOG_Selected)
                    OBJ_CORE_LOG.fcn_DeleteAllLogFiles(OBJ_MIS_PATH.PATH_LOG)
                    
                if UC_Selected == OBJ_MIS_ENUMERATION.UseCaseSelector.UC_RFI.value:
                    time.sleep(OBJ_CORE_CONSTANT.DELAY_LOG)  # Pauses the program for 1 second
                    # run RFI
                    extractedStrings = OBJ_XML.fcn_XMLextract(PATH__Labels)
                    fcn_mainProcessFolderContent(TypeOfImages, extractedStrings, OBJ_MIS_PATH.RFI_USECASE, 
                                                 PATH__Tiles, PATH__Tiles_0, PATH__Raw, PATH__Labels, PATH__Mask,
                                                 OBJ_CORE_ENUMERATION.Verbosity.LOW, 
                                                 OBJ_CORE_ENUMERATION.ImageFormat.TIFF.value,
                                                 OBJ_CORE_ENUMERATION.LoggingValidity.SAVE_LOG,
                                                 OBJ_MIS_ENUMERATION.LogDetailLevel.DETAILED)
                
    # automaticaly select all cases
    else:
        OBJ_CORE_LOG.fcn_DeleteAllLogFiles(OBJ_MIS_PATH.PATH_LOG)
        extractedStrings = ''
        if TypeOfImages == 'L1':
            fcn_mainProcessFolderContent(TypeOfImages, extractedStrings, OBJ_MIS_PATH.FLOOD_USECASE, 
                                         PATH__Tiles, PATH__Tiles_0, PATH__Raw, PATH__Labels, PATH__Mask, 
                                         OBJ_CORE_ENUMERATION.Verbosity.LOW, 
                                         OBJ_CORE_ENUMERATION.ImageFormat.TIFF.value, 
                                         OBJ_CORE_ENUMERATION.LoggingValidity.SAVE_LOG, 
                                         OBJ_MIS_ENUMERATION.LogDetailLevel.DETAILED)
            time.sleep(OBJ_CORE_CONSTANT.DELAY_LOG)  # Pauses the program for 1 second
            
            fcn_mainProcessFolderContent(TypeOfImages, extractedStrings, OBJ_MIS_PATH.VESSELS_USECASE, 
                                         PATH__Tiles, PATH__Tiles_0, PATH__Raw, PATH__Labels, PATH__Mask,
                                         OBJ_CORE_ENUMERATION.Verbosity.LOW, 
                                         OBJ_CORE_ENUMERATION.ImageFormat.TIFF.value,
                                         OBJ_CORE_ENUMERATION.LoggingValidity.SAVE_LOG,
                                         OBJ_MIS_ENUMERATION.LogDetailLevel.DETAILED)
            time.sleep(OBJ_CORE_CONSTANT.DELAY_LOG)  # Pauses the program for 1 second
            
            fcn_mainProcessFolderContent(TypeOfImages, extractedStrings, OBJ_MIS_PATH.RFI_USECASE, 
                                         PATH__Tiles, PATH__Tiles_0, PATH__Raw, PATH__Labels, PATH__Mask,
                                         OBJ_CORE_ENUMERATION.Verbosity.LOW, 
                                         OBJ_CORE_ENUMERATION.ImageFormat.TIFF.value,
                                         OBJ_CORE_ENUMERATION.LoggingValidity.SAVE_LOG,
                                         OBJ_MIS_ENUMERATION.LogDetailLevel.DETAILED)
        if TypeOfImages == 'Raw':
            
            extractedStrings = OBJ_XML.fcn_XMLextract(OBJ_MIS_PATH.PATH_VESSELS_XML)
            
            fcn_mainProcessFolderContent(TypeOfImages, extractedStrings, OBJ_MIS_PATH.FLOOD_USECASE, 
                                         PATH__Tiles, PATH__Tiles_0, PATH__Raw, PATH__Labels, PATH__Mask,
                                         OBJ_CORE_ENUMERATION.Verbosity.LOW, 
                                         OBJ_CORE_ENUMERATION.ImageFormat.TIFF.value,
                                         OBJ_CORE_ENUMERATION.LoggingValidity.SAVE_LOG,
                                         OBJ_MIS_ENUMERATION.LogDetailLevel.DETAILED)
            time.sleep(OBJ_CORE_CONSTANT.DELAY_LOG)  # Pauses the program for 1 second
            
            fcn_mainProcessFolderContent(TypeOfImages, extractedStrings, OBJ_MIS_PATH.VESSELS_USECASE, 
                                         PATH__Tiles, PATH__Tiles_0, PATH__Raw, PATH__Labels, PATH__Mask,
                                         OBJ_CORE_ENUMERATION.Verbosity.LOW, 
                                         OBJ_CORE_ENUMERATION.ImageFormat.TIFF.value,
                                         OBJ_CORE_ENUMERATION.LoggingValidity.SAVE_LOG,
                                         OBJ_MIS_ENUMERATION.LogDetailLevel.DETAILED)
            time.sleep(OBJ_CORE_CONSTANT.DELAY_LOG)  # Pauses the program for 1 second
            
            fcn_mainProcessFolderContent(TypeOfImages,extractedStrings, OBJ_MIS_PATH.RFI_USECASE, 
                                         PATH__Tiles, PATH__Tiles_0, PATH__Raw, PATH__Labels, PATH__Mask,
                                         OBJ_CORE_ENUMERATION.Verbosity.LOW, 
                                         OBJ_CORE_ENUMERATION.ImageFormat.TIFF.value,
                                         OBJ_CORE_ENUMERATION.LoggingValidity.SAVE_LOG,
                                         OBJ_MIS_ENUMERATION.LogDetailLevel.DETAILED)    
