import CoreAssert_lib as OBJ_CORE_ASSERT
import CoreEnumeration_def as OBJ_CORE_ENUM
import RadarConstants_def as OBJ_MIS_RADAR
import CoreLog_lib as OBJ_CORE_LOG
import UseCaseConstants_def as OBJ_MIS_UCCONSTANTS

def fcn_VerReq_UR_GEN_DAT_01_RawFormat(fileName, FormatValidity, verbosity: OBJ_CORE_ENUM.Verbosity):
    ErrCode = True
    ErrMessage = "Raw image format is valid."
    if FormatValidity == True:
        return ErrCode, ErrMessage
    else:
        ErrMessage = "Raw image format is invalid: expected *.dat file."
        return FormatValidity, ErrMessage
    
def fcn_VerReq_UR_GEN_DAT_01_RawCorrespondent(filename, match, verbosity: OBJ_CORE_ENUM.Verbosity.LOW):
    ErrCode = True
    ErrMessage = "Raw image match valid."
    if match > 0:
        return ErrCode, ErrMessage
    else:
        ErrCode = False
        ErrMessage = "Raw image does not have a correspondent XML file"
        return ErrCode, ErrMessage
  
def fcn_VerReq_NumberOfSARProducts(extractedStrings):
    unique_products = set(extractedStrings)
    return len(unique_products)

# Verify if Use case reference is integrated in the file name
def fcn_VerReq_UR_GEN_DAT_01(fileName, ReferenceCase, verbosity: OBJ_CORE_ENUM.Verbosity):
    ErrCode = True
    ErrMessage = "Applicable use-case is valid."
    # Correct the call to fcn_AssertString by passing the verbosity directly
    if OBJ_CORE_ASSERT.fcn_AssertString(fileName, OBJ_MIS_UCCONSTANTS.strFloodDetection, verbosity, 11, 0, 2):
        return ErrCode, ErrMessage, OBJ_MIS_UCCONSTANTS.strFloodDetection
    if OBJ_CORE_ASSERT.fcn_AssertString(fileName, OBJ_MIS_UCCONSTANTS.strRadioFrequencyInterference, verbosity, 11, 0, 3):
        return ErrCode, ErrMessage, OBJ_MIS_UCCONSTANTS.strRadioFrequencyInterference
    if OBJ_CORE_ASSERT.fcn_AssertString(fileName, OBJ_MIS_UCCONSTANTS.strVesselDetection, verbosity, 11, 0, 2):
        return ErrCode, ErrMessage, OBJ_MIS_UCCONSTANTS.strVesselDetection
    ErrCode = False
    ErrMessage = "Applicable use-case is invalid."
    if ReferenceCase == 'Flood':
        Case = 'FD'
    if ReferenceCase == 'Vessel':
        Case = 'VD'
    if ReferenceCase == 'RFI':      
        Case = 'RFI'
    return ErrCode, ErrMessage, Case

# Verify image resolution
def fcn_VerReq_SR_DAT_FUN_03(TypeOfImages, ReferenceCase, ImgType, imgWidth, imgHeight, verbosity: OBJ_CORE_ENUM.Verbosity):
    ErrCode = False
    ErrMessage = "SAR image resolution is invalid."
    if ImgType == "SLC":
        if TypeOfImages == 'L1' and ReferenceCase != 'RFI':
            if abs(imgWidth - OBJ_MIS_RADAR.RADAR_WIDTH) > 1 or abs(imgHeight - OBJ_MIS_RADAR.RADAR_HEIGH) > 1:
                ErrMessage = f"SAR image resolution is invalid. Actual: {imgWidth}x{imgHeight}"
                return ErrCode, ErrMessage
            else:
                ErrCode = True
                ErrMessage = "SAR image resolution is valid."
            return ErrCode, ErrMessage
        if TypeOfImages == 'L1' and ReferenceCase == 'RFI':
            ErrCode = False
            ErrMessage = "Warning: SAR image is SLC in RFI use case - requirement not applicable."
    else: 
        ErrCode = False
        ErrMessage = "Warning: SAR image is GRD - requirement not applicable."
    return ErrCode, ErrMessage

# Verify if there are at least 1000 sample
def fcn_VerReq_SR_DAT_FUN_02(numOfFiles, verbosity: OBJ_CORE_ENUM.Verbosity):
    ErrCode = True
    ErrMessage = "Total number of images per use-case is valid."
    # Correct the call to fcn_AssertString by passing the verbosity directly
    if OBJ_CORE_ASSERT.fcn_AssertInteger(numOfFiles, 1000, verbosity):
        return ErrCode, ErrMessage
    ErrCode = False
    ErrMessage = "Total number of images per use-case is invalid."
    return ErrCode, ErrMessage

#Search for the string "GRD" or "SLC" in the input string.    
def fcn_VerReq_SR_DAT_FUN_09(ImgType, input_string):
    ErrMessage = "SAR image product is valid."
    if ImgType == 'RFI':
        return True, ErrMessage
    else:    
        ErrCode = "GRD" in input_string or "SLC" in input_string
        if ErrCode is False:
            ErrMessage = "SAR image product is invalid."
            return ErrCode, ErrMessage
        else:
            return ErrCode, ErrMessage
    

def fcn_LogErrorsRaw(filename, log_filename, Case, ReferenceCase,
                        RAW_FORMAT_status, IndexFailed_RAW_FORMAT, IndexValid_RAW_FORMAT, ErrMsg_RAW_FORMAT,
                        RAW_CORRESPONDENT_status, IndexFailed_RAW_CORRESPONDENT, IndexValid_RAW_CORRESPONDENT, ErrMsg_RAW_CORRESPONDENT,
                        verbosity):
    if OBJ_CORE_ASSERT.fcn_AssertString(Case, ReferenceCase, verbosity, 0, 0, 1) is True:
        if RAW_FORMAT_status is False:
            IndexFailed_RAW_FORMAT = IndexFailed_RAW_FORMAT + 1
            OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"- Error in {filename}: ['{ErrMsg_RAW_FORMAT}']", verbosity)                    
            print(f"- {OBJ_CORE_ENUM.Colors.RED.value}Error{OBJ_CORE_ENUM.Colors.RESET.value} in {filename}: ['{OBJ_CORE_ENUM.Colors.YELLOW.value}{ErrMsg_RAW_FORMAT}{OBJ_CORE_ENUM.Colors.RESET.value}']")
        else:
            IndexValid_RAW_FORMAT = IndexValid_RAW_FORMAT + 1
        if RAW_CORRESPONDENT_status is False:
            IndexFailed_RAW_CORRESPONDENT = IndexFailed_RAW_CORRESPONDENT + 1
            OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"- Error in {filename}: ['{ErrMsg_RAW_CORRESPONDENT}']", verbosity)                    
            print(f"- {OBJ_CORE_ENUM.Colors.RED.value}Error{OBJ_CORE_ENUM.Colors.RESET.value} in {filename}: ['{OBJ_CORE_ENUM.Colors.YELLOW.value}{ErrMsg_RAW_CORRESPONDENT}{OBJ_CORE_ENUM.Colors.RESET.value}']")
        else:
            IndexValid_RAW_CORRESPONDENT = IndexValid_RAW_CORRESPONDENT + 1   
    else:
        OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"- Error in {filename}: ['Invalid use-case identified in folder.']", verbosity)                    
        print(f"- {OBJ_CORE_ENUM.Colors.RED.value}Error{OBJ_CORE_ENUM.Colors.RESET.value}: ['{OBJ_CORE_ENUM.Colors.YELLOW.value}Invalid use-case identified in folder{OBJ_CORE_ENUM.Colors.RESET.value}']")
    return IndexValid_RAW_FORMAT, IndexFailed_RAW_FORMAT,IndexValid_RAW_CORRESPONDENT, IndexFailed_RAW_CORRESPONDENT
            

def fcn_LogErrors(TypeOfImages,
                  filename, log_filename, Case, ReferenceCase,
                  UR_GEN_DAT_01_status, ErrMsg_UR_GEN_DAT_01, IndexValid_UR_GEN_DAT_01, IndexFailed_UR_GEN_DAT_01,
                  SR_DAT_FUN_03_status, ErrMsg_SR_DAT_FUN_03, IndexValid_SR_DAT_FUN_03, IndexFailed_SR_DAT_FUN_03, IndexSLC,
                  SR_DAT_FUN_09_status, ErrMsg_SR_DAT_FUN_09, IndexValid_SR_DAT_FUN_09, IndexFailed_SR_DAT_FUN_09,
                  verbosity):
    if OBJ_CORE_ASSERT.fcn_AssertString(Case, ReferenceCase, verbosity, 0, 0, 1) is True:
        if UR_GEN_DAT_01_status is False:
            IndexFailed_UR_GEN_DAT_01 = IndexFailed_UR_GEN_DAT_01 + 1
            OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"- Error in {filename}: ['{ErrMsg_UR_GEN_DAT_01}']", verbosity)                    
            print(f"- {OBJ_CORE_ENUM.Colors.RED.value}Error{OBJ_CORE_ENUM.Colors.RESET.value} in {filename}: ['{OBJ_CORE_ENUM.Colors.YELLOW.value}{ErrMsg_UR_GEN_DAT_01}{OBJ_CORE_ENUM.Colors.RESET.value}']")
        else:
            IndexValid_UR_GEN_DAT_01 = IndexValid_UR_GEN_DAT_01 + 1
             
        if TypeOfImages == 'L1':  
        #if ReferenceCase != 'RFI':   
            if SR_DAT_FUN_03_status is False:
                if ErrMsg_SR_DAT_FUN_03.startswith("SAR image resolution is invalid."):
                    IndexFailed_SR_DAT_FUN_03 = IndexFailed_SR_DAT_FUN_03 + 1 
                    OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"- Error in {filename}: ['{ErrMsg_SR_DAT_FUN_03}']", verbosity)
                    print(f"- {OBJ_CORE_ENUM.Colors.RED.value}Error{OBJ_CORE_ENUM.Colors.RESET.value} in {filename}: ['{OBJ_CORE_ENUM.Colors.YELLOW.value}{ErrMsg_SR_DAT_FUN_03}{OBJ_CORE_ENUM.Colors.RESET.value}']")
                if ErrMsg_SR_DAT_FUN_03 == "Warning: SAR image is GRD - requirement not applicable." or ErrMsg_SR_DAT_FUN_03 == "Warning: SAR image is SLC in RFI use case - requirement not applicable.":
                    IndexFailed_SR_DAT_FUN_03 = IndexFailed_SR_DAT_FUN_03 + 1 
                    OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"- Warning in {filename}: ['{ErrMsg_SR_DAT_FUN_03}']", verbosity)
                    print(f"- {OBJ_CORE_ENUM.Colors.YELLOW.value}Warning{OBJ_CORE_ENUM.Colors.RESET.value} in {filename}: ['{OBJ_CORE_ENUM.Colors.YELLOW.value}{ErrMsg_SR_DAT_FUN_03}{OBJ_CORE_ENUM.Colors.RESET.value}']")
                    IndexSLC = IndexSLC + 1 
            else:
                IndexValid_SR_DAT_FUN_03 = IndexValid_SR_DAT_FUN_03 + 1 
                
            if SR_DAT_FUN_09_status is False:
                IndexFailed_SR_DAT_FUN_09 = IndexFailed_SR_DAT_FUN_09 + 1                        
                OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"- Error in {filename}: ['{ErrMsg_SR_DAT_FUN_09}']", verbosity)
                print(f"- {OBJ_CORE_ENUM.Colors.RED.value}Error{OBJ_CORE_ENUM.Colors.RESET.value} in {filename}: ['{OBJ_CORE_ENUM.Colors.YELLOW.value}{ErrMsg_SR_DAT_FUN_09}{OBJ_CORE_ENUM.Colors.RESET.value}']")
            else:
                IndexValid_SR_DAT_FUN_09 = IndexValid_SR_DAT_FUN_09 + 1
    else:
        OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"- Error in {filename}: ['Invalid use-case identified in folder.']", verbosity)                    
        print(f"- {OBJ_CORE_ENUM.Colors.RED.value}Error{OBJ_CORE_ENUM.Colors.RESET.value}: ['{OBJ_CORE_ENUM.Colors.YELLOW.value}Invalid use-case identified in folder{OBJ_CORE_ENUM.Colors.RESET.value}']")
    return IndexValid_UR_GEN_DAT_01, IndexFailed_UR_GEN_DAT_01,IndexValid_SR_DAT_FUN_03, IndexFailed_SR_DAT_FUN_03,IndexValid_SR_DAT_FUN_09, IndexFailed_SR_DAT_FUN_09, IndexSLC

def fcn_LogValidFolder(TypeOfImagess, FormatValidity, NumOfRaw, foundCorrespondent,
                       log_filename, num_samples, extra_samples, 
                       IndexFailed_UR_GEN_DAT_01, IndexValid_SR_DAT_FUN_03,
                       IndexFailed_SR_DAT_FUN_03, IndexFailed_SR_DAT_FUN_09,
                       SR_DAT_FUN_02_status, 
                       NumberOfSARProducts,
                       IndexSLC, countRAW, countXML, countL1,countL1_0,countMask,
                       verbosity):
    OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"\n", verbosity) 
      
    if TypeOfImagess == 'L1' or TypeOfImagess == 'PNG':   
        # TC1
        if IndexFailed_UR_GEN_DAT_01 == 0:   
            OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-1] - [UR_GEN_DAT_01]: ✅ - ✅ Valid Files: {num_samples+extra_samples-IndexFailed_UR_GEN_DAT_01} | ❌ Invalid Files: {IndexFailed_UR_GEN_DAT_01} | ⚠️ Warnings: 0 | Total Files: {num_samples+extra_samples}", verbosity)
        else:
            OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-1] - [UR_GEN_DAT_01]: ❌ - ✅ Valid Files: {num_samples+extra_samples-IndexFailed_UR_GEN_DAT_01} | ❌ Invalid Files: {IndexFailed_UR_GEN_DAT_01} | ⚠️ Warnings: 0 | Total Files: {num_samples+extra_samples}", verbosity) 
        # TC2
        if num_samples+extra_samples > 999:
            OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-2] - [SR_DAT_FUN_02]: ✅ - ✅ Number of Files: {num_samples+extra_samples} | Expected Minumum Number of Files: 1000", verbosity)
        else:
            OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-2] - [SR_DAT_FUN_02]: ❌ - ✅ Number of Files: {num_samples+extra_samples} | Expected Minumum Number of Files: 1000", verbosity)
        # TC3
        if TypeOfImagess == 'L1':
            if IndexValid_SR_DAT_FUN_03 == IndexSLC:
                OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-3] - [SR_DAT_FUN_03]: ✅ - ✅ Valid Files: {IndexValid_SR_DAT_FUN_03} | ❌ Invalid/Not applicable Files: {IndexFailed_SR_DAT_FUN_03} | ⚠️ GRD Files: {IndexSLC} | Total Files: {IndexValid_SR_DAT_FUN_03+IndexFailed_SR_DAT_FUN_03}", verbosity)
            else:
                OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-3] - [SR_DAT_FUN_03]: ❌ - ✅ Valid Files: {IndexValid_SR_DAT_FUN_03} | ❌ Invalid/Not applicable Files: {IndexFailed_SR_DAT_FUN_03} | ⚠️ GRD Files: {IndexSLC} | Total Files: {IndexValid_SR_DAT_FUN_03+IndexFailed_SR_DAT_FUN_03}", verbosity)    
            # TC4
            if IndexFailed_SR_DAT_FUN_09 == 0:
                OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-4] - [SR_DAT_FUN_09]: ✅ - ✅ Valid Files: {num_samples+extra_samples-IndexFailed_SR_DAT_FUN_09} | ❌ Invalid Files: {IndexFailed_SR_DAT_FUN_09} | ⚠️ Warnings: 0 | Total Files: {num_samples+extra_samples}", verbosity)
            else:
                OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-4] - [SR_DAT_FUN_09]: ❌ - ✅ Valid Files: {num_samples+extra_samples-IndexFailed_SR_DAT_FUN_09} | ❌ Invalid Files: {IndexFailed_SR_DAT_FUN_09} | ⚠️ Warnings: 0 | Total Files: {num_samples+extra_samples}", verbosity)  
        OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"\n", verbosity)
        
        
        if IndexFailed_UR_GEN_DAT_01 == 0:
            print(f"[TC-IMG-1] - [UR_GEN_DAT_01]: ✅ - ✅ Valid Files: {num_samples+extra_samples-IndexFailed_UR_GEN_DAT_01} | ❌ Invalid Files: {IndexFailed_UR_GEN_DAT_01} | ⚠️ Optional Files: 0 | Total Files: {num_samples+extra_samples}")
        else:
            print(f"[TC-IMG-1] - [UR_GEN_DAT_01]: ❌ - ✅Valid Files: {num_samples+extra_samples-IndexFailed_UR_GEN_DAT_01} | ❌ Invalid Files: {IndexFailed_UR_GEN_DAT_01} | ⚠️ Optional Files: 0 | Total Files: {num_samples+extra_samples}")
        if num_samples+extra_samples > 999:
            print(f"[TC-IMG-2] - [SR_DAT_FUN_02]: ✅ - ✅ Number of Files: {num_samples+extra_samples} | Expected Minumum Number of Files: 1000")
        else:
            print(f"[TC-IMG-2] - [SR_DAT_FUN_02]: ❌ - ✅ Number of Files: {num_samples+extra_samples} | Expected Minumum Number of Files: 1000")
        if TypeOfImagess == 'L1':
            if IndexFailed_SR_DAT_FUN_03 == IndexSLC:
                print(f"[TC-IMG-3] - [SR_DAT_FUN_03]: ✅ - ✅ Valid Files: {IndexValid_SR_DAT_FUN_03} | ❌ Invalid Files/Not applicable: {IndexFailed_SR_DAT_FUN_03} | ⚠️ GRD Files: {IndexSLC} | Total Files: {IndexValid_SR_DAT_FUN_03+IndexFailed_SR_DAT_FUN_03}")
            else:
                print(f"[TC-IMG-3] - [SR_DAT_FUN_03]: ❌ - ✅ Valid Files: {IndexValid_SR_DAT_FUN_03} | ❌ Invalid Files/Not applicable: {IndexFailed_SR_DAT_FUN_03} | ⚠️ GRD Files: {IndexSLC} | Total Files: {IndexValid_SR_DAT_FUN_03+IndexFailed_SR_DAT_FUN_03}")
            if IndexFailed_SR_DAT_FUN_09 == 0:
                print(f"[TC-IMG-4] - [SR_DAT_FUN_09]: ✅ - ✅ Valid Files: {num_samples+extra_samples-IndexFailed_SR_DAT_FUN_09} | ❌ Invalid Files: {IndexFailed_SR_DAT_FUN_09} | ⚠️ Optional Files: 0 | Total Files: {num_samples+extra_samples}")
            else:
                print(f"[TC-IMG-4] - [SR_DAT_FUN_09]: ❌ - ✅  Valid Files: {num_samples+extra_samples-IndexFailed_SR_DAT_FUN_09} | ❌ Invalid Files: {IndexFailed_SR_DAT_FUN_09} | ⚠️ Optional Files: 0 | Total Files: {num_samples+extra_samples}")
        
        if TypeOfImagess ==  'PNG':
            # check number of samples
            if countMask == countXML:
                OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-10]: ✅ - ✅ Number of Mask files: {countMask} | ✅ Number of XML Files: {countXML} | ⚠️ Warnings: 0 | Condition: {countMask} = {countXML}", verbosity)
                print(f"[TC-IMG-10]: ✅ - ✅ Number of Mask files: {countMask} | ✅ Number of XML Files: {countXML} | ⚠️ Warnings: 0 | Condition: {countMask} = {countXML}")
            else:
                OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-10]: ❌ - ✅ Number of Mask files: {countMask} | ✅ Number of XML Files: {countXML} | ⚠️ Warnings: 0 | Condition: {countMask} = {countXML}", verbosity)
                print(f"[TC-IMG-10]: ❌ - ✅ Number of Mask files: {countMask} | ✅ Number of XML Files: {countXML} | ⚠️ Warnings: 0 | Condition: {countMask} = {countXML}")
        
        #print(f"\n\n\n\n\n{IndexFailed_UR_GEN_DAT_01 == 0} {IndexValid_SR_DAT_FUN_03 == IndexSLC} {SR_DAT_FUN_02_status == True} {IndexFailed_SR_DAT_FUN_09 == 0}\n\n\n")
        if TypeOfImagess == 'L1':
            if IndexFailed_UR_GEN_DAT_01 == 0 and IndexValid_SR_DAT_FUN_03 == IndexSLC and num_samples+extra_samples > 999 and IndexFailed_SR_DAT_FUN_09 == 0:
                OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"✅ Validation succesful. Check {log_filename} for details.", verbosity)
                print(f"\n✅ Validation succesful. Check {OBJ_CORE_ENUM.Colors.GREEN.value}{log_filename}{OBJ_CORE_ENUM.Colors.RESET.value} for details.")
            else:
                OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"❌ Validation failed. Check {log_filename} for details.", verbosity)
                print(f"\n❌ Validation failed. Check {OBJ_CORE_ENUM.Colors.YELLOW.value}{log_filename}{OBJ_CORE_ENUM.Colors.RESET.value} for details.")
        if TypeOfImagess == 'PNG':
            if IndexFailed_UR_GEN_DAT_01 == 0 and IndexValid_SR_DAT_FUN_03 == IndexSLC and num_samples+extra_samples > 999 and IndexFailed_SR_DAT_FUN_09 == 0 and countMask == countXML:
                OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"✅ Validation succesful. Check {log_filename} for details.", verbosity)
                print(f"\n✅ Validation succesful. Check {OBJ_CORE_ENUM.Colors.GREEN.value}{log_filename}{OBJ_CORE_ENUM.Colors.RESET.value} for details.")
            else:
                OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"❌ Validation failed. Check {log_filename} for details.", verbosity)
                print(f"\n❌ Validation failed. Check {OBJ_CORE_ENUM.Colors.YELLOW.value}{log_filename}{OBJ_CORE_ENUM.Colors.RESET.value} for details.")

    if TypeOfImagess == 'Raw':
        if NumOfRaw == num_samples:
            OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-5] - [UR_GEN_DAT_01]: ✅ - ✅ Valid Files: {NumOfRaw} | ❌ Invalid Files: {num_samples-NumOfRaw} | ⚠️ Warnings: 0 | Total Files: {num_samples}", verbosity)
            print(f"\n[TC-IMG-5] - [UR_GEN_DAT_01]: ✅ - ✅ Valid Files: {NumOfRaw} | ❌ Invalid Files: {num_samples-NumOfRaw} | ⚠️ Warnings: 0 | Total Files: {num_samples}")
        else:
            OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-5] - [UR_GEN_DAT_01]: ❌ - ✅ Valid Files: {NumOfRaw} | ❌ Invalid Files: {num_samples-NumOfRaw} | ⚠️ Warnings: 0 | Total Files: {num_samples}", verbosity)
            print(f"\n[TC-IMG-5] - [UR_GEN_DAT_01]: ❌ - ✅ Valid Files: {NumOfRaw} | ❌ Invalid Files: {num_samples-NumOfRaw} | ⚠️ Warnings: 0 | Total Files: {num_samples}")
         
        # FormatValidity
        if foundCorrespondent == num_samples:
            OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-6] - [UR_GEN_DAT_01]: ✅ - ✅ Valid Files: {foundCorrespondent} | ❌ Invalid Files: {num_samples-foundCorrespondent} | ⚠️ Warnings: 0 | Total Files: {num_samples}", verbosity)
            print(f"[TC-IMG-6] - [UR_GEN_DAT_01]: ✅ - ✅ Valid Files: {foundCorrespondent} | ❌ Invalid Files: {num_samples-foundCorrespondent} | ⚠️ Warnings: 0 | Total Files: {num_samples}")
        else:
            OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-6] - [UR_GEN_DAT_01]: ❌ - ✅ Valid Files: {foundCorrespondent} | ❌ Invalid Files: {num_samples-foundCorrespondent} | ⚠️ Warnings: 0 | Total Files: {num_samples}", verbosity)
            print(f"[TC-IMG-6] - [UR_GEN_DAT_01]: ❌ - ✅ Valid Files: {foundCorrespondent} | ❌ Invalid Files: {num_samples-foundCorrespondent} | ⚠️ Warnings: 0 | Total Files: {num_samples}")
         
        # NumberOfSARProducts
        if NumberOfSARProducts > 9:
            OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-7] - [UR_GEN_DAT_01]: ✅ - ✅ Number of unique SAR Products: {NumberOfSARProducts} | Expected Minumum Number of Files: 10", verbosity)
            print(f"[TC-IMG-7] - [UR_GEN_DAT_01]: ✅ - ✅ Number of unique SAR Products: {NumberOfSARProducts} | Expected Minumum Number of Files: 10")
        else:
            OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-7] - [UR_GEN_DAT_01]: ❌ - - ✅ Number of unique SAR Products: {NumberOfSARProducts} | Expected Minumum Number of Files: 10", verbosity)
            print(f"[TC-IMG-7] - [UR_GEN_DAT_01]: ❌ - ✅ Number of unique SAR Products: {NumberOfSARProducts} | Expected Minumum Number of Files: 10") 
        
        # check number of samples
        if countRAW == countXML * 2:
            OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-8]: ✅ - ✅ Number of RAW files: {countRAW} | ✅ Number of XML Files: {countXML} | ⚠️ Warnings: 0 | Condition: {countRAW} = 2 x {countXML}", verbosity)
            print(f"[TC-IMG-8]: ✅ - ✅ Number of RAW files: {countRAW} | ✅ Number of XML Files: {countXML} | ⚠️ Warnings: 0 | Condition: {countRAW} = 2 x {countXML}")
        else:
            OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-8]: ❌ - ✅ Number of RAW files: {countRAW} | ✅ Number of XML Files: {countXML} | ⚠️ Warnings: 0 | Condition: {countRAW} = 2 x {countXML}", verbosity)
            print(f"[TC-IMG-8]: ❌ - ✅ Number of RAW files: {countRAW} | ✅ Number of XML Files: {countXML} | ⚠️ Warnings: 0 | Condition: {countRAW} = 2 x {countXML}")
            
        if 4*countXML == (countL1 + countL1_0):
            OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-9]: ✅ - ✅ Number of XML files: {countXML} | ✅ Number of L1 Files: {(countL1 + countL1_0)} | ⚠️ Warnings: 0 | Condition: {countL1 + countL1_0} = 4 x {(countXML)}", verbosity)
            print(f"[TC-IMG-9]: ✅ - ✅ Number of XML files: {countXML} | ✅ Number of L1 Files: {(countL1 + countL1_0)} | ⚠️ Warnings: 0 | Condition: {(countL1 + countL1_0)} = 4 x {(countL1 + countL1_0)}")
        else:
            OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-9]: ❌ - ✅ Number of XML files: {(countL1 + countL1_0)} | ✅ Number of L1 Files: {countXML} | ⚠️ Warnings: 0 | Condition: {countL1 + countL1_0} = 4 x {(countXML)}", verbosity)
            print(f"[TC-IMG-9]: ❌ - ✅ Number of XML files: {countXML} | ✅ Number of L1 Files: {(countL1 + countL1_0)} | ⚠️ Warnings: 0 | Condition: {countL1 + countL1_0} = 4 x {(countXML)}")
            
        #countL1
        
        # full valid
        if foundCorrespondent == num_samples and NumOfRaw == num_samples and countRAW == countXML * 2 and 4*countXML == (countL1 + countL1_0):
            OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"✅ Validation succesful. Check {log_filename} for details.", verbosity)
            print(f"\n✅ Validation succesful. Check {OBJ_CORE_ENUM.Colors.GREEN.value}{log_filename}{OBJ_CORE_ENUM.Colors.RESET.value} for details.")
        else:
            OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"❌ Validation failed. Check {log_filename} for details.", verbosity)
            print(f"\n❌ Validation failed. Check {OBJ_CORE_ENUM.Colors.GREEN.value}{log_filename}{OBJ_CORE_ENUM.Colors.RESET.value} for details.")
         
    
    OBJ_CORE_LOG.fcn_CloseLog(log_filename, verbosity)

def fcn_LogInvalidFolder(TypeOfImages, log_filename, verbosity):
    if TypeOfImages == 'L1':
        OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"\n", verbosity)
        OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-1] - [UR_GEN_DAT_01]: ❌ Folder is missing.", verbosity)
        OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-2] - [SR_DAT_FUN_02]: ❌ Folder is missing.", verbosity)
        OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-3] - [SR_DAT_FUN_03]: ❌ Folder is missing.", verbosity)
        OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-4] - [SR_DAT_FUN_09]: ❌ Folder is missing.", verbosity)
        OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"\n", verbosity)
        OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"❌ Validation failed. Invalid folder.", verbosity)
        OBJ_CORE_LOG.fcn_CloseLog(log_filename, verbosity)
        print(f"[TC-IMG-1] - [UR_GEN_DAT_01]: ❌ Folder is missing.")
        print(f"[TC-IMG-2] - [SR_DAT_FUN_02]: ❌ Folder is missing.")
        print(f"[TC-IMG-3] - [SR_DAT_FUN_03]: ❌ Folder is missing.")
        print(f"[TC-IMG-4] - [SR_DAT_FUN_09]: ❌ Folder is missing.")
        print(f"\n❌ Validation failed. Check {OBJ_CORE_ENUM.Colors.YELLOW.value}{log_filename}{OBJ_CORE_ENUM.Colors.RESET.value} for details.")
    if TypeOfImages == 'Raw':
        OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"\n", verbosity)
        OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-5] - [UR_GEN_DAT_01]: ❌ Folder is missing.", verbosity)
        OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-6] - [UR_GEN_DAT_01]: ❌ Folder is missing.", verbosity)
        OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-7] - [UR_GEN_DAT_01]: ❌ Folder is missing.", verbosity)
        OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-8]: ❌ Folder is missing.", verbosity)
        OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-9]: ❌ Folder is missing.", verbosity)
        OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"\n", verbosity)
        OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"❌ Validation failed. Invalid folder.", verbosity)
        OBJ_CORE_LOG.fcn_CloseLog(log_filename, verbosity)
        print(f"[TC-IMG-5] - [UR_GEN_DAT_01]: ❌ Folder is missing.")
        print(f"[TC-IMG-6] - [UR_GEN_DAT_01]: ❌ Folder is missing.")
        print(f"[TC-IMG-7] - [UR_GEN_DAT_01]: ❌ Folder is missing.")
        print(f"[TC-IMG-8]: ❌ Folder is missing.")
        print(f"[TC-IMG-9]: ❌ Folder is missing.")
        print(f"\n❌ Validation failed. Check {OBJ_CORE_ENUM.Colors.YELLOW.value}{log_filename}{OBJ_CORE_ENUM.Colors.RESET.value} for details.")


# import CoreAssert_lib as OBJ_CORE_ASSERT
# import CoreEnumeration_def as OBJ_CORE_ENUM
# import RadarConstants_def as OBJ_MIS_RADAR
# import CoreLog_lib as OBJ_CORE_LOG
# import UseCaseConstants_def as OBJ_MIS_UCCONSTANTS

# def fcn_VerReq_UR_GEN_DAT_01_RawFormat(fileName, FormatValidity, verbosity: OBJ_CORE_ENUM.Verbosity):
#     ErrCode = True
#     ErrMessage = "Raw image format is valid."
#     if FormatValidity == True:
#         return ErrCode, ErrMessage
#     else:
#         ErrMessage = "Raw image format is invalid: expected *.dat file."
#         return FormatValidity, ErrMessage
    
# def fcn_VerReq_UR_GEN_DAT_01_RawCorrespondent(filename, match, verbosity: OBJ_CORE_ENUM.Verbosity.LOW):
#     ErrCode = True
#     ErrMessage = "Raw image match valid."
#     if match > 0:
#         return ErrCode, ErrMessage
#     else:
#         ErrCode = False
#         ErrMessage = "Raw image does not have a correspondent XML file"
#         return ErrCode, ErrMessage
  
# def fcn_VerReq_NumberOfSARProducts(extractedStrings):
#     unique_products = set(extractedStrings)
#     return len(unique_products)

# # Verify if Use case reference is integrated in the file name
# def fcn_VerReq_UR_GEN_DAT_01(fileName, ReferenceCase, verbosity: OBJ_CORE_ENUM.Verbosity):
#     ErrCode = True
#     ErrMessage = "Applicable use-case is valid."
#     # Correct the call to fcn_AssertString by passing the verbosity directly
#     if OBJ_CORE_ASSERT.fcn_AssertString(fileName, OBJ_MIS_UCCONSTANTS.strFloodDetection, verbosity, 11, 0, 2):
#         return ErrCode, ErrMessage, OBJ_MIS_UCCONSTANTS.strFloodDetection
#     if OBJ_CORE_ASSERT.fcn_AssertString(fileName, OBJ_MIS_UCCONSTANTS.strRadioFrequencyInterference, verbosity, 11, 0, 3):
#         return ErrCode, ErrMessage, OBJ_MIS_UCCONSTANTS.strRadioFrequencyInterference
#     if OBJ_CORE_ASSERT.fcn_AssertString(fileName, OBJ_MIS_UCCONSTANTS.strVesselDetection, verbosity, 11, 0, 2):
#         return ErrCode, ErrMessage, OBJ_MIS_UCCONSTANTS.strVesselDetection
#     ErrCode = False
#     ErrMessage = "Applicable use-case is invalid."
#     if ReferenceCase == 'Flood':
#         Case = 'FD'
#     if ReferenceCase == 'Vessel':
#         Case = 'VD'
#     if ReferenceCase == 'RFI':      
#         Case = 'RFI'
#     return ErrCode, ErrMessage, Case

# # Verify image resolution
# def fcn_VerReq_SR_DAT_FUN_03(ImgType, ReferenceCase, imgWidth, imgHeight, verbosity: OBJ_CORE_ENUM.Verbosity):
#     ErrCode = True
#     ErrMessage = "SAR image resolution is valid."
#     if ImgType == "SLC":
#         if ReferenceCase != 'RFI':
#             if OBJ_CORE_ASSERT.fcn_AssertInteger(imgWidth, OBJ_MIS_RADAR.RADAR_WIDTH, verbosity):
#                 return ErrCode, ErrMessage
#             if OBJ_CORE_ASSERT.fcn_AssertInteger(imgHeight, OBJ_MIS_RADAR.RADAR_HEIGH, verbosity):
#                 return ErrCode, ErrMessage
#             ErrCode = False
#             ErrMessage = "SAR image resolution is invalid."
#         if ReferenceCase == 'RFI':
#             if imgWidth == imgHeight and imgWidth>1399 and imgWidth<1601:
#                 return ErrCode, ErrMessage
#             ErrCode = False
#             ErrMessage = "SAR image resolution is invalid."
#     else: 
#         ErrCode = False
#         ErrMessage = "Warning: SAR image is GRD - requirement not applicable."
#     return ErrCode, ErrMessage

# # Verify if there are at least 1000 sample
# def fcn_VerReq_SR_DAT_FUN_02(numOfFiles, verbosity: OBJ_CORE_ENUM.Verbosity):
#     ErrCode = True
#     ErrMessage = "Total number of images per use-case is valid."
#     # Correct the call to fcn_AssertString by passing the verbosity directly
#     if OBJ_CORE_ASSERT.fcn_AssertInteger(numOfFiles, 1000, verbosity):
#         return ErrCode, ErrMessage
#     ErrCode = False
#     ErrMessage = "Total number of images per use-case is invalid."
#     return ErrCode, ErrMessage

# #Search for the string "GRD" or "SLC" in the input string.    
# def fcn_VerReq_SR_DAT_FUN_09(ImgType, input_string):
#     ErrMessage = "SAR image product is valid."
#     if ImgType == 'RFI':
#         return True, ErrMessage
#     else:    
#         ErrCode = "GRD" in input_string or "SLC" in input_string
#         if ErrCode is False:
#             ErrMessage = "SAR image product is invalid."
#             return ErrCode, ErrMessage
#         else:
#             return ErrCode, ErrMessage
    

# def fcn_LogErrorsRaw(filename, log_filename, Case, ReferenceCase,
#                         RAW_FORMAT_status, IndexFailed_RAW_FORMAT, IndexValid_RAW_FORMAT, ErrMsg_RAW_FORMAT,
#                         RAW_CORRESPONDENT_status, IndexFailed_RAW_CORRESPONDENT, IndexValid_RAW_CORRESPONDENT, ErrMsg_RAW_CORRESPONDENT,
#                         verbosity):
#     if OBJ_CORE_ASSERT.fcn_AssertString(Case, ReferenceCase, verbosity, 0, 0, 1) is True:
#         if RAW_FORMAT_status is False:
#             IndexFailed_RAW_FORMAT = IndexFailed_RAW_FORMAT + 1
#             OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"- Error in {filename}: ['{ErrMsg_RAW_FORMAT}']", verbosity)                    
#             print(f"- {OBJ_CORE_ENUM.Colors.RED.value}Error{OBJ_CORE_ENUM.Colors.RESET.value} in {filename}: ['{OBJ_CORE_ENUM.Colors.YELLOW.value}{ErrMsg_RAW_FORMAT}{OBJ_CORE_ENUM.Colors.RESET.value}']")
#         else:
#             IndexValid_RAW_FORMAT = IndexValid_RAW_FORMAT + 1
#         if RAW_CORRESPONDENT_status is False:
#             IndexFailed_RAW_CORRESPONDENT = IndexFailed_RAW_CORRESPONDENT + 1
#             OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"- Error in {filename}: ['{ErrMsg_RAW_CORRESPONDENT}']", verbosity)                    
#             print(f"- {OBJ_CORE_ENUM.Colors.RED.value}Error{OBJ_CORE_ENUM.Colors.RESET.value} in {filename}: ['{OBJ_CORE_ENUM.Colors.YELLOW.value}{ErrMsg_RAW_CORRESPONDENT}{OBJ_CORE_ENUM.Colors.RESET.value}']")
#         else:
#             IndexValid_RAW_CORRESPONDENT = IndexValid_RAW_CORRESPONDENT + 1   
#     else:
#         OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"- Error in {filename}: ['Invalid use-case identified in folder.']", verbosity)                    
#         print(f"- {OBJ_CORE_ENUM.Colors.RED.value}Error{OBJ_CORE_ENUM.Colors.RESET.value}: ['{OBJ_CORE_ENUM.Colors.YELLOW.value}Invalid use-case identified in folder{OBJ_CORE_ENUM.Colors.RESET.value}']")
#     return IndexValid_RAW_FORMAT, IndexFailed_RAW_FORMAT,IndexValid_RAW_CORRESPONDENT, IndexFailed_RAW_CORRESPONDENT
            

# def fcn_LogErrors(TypeOfImages,
#                   filename, log_filename, Case, ReferenceCase,
#                   UR_GEN_DAT_01_status, ErrMsg_UR_GEN_DAT_01, IndexValid_UR_GEN_DAT_01, IndexFailed_UR_GEN_DAT_01,
#                   SR_DAT_FUN_03_status, ErrMsg_SR_DAT_FUN_03, IndexValid_SR_DAT_FUN_03, IndexFailed_SR_DAT_FUN_03, IndexSLC,
#                   SR_DAT_FUN_09_status, ErrMsg_SR_DAT_FUN_09, IndexValid_SR_DAT_FUN_09, IndexFailed_SR_DAT_FUN_09,
#                   verbosity):
#     if OBJ_CORE_ASSERT.fcn_AssertString(Case, ReferenceCase, verbosity, 0, 0, 1) is True:
#         if UR_GEN_DAT_01_status is False:
#             IndexFailed_UR_GEN_DAT_01 = IndexFailed_UR_GEN_DAT_01 + 1
#             OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"- Error in {filename}: ['{ErrMsg_UR_GEN_DAT_01}']", verbosity)                    
#             print(f"- {OBJ_CORE_ENUM.Colors.RED.value}Error{OBJ_CORE_ENUM.Colors.RESET.value} in {filename}: ['{OBJ_CORE_ENUM.Colors.YELLOW.value}{ErrMsg_UR_GEN_DAT_01}{OBJ_CORE_ENUM.Colors.RESET.value}']")
#         else:
#             IndexValid_UR_GEN_DAT_01 = IndexValid_UR_GEN_DAT_01 + 1
             
#         if TypeOfImages == 'L1':     
#             if SR_DAT_FUN_03_status is False:
#                 if ErrMsg_SR_DAT_FUN_03 != "Warning: SAR image is GRD - requirement not applicable.":
#                     IndexFailed_SR_DAT_FUN_03 = IndexFailed_SR_DAT_FUN_03 + 1 
#                     OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"- Error in {filename}: ['{ErrMsg_SR_DAT_FUN_03}']", verbosity)
#                     print(f"- {OBJ_CORE_ENUM.Colors.RED.value}Error{OBJ_CORE_ENUM.Colors.RESET.value} in {filename}: ['{OBJ_CORE_ENUM.Colors.YELLOW.value}{ErrMsg_SR_DAT_FUN_03}{OBJ_CORE_ENUM.Colors.RESET.value}']")
#                 if ErrMsg_SR_DAT_FUN_03 == "Warning: SAR image is GRD - requirement not applicable.":
#                     IndexFailed_SR_DAT_FUN_03 = IndexFailed_SR_DAT_FUN_03 + 1 
#                     OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"- Warning in {filename}: ['{ErrMsg_SR_DAT_FUN_03}']", verbosity)
#                     print(f"- {OBJ_CORE_ENUM.Colors.YELLOW.value}Warning{OBJ_CORE_ENUM.Colors.RESET.value} in {filename}: ['{OBJ_CORE_ENUM.Colors.YELLOW.value}{ErrMsg_SR_DAT_FUN_03}{OBJ_CORE_ENUM.Colors.RESET.value}']")
#                     IndexSLC = IndexSLC + 1 
#             else:
#                 IndexValid_SR_DAT_FUN_03 = IndexValid_SR_DAT_FUN_03 + 1 
                
#             if SR_DAT_FUN_09_status is False:
#                 IndexFailed_SR_DAT_FUN_09 = IndexFailed_SR_DAT_FUN_09 + 1                        
#                 OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"- Error in {filename}: ['{ErrMsg_SR_DAT_FUN_09}']", verbosity)
#                 print(f"- {OBJ_CORE_ENUM.Colors.RED.value}Error{OBJ_CORE_ENUM.Colors.RESET.value} in {filename}: ['{OBJ_CORE_ENUM.Colors.YELLOW.value}{ErrMsg_SR_DAT_FUN_09}{OBJ_CORE_ENUM.Colors.RESET.value}']")
#             else:
#                 IndexValid_SR_DAT_FUN_09 = IndexValid_SR_DAT_FUN_09 + 1
#     else:
#         OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"- Error in {filename}: ['Invalid use-case identified in folder.']", verbosity)                    
#         print(f"- {OBJ_CORE_ENUM.Colors.RED.value}Error{OBJ_CORE_ENUM.Colors.RESET.value}: ['{OBJ_CORE_ENUM.Colors.YELLOW.value}Invalid use-case identified in folder{OBJ_CORE_ENUM.Colors.RESET.value}']")
#     return IndexValid_UR_GEN_DAT_01, IndexFailed_UR_GEN_DAT_01,IndexValid_SR_DAT_FUN_03, IndexFailed_SR_DAT_FUN_03,IndexValid_SR_DAT_FUN_09, IndexFailed_SR_DAT_FUN_09, IndexSLC

# def fcn_LogValidFolder(TypeOfImagess, FormatValidity, NumOfRaw, foundCorrespondent,
#                        log_filename, num_samples, extra_samples, 
#                        IndexFailed_UR_GEN_DAT_01, IndexValid_SR_DAT_FUN_03,
#                        IndexFailed_SR_DAT_FUN_03, IndexFailed_SR_DAT_FUN_09,
#                        SR_DAT_FUN_02_status, 
#                        NumberOfSARProducts,
#                        IndexSLC, countRAW, countXML, countL1,countL1_0,countMask,
#                        verbosity):
#     OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"\n", verbosity) 
      
#     if TypeOfImagess == 'L1' or TypeOfImagess == 'PNG':   
#         # TC1
#         if IndexFailed_UR_GEN_DAT_01 == 0:   
#             OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-1] - [UR_GEN_DAT_01]: ✅ - ✅ Valid Files: {num_samples+extra_samples-IndexFailed_UR_GEN_DAT_01} | ❌ Invalid Files: {IndexFailed_UR_GEN_DAT_01} | ⚠️ Warnings: 0 | Total Files: {num_samples+extra_samples}", verbosity)
#         else:
#             OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-1] - [UR_GEN_DAT_01]: ❌ - ✅ Valid Files: {num_samples+extra_samples-IndexFailed_UR_GEN_DAT_01} | ❌ Invalid Files: {IndexFailed_UR_GEN_DAT_01} | ⚠️ Warnings: 0 | Total Files: {num_samples+extra_samples}", verbosity) 
#         # TC2
#         if num_samples+extra_samples > 999:
#             OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-2] - [SR_DAT_FUN_02]: ✅ - ✅ Number of Files: {num_samples+extra_samples} | Expected Minumum Number of Files: 1000", verbosity)
#         else:
#             OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-2] - [SR_DAT_FUN_02]: ❌ - ✅ Number of Files: {num_samples+extra_samples} | Expected Minumum Number of Files: 1000", verbosity)
#         # TC3
#         if TypeOfImagess == 'L1':
#             if IndexValid_SR_DAT_FUN_03 == IndexSLC:
#                 OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-3] - [SR_DAT_FUN_03]: ✅ - ✅ Valid Files: {IndexValid_SR_DAT_FUN_03} | ❌ Invalid/Not applicable Files: {IndexFailed_SR_DAT_FUN_03} | ⚠️ GRD Files: {IndexSLC} | Total Files: {IndexValid_SR_DAT_FUN_03+IndexFailed_SR_DAT_FUN_03}", verbosity)
#             else:
#                 OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-3] - [SR_DAT_FUN_03]: ❌ - ✅ Valid Files: {IndexValid_SR_DAT_FUN_03} | ❌ Invalid/Not applicable Files: {IndexFailed_SR_DAT_FUN_03} | ⚠️ GRD Files: {IndexSLC} | Total Files: {IndexValid_SR_DAT_FUN_03+IndexFailed_SR_DAT_FUN_03}", verbosity)    
#             # TC4
#             if IndexFailed_SR_DAT_FUN_09 == 0:
#                 OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-4] - [SR_DAT_FUN_09]: ✅ - ✅ Valid Files: {num_samples+extra_samples-IndexFailed_SR_DAT_FUN_09} | ❌ Invalid Files: {IndexFailed_SR_DAT_FUN_09} | ⚠️ Warnings: 0 | Total Files: {num_samples+extra_samples}", verbosity)
#             else:
#                 OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-4] - [SR_DAT_FUN_09]: ❌ - ✅ Valid Files: {num_samples+extra_samples-IndexFailed_SR_DAT_FUN_09} | ❌ Invalid Files: {IndexFailed_SR_DAT_FUN_09} | ⚠️ Warnings: 0 | Total Files: {num_samples+extra_samples}", verbosity)  
#         OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"\n", verbosity)
        
        
#         if IndexFailed_UR_GEN_DAT_01 == 0:
#             print(f"[TC-IMG-1] - [UR_GEN_DAT_01]: ✅ - ✅ Valid Files: {num_samples+extra_samples-IndexFailed_UR_GEN_DAT_01} | ❌ Invalid Files: {IndexFailed_UR_GEN_DAT_01} | ⚠️ Optional Files: 0 | Total Files: {num_samples+extra_samples}")
#         else:
#             print(f"[TC-IMG-1] - [UR_GEN_DAT_01]: ❌ - ✅Valid Files: {num_samples+extra_samples-IndexFailed_UR_GEN_DAT_01} | ❌ Invalid Files: {IndexFailed_UR_GEN_DAT_01} | ⚠️ Optional Files: 0 | Total Files: {num_samples+extra_samples}")
#         if num_samples+extra_samples > 999:
#             print(f"[TC-IMG-2] - [SR_DAT_FUN_02]: ✅ - ✅ Number of Files: {num_samples+extra_samples} | Expected Minumum Number of Files: 1000")
#         else:
#             print(f"[TC-IMG-2] - [SR_DAT_FUN_02]: ❌ - ✅ Number of Files: {num_samples+extra_samples} | Expected Minumum Number of Files: 1000")
#         if TypeOfImagess == 'L1':
#             if IndexFailed_SR_DAT_FUN_03 == IndexSLC:
#                 print(f"[TC-IMG-3] - [SR_DAT_FUN_03]: ✅ - ✅ Valid Files: {IndexValid_SR_DAT_FUN_03} | ❌ Invalid Files/Not applicable: {IndexFailed_SR_DAT_FUN_03} | ⚠️ GRD Files: {IndexSLC} | Total Files: {IndexValid_SR_DAT_FUN_03+IndexFailed_SR_DAT_FUN_03}")
#             else:
#                 print(f"[TC-IMG-3] - [SR_DAT_FUN_03]: ❌ - ✅ Valid Files: {IndexValid_SR_DAT_FUN_03} | ❌ Invalid Files/Not applicable: {IndexFailed_SR_DAT_FUN_03} | ⚠️ GRD Files: {IndexSLC} | Total Files: {IndexValid_SR_DAT_FUN_03+IndexFailed_SR_DAT_FUN_03}")
#             if IndexFailed_SR_DAT_FUN_09 == 0:
#                 print(f"[TC-IMG-4] - [SR_DAT_FUN_09]: ✅ - ✅ Valid Files: {num_samples+extra_samples-IndexFailed_SR_DAT_FUN_09} | ❌ Invalid Files: {IndexFailed_SR_DAT_FUN_09} | ⚠️ Optional Files: 0 | Total Files: {num_samples+extra_samples}")
#             else:
#                 print(f"[TC-IMG-4] - [SR_DAT_FUN_09]: ❌ - ✅  Valid Files: {num_samples+extra_samples-IndexFailed_SR_DAT_FUN_09} | ❌ Invalid Files: {IndexFailed_SR_DAT_FUN_09} | ⚠️ Optional Files: 0 | Total Files: {num_samples+extra_samples}")
        
#         if TypeOfImagess ==  'PNG':
#             # check number of samples
#             if countMask == countXML:
#                 OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-10]: ✅ - ✅ Number of Mask files: {countMask} | ✅ Number of XML Files: {countXML} | ⚠️ Warnings: 0 | Condition: {countMask} = {countXML}", verbosity)
#                 print(f"[TC-IMG-10]: ✅ - ✅ Number of Mask files: {countMask} | ✅ Number of XML Files: {countXML} | ⚠️ Warnings: 0 | Condition: {countMask} = {countXML}")
#             else:
#                 OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-10]: ❌ - ✅ Number of Mask files: {countMask} | ✅ Number of XML Files: {countXML} | ⚠️ Warnings: 0 | Condition: {countMask} = {countXML}", verbosity)
#                 print(f"[TC-IMG-10]: ❌ - ✅ Number of Mask files: {countMask} | ✅ Number of XML Files: {countXML} | ⚠️ Warnings: 0 | Condition: {countMask} = {countXML}")
        
#         #print(f"\n\n\n\n\n{IndexFailed_UR_GEN_DAT_01 == 0} {IndexValid_SR_DAT_FUN_03 == IndexSLC} {SR_DAT_FUN_02_status == True} {IndexFailed_SR_DAT_FUN_09 == 0}\n\n\n")
#         if TypeOfImagess == 'L1':
#             if IndexFailed_UR_GEN_DAT_01 == 0 and IndexValid_SR_DAT_FUN_03 == IndexSLC and num_samples+extra_samples > 999 and IndexFailed_SR_DAT_FUN_09 == 0:
#                 OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"✅ Validation succesful. Check {log_filename} for details.", verbosity)
#                 print(f"\n✅ Validation succesful. Check {OBJ_CORE_ENUM.Colors.GREEN.value}{log_filename}{OBJ_CORE_ENUM.Colors.RESET.value} for details.")
#             else:
#                 OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"❌ Validation failed. Check {log_filename} for details.", verbosity)
#                 print(f"\n❌ Validation failed. Check {OBJ_CORE_ENUM.Colors.YELLOW.value}{log_filename}{OBJ_CORE_ENUM.Colors.RESET.value} for details.")
#         if TypeOfImagess == 'PNG':
#             if IndexFailed_UR_GEN_DAT_01 == 0 and IndexValid_SR_DAT_FUN_03 == IndexSLC and num_samples+extra_samples > 999 and IndexFailed_SR_DAT_FUN_09 == 0 and countMask == countXML:
#                 OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"✅ Validation succesful. Check {log_filename} for details.", verbosity)
#                 print(f"\n✅ Validation succesful. Check {OBJ_CORE_ENUM.Colors.GREEN.value}{log_filename}{OBJ_CORE_ENUM.Colors.RESET.value} for details.")
#             else:
#                 OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"❌ Validation failed. Check {log_filename} for details.", verbosity)
#                 print(f"\n❌ Validation failed. Check {OBJ_CORE_ENUM.Colors.YELLOW.value}{log_filename}{OBJ_CORE_ENUM.Colors.RESET.value} for details.")

#     if TypeOfImagess == 'Raw':
#         if NumOfRaw == num_samples:
#             OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-5] - [UR_GEN_DAT_01]: ✅ - ✅ Valid Files: {NumOfRaw} | ❌ Invalid Files: {num_samples-NumOfRaw} | ⚠️ Warnings: 0 | Total Files: {num_samples}", verbosity)
#             print(f"\n[TC-IMG-5] - [UR_GEN_DAT_01]: ✅ - ✅ Valid Files: {NumOfRaw} | ❌ Invalid Files: {num_samples-NumOfRaw} | ⚠️ Warnings: 0 | Total Files: {num_samples}")
#         else:
#             OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-5] - [UR_GEN_DAT_01]: ❌ - ✅ Valid Files: {NumOfRaw} | ❌ Invalid Files: {num_samples-NumOfRaw} | ⚠️ Warnings: 0 | Total Files: {num_samples}", verbosity)
#             print(f"\n[TC-IMG-5] - [UR_GEN_DAT_01]: ❌ - ✅ Valid Files: {NumOfRaw} | ❌ Invalid Files: {num_samples-NumOfRaw} | ⚠️ Warnings: 0 | Total Files: {num_samples}")
         
#         # FormatValidity
#         if foundCorrespondent == num_samples:
#             OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-6] - [UR_GEN_DAT_01]: ✅ - ✅ Valid Files: {foundCorrespondent} | ❌ Invalid Files: {num_samples-foundCorrespondent} | ⚠️ Warnings: 0 | Total Files: {num_samples}", verbosity)
#             print(f"[TC-IMG-6] - [UR_GEN_DAT_01]: ✅ - ✅ Valid Files: {foundCorrespondent} | ❌ Invalid Files: {num_samples-foundCorrespondent} | ⚠️ Warnings: 0 | Total Files: {num_samples}")
#         else:
#             OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-6] - [UR_GEN_DAT_01]: ❌ - ✅ Valid Files: {foundCorrespondent} | ❌ Invalid Files: {num_samples-foundCorrespondent} | ⚠️ Warnings: 0 | Total Files: {num_samples}", verbosity)
#             print(f"[TC-IMG-6] - [UR_GEN_DAT_01]: ❌ - ✅ Valid Files: {foundCorrespondent} | ❌ Invalid Files: {num_samples-foundCorrespondent} | ⚠️ Warnings: 0 | Total Files: {num_samples}")
         
#         # NumberOfSARProducts
#         if NumberOfSARProducts > 9:
#             OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-7] - [UR_GEN_DAT_01]: ✅ - ✅ Number of unique SAR Products: {NumberOfSARProducts} | Expected Minumum Number of Files: 10", verbosity)
#             print(f"[TC-IMG-7] - [UR_GEN_DAT_01]: ✅ - ✅ Number of unique SAR Products: {NumberOfSARProducts} | Expected Minumum Number of Files: 10")
#         else:
#             OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-7] - [UR_GEN_DAT_01]: ❌ - - ✅ Number of unique SAR Products: {NumberOfSARProducts} | Expected Minumum Number of Files: 10", verbosity)
#             print(f"[TC-IMG-7] - [UR_GEN_DAT_01]: ❌ - ✅ Number of unique SAR Products: {NumberOfSARProducts} | Expected Minumum Number of Files: 10") 
        
#         # check number of samples
#         if countRAW == countXML * 2:
#             OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-8]: ✅ - ✅ Number of RAW files: {countRAW} | ✅ Number of XML Files: {countXML} | ⚠️ Warnings: 0 | Condition: {countRAW} = 2 x {countXML}", verbosity)
#             print(f"[TC-IMG-8]: ✅ - ✅ Number of RAW files: {countRAW} | ✅ Number of XML Files: {countXML} | ⚠️ Warnings: 0 | Condition: {countRAW} = 2 x {countXML}")
#         else:
#             OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-8]: ❌ - ✅ Number of RAW files: {countRAW} | ✅ Number of XML Files: {countXML} | ⚠️ Warnings: 0 | Condition: {countRAW} = 2 x {countXML}", verbosity)
#             print(f"[TC-IMG-8]: ❌ - ✅ Number of RAW files: {countRAW} | ✅ Number of XML Files: {countXML} | ⚠️ Warnings: 0 | Condition: {countRAW} = 2 x {countXML}")
            
#         if 4 *countXML == (countL1 + countL1_0):
#             OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-9]: ✅ - ✅ Number of XML files: {countXML} | ✅ Number of L1 Files: {(countL1 + countL1_0)} | ⚠️ Warnings: 0 | Condition: {countL1 + countL1_0} = 4 x {(countXML)}", verbosity)
#             print(f"[TC-IMG-9]: ✅ - ✅ Number of XML files: {countXML} | ✅ Number of L1 Files: {(countL1 + countL1_0)} | ⚠️ Warnings: 0 | Condition: {(countL1 + countL1_0)} = 4 x {(countXML)}")
#         else:
#             OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-9]: ❌ - ✅ Number of XML files: {(countL1 + countL1_0)} | ✅ Number of L1 Files: {countXML} | ⚠️ Warnings: 0 | Condition: {countL1 + countL1_0} = 4 x {(countXML)}", verbosity)
#             print(f"[TC-IMG-9]: ❌ - ✅ Number of XML files: {countXML} | ✅ Number of L1 Files: {(countL1 + countL1_0)} | ⚠️ Warnings: 0 | Condition: {countL1 + countL1_0} = 4 x {(countXML)}")
            
#         #countL1
        
#         # full valid
#         if foundCorrespondent == num_samples and NumOfRaw == num_samples and countRAW == countXML * 2 and 4*countXML == (countL1 + countL1_0) and NumberOfSARProducts > 9:
#             OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"✅ Validation succesful. Check {log_filename} for details.", verbosity)
#             print(f"\n✅ Validation succesful. Check {OBJ_CORE_ENUM.Colors.GREEN.value}{log_filename}{OBJ_CORE_ENUM.Colors.RESET.value} for details.")
#         else:
#             OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"❌ Validation failed. Check {log_filename} for details.", verbosity)
#             print(f"\n❌ Validation failed. Check {OBJ_CORE_ENUM.Colors.GREEN.value}{log_filename}{OBJ_CORE_ENUM.Colors.RESET.value} for details.")
         
    
#     OBJ_CORE_LOG.fcn_CloseLog(log_filename, verbosity)

# def fcn_LogInvalidFolder(TypeOfImages, log_filename, verbosity):
#     if TypeOfImages == 'L1':
#         OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"\n", verbosity)
#         OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-1] - [UR_GEN_DAT_01]: ❌ Folder is missing.", verbosity)
#         OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-2] - [SR_DAT_FUN_02]: ❌ Folder is missing.", verbosity)
#         OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-3] - [SR_DAT_FUN_03]: ❌ Folder is missing.", verbosity)
#         OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-4] - [SR_DAT_FUN_09]: ❌ Folder is missing.", verbosity)
#         OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"\n", verbosity)
#         OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"❌ Validation failed. Invalid folder.", verbosity)
#         OBJ_CORE_LOG.fcn_CloseLog(log_filename, verbosity)
#         print(f"[TC-IMG-1] - [UR_GEN_DAT_01]: ❌ Folder is missing.")
#         print(f"[TC-IMG-2] - [SR_DAT_FUN_02]: ❌ Folder is missing.")
#         print(f"[TC-IMG-3] - [SR_DAT_FUN_03]: ❌ Folder is missing.")
#         print(f"[TC-IMG-4] - [SR_DAT_FUN_09]: ❌ Folder is missing.")
#         print(f"\n❌ Validation failed. Check {OBJ_CORE_ENUM.Colors.YELLOW.value}{log_filename}{OBJ_CORE_ENUM.Colors.RESET.value} for details.")
#     if TypeOfImages == 'Raw':
#         OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"\n", verbosity)
#         OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-5] - [UR_GEN_DAT_01]: ❌ Folder is missing.", verbosity)
#         OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-6] - [UR_GEN_DAT_01]: ❌ Folder is missing.", verbosity)
#         OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-7] - [UR_GEN_DAT_01]: ❌ Folder is missing.", verbosity)
#         OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-8]: ❌ Folder is missing.", verbosity)
#         OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"[TC-IMG-9]: ❌ Folder is missing.", verbosity)
#         OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"\n", verbosity)
#         OBJ_CORE_LOG.fcn_AppendLog(log_filename, f"❌ Validation failed. Invalid folder.", verbosity)
#         OBJ_CORE_LOG.fcn_CloseLog(log_filename, verbosity)
#         print(f"[TC-IMG-5] - [UR_GEN_DAT_01]: ❌ Folder is missing.")
#         print(f"[TC-IMG-6] - [UR_GEN_DAT_01]: ❌ Folder is missing.")
#         print(f"[TC-IMG-7] - [UR_GEN_DAT_01]: ❌ Folder is missing.")
#         print(f"[TC-IMG-8]: ❌ Folder is missing.")
#         print(f"[TC-IMG-9]: ❌ Folder is missing.")
#         print(f"\n❌ Validation failed. Check {OBJ_CORE_ENUM.Colors.YELLOW.value}{log_filename}{OBJ_CORE_ENUM.Colors.RESET.value} for details.")


