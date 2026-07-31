#!/bin/bash

# === Set Script Path ===
#SCRIPT_PATH_TEST="/home/alaa/Desktop/Open_SAR_Insight/OpenSARI/CV_Manager.py"
SCRIPT_PATH_TEST="/home/rili/develop/backend/dataset_validation/Dataset_Image_Validation/CV_Manager.py"
PATH_Placeholder="placeholder" #"/home/rili/Videos" # empty folder
# === Set paths for Flood Detection use case ===
# FD - L1
PATH_FLOODS_Test="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/flood/flood_kurosiwo_dataset_FR_v1/test/patches"
PATH_FLOODS_Train="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/flood/flood_kurosiwo_dataset_FR_v1/train/patches"
PATH_FLOODS_Val="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/flood/flood_kurosiwo_dataset_FR_v1/val/patches"
# FD - Labels
PATH_FLOODS_XML_Test="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/flood/flood_kurosiwo_dataset_FR_v1/test/labels/xml_labelling"
PATH_FLOODS_XML_Train="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/flood/flood_kurosiwo_dataset_FR_v1/train/labels/xml_labelling"
PATH_FLOODS_XML_Val="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/flood/flood_kurosiwo_dataset_FR_v1/val/labels/xml_labelling"
# FD - Raw
PATH_FLOODS_RAW_Test="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/flood/flood_kurosiwo_dataset_FR_v1/test/raw"
PATH_FLOODS_RAW_Train="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/flood/flood_kurosiwo_dataset_FR_v1/train/raw"
PATH_FLOODS_RAW_Val="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/flood/flood_kurosiwo_dataset_FR_v1/val/raw"

# === Set paths for Vessel Detection use case ===
# VD - L1
PATH_VESSELS_Test="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/vessels/vessel_viewsar3_dataset_FR_v1/test/patches"
PATH_VESSELS_Train="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/vessels/vessel_viewsar3_dataset_FR_v1/train/patches"
PATH_VESSELS_Val="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/vessels/vessel_viewsar3_dataset_FR_v1/val/patches"
# VD - L1_0
PATH_VESSELS_0_Test="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/vessels/vessel_viewsar3_dataset_FR_v1/test/patches"
PATH_VESSELS_0_Train="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/vessels/vessel_viewsar3_dataset_FR_v1/train/patches"
PATH_VESSELS_0_Val="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/vessels/vessel_viewsar3_dataset_FR_v1/val/patches"
# VD - Labels
PATH_VESSELS_XML_Test="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/vessels/vessel_viewsar3_dataset_FR_v1/test/labels"
PATH_VESSELS_XML_Train="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/vessels/vessel_viewsar3_dataset_FR_v1/train/labels"
PATH_VESSELS_XML_Val="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/vessels/vessel_viewsar3_dataset_FR_v1/val/labels"
# VD - Raw
PATH_VESSELS_RAW_Test="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/vessels/vessel_viewsar3_dataset_FR_v1/test/raw"
PATH_VESSELS_RAW_Train="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/vessels/vessel_viewsar3_dataset_FR_v1/train/raw"
PATH_VESSELS_RAW_Val="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/vessels/vessel_viewsar3_dataset_FR_v1/val/raw"

# === Set paths for Radio Frequency Interference use case ===
#RFI - L1
PATH_RFI_Test="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/rfi/rfi_aresys_dataset_FR_v1/test/patches"
PATH_RFI_Train="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/rfi/rfi_aresys_dataset_FR_v1/train/patches"
PATH_RFI_Val="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/rfi/rfi_aresys_dataset_FR_v1/val/patches"
#RFI - Labels
PATH_RFI_XML_Test="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/rfi/rfi_aresys_dataset_FR_v1/test/labels/xml_labelling"
PATH_RFI_XML_Train="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/rfi/rfi_aresys_dataset_FR_v1/train/labels/xml_labelling"
PATH_RFI_XML_Val="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/rfi/rfi_aresys_dataset_FR_v1/val/labels/xml_labelling"
#RFI - Raw
PATH_RFI_RAW_Test="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/rfi/rfi_aresys_dataset_FR_v1/test/raw"
PATH_RFI_RAW_Train="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/rfi/rfi_aresys_dataset_FR_v1/train/raw"
PATH_RFI_RAW_Val="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/rfi/rfi_aresys_dataset_FR_v1/val/raw"
#RFI - Mask
PATH_RFI_MASK_Test="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/rfi/rfi_aresys_dataset_FR_v1/test/labels/binary_masks"
PATH_RFI_MASK_Train="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/rfi/rfi_aresys_dataset_FR_v1/train/labels/binary_masks"
PATH_RFI_MASK_Val="/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/rfi/rfi_aresys_dataset_FR_v1/val/labels/binary_masks"


# === Header ===
echo "================ INDRA DEIMOS - OpenSAR Insight ================"

# === Run for L1 type, Use Cases 1 to 3 ===
echo "Running L1 validations..."
# === === FD -> ./test & ./train & ./val
python3 "$SCRIPT_PATH_TEST" L1 1 1 0 "$PATH_FLOODS_Test" "$PATH_Placeholder" "$PATH_FLOODS_RAW_Test" "$PATH_FLOODS_XML_Test" "$PATH_Placeholder"
python3 "$SCRIPT_PATH_TEST" L1 1 1 0 "$PATH_FLOODS_Train" "$PATH_FLOODS_0_Train" "$PATH_FLOODS_RAW_Train" "$PATH_FLOODS_XML_Train" "$PATH_Placeholder"
python3 "$SCRIPT_PATH_TEST" L1 1 1 0 "$PATH_FLOODS_Val" "$PATH_FLOODS_0_Val" "$PATH_FLOODS_RAW_Val" "$PATH_FLOODS_XML_Val" "$PATH_Placeholder"
# === === VD -> ./test & ./train & ./val
python3 "$SCRIPT_PATH_TEST" L1 2 1 0 "$PATH_VESSELS_Test" "$PATH_VESSELS_0_Test" "$PATH_VESSELS_RAW_Test" "$PATH_VESSELS_XML_Test" "$PATH_Placeholder"
python3 "$SCRIPT_PATH_TEST" L1 2 1 0 "$PATH_VESSELS_0_Test" "$PATH_VESSELS_Test" "$PATH_VESSELS_RAW_Test" "$PATH_VESSELS_XML_Test" "$PATH_Placeholder"
python3 "$SCRIPT_PATH_TEST" L1 2 1 0 "$PATH_VESSELS_Train" "$PATH_VESSELS_0_Train" "$PATH_VESSELS_RAW_Train" "$PATH_VESSELS_XML_Train" "$PATH_Placeholder"
python3 "$SCRIPT_PATH_TEST" L1 2 1 0 "$PATH_VESSELS_0_Train" "$PATH_VESSELS_Train" "$PATH_VESSELS_RAW_Train" "$PATH_VESSELS_XML_Train" "$PATH_Placeholder"
python3 "$SCRIPT_PATH_TEST" L1 2 1 0 "$PATH_VESSELS_Val" "$PATH_VESSELS_0_Val" "$PATH_VESSELS_RAW_Val" "$PATH_VESSELS_XML_Val" "$PATH_Placeholder"
python3 "$SCRIPT_PATH_TEST" L1 2 1 0 "$PATH_VESSELS_0_Val" "$PATH_VESSELS_Val" "$PATH_VESSELS_RAW_Val" "$PATH_VESSELS_XML_Val" "$PATH_Placeholder"
# === === RFI -> ./test & ./train & ./val
python3 "$SCRIPT_PATH_TEST" L1 3 1 0 "$PATH_RFI_Test" "$PATH_Placeholder" "$PATH_RFI_RAW_Test" "$PATH_RFI_XML_Test" "$PATH_RFI_MASK_Test"
python3 "$SCRIPT_PATH_TEST" L1 3 1 0 "$PATH_RFI_Train" "$PATH_Placeholder" "$PATH_RFI_RAW_Train" "$PATH_RFI_XML_Train" "$PATH_RFI_MASK_Train"
python3 "$SCRIPT_PATH_TEST" L1 3 1 0 "$PATH_RFI_Val" "$PATH_Placeholder" "$PATH_RFI_RAW_Val" "$PATH_RFI_XML_Val" "$PATH_RFI_MASK_Val"

# === Run for Raw type, Use Cases 1 to 3 ===
echo "Running Raw validations..."
# === === FD -> ./test & ./train & ./val
python3 "$SCRIPT_PATH_TEST" Raw 1 1 0 "$PATH_FLOODS_Test" "$PATH_Placeholder" "$PATH_FLOODS_RAW_Test" "$PATH_FLOODS_XML_Test" "$PATH_Placeholder"
python3 "$SCRIPT_PATH_TEST" Raw 1 1 0 "$PATH_FLOODS_Train" "$PATH_FLOODS_0_Train" "$PATH_FLOODS_RAW_Train" "$PATH_FLOODS_XML_Train" "$PATH_Placeholder"
python3 "$SCRIPT_PATH_TEST" Raw 1 1 0 "$PATH_FLOODS_Val" "$PATH_FLOODS_0_Val" "$PATH_FLOODS_RAW_Val" "$PATH_FLOODS_XML_Val" "$PATH_Placeholder"
# === === VD -> ./test & ./train & ./val
python3 "$SCRIPT_PATH_TEST" Raw 2 1 0 "$PATH_VESSELS_Test" "$PATH_Placeholder" "$PATH_VESSELS_RAW_Test" "$PATH_VESSELS_XML_Test" "$PATH_Placeholder"
python3 "$SCRIPT_PATH_TEST" Raw 2 1 0 "$PATH_VESSELS_Train" "$PATH_Placeholder" "$PATH_VESSELS_RAW_Train" "$PATH_VESSELS_XML_Train" "$PATH_Placeholder"
python3 "$SCRIPT_PATH_TEST" Raw 2 1 0 "$PATH_VESSELS_Val" "$PATH_Placeholder" "$PATH_VESSELS_RAW_Val" "$PATH_VESSELS_XML_Val" "$PATH_Placeholder"
# === === RFI -> ./test & ./train & ./val
python3 "$SCRIPT_PATH_TEST" Raw 3 1 0 "$PATH_RFI_Test" "$PATH_Placeholder" "$PATH_RFI_RAW_Test" "$PATH_RFI_XML_Test" "$PATH_RFI_MASK_Test"
python3 "$SCRIPT_PATH_TEST" Raw 3 1 0 "$PATH_RFI_Train" "$PATH_Placeholder" "$PATH_RFI_RAW_Train" "$PATH_RFI_XML_Train" "$PATH_RFI_MASK_Train"
python3 "$SCRIPT_PATH_TEST" Raw 3 1 0 "$PATH_RFI_Val" "$PATH_Placeholder" "$PATH_RFI_RAW_Val" "$PATH_RFI_XML_Val" "$PATH_RFI_MASK_Val"

# === Run for Mask PNG validation (Only for Use Case 3) ===
echo "Running Mask validations..."
# === === RFI -> ./binary_mask
python3 "$SCRIPT_PATH_TEST" PNG 3 1 0 "$PATH_RFI_Test" "$PATH_Placeholder" "$PATH_RFI_RAW_Test" "$PATH_RFI_XML_Test" "$PATH_RFI_MASK_Test"
python3 "$SCRIPT_PATH_TEST" PNG 3 1 0 "$PATH_RFI_Train" "$PATH_Placeholder" "$PATH_RFI_RAW_Train" "$PATH_RFI_XML_Train" "$PATH_RFI_MASK_Train"
python3 "$SCRIPT_PATH_TEST" PNG 3 1 0 "$PATH_RFI_Val" "$PATH_Placeholder" "$PATH_RFI_RAW_Val" "$PATH_RFI_XML_Val" "$PATH_RFI_MASK_Val"

# === Footer ===
echo "================ Validation complete ================"
