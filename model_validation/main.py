# Main/master script to start the execution of the test cases 


import sys
import os
from pathlib import Path
import time

# Add the backend root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.model_validation.rfi_model.RFI_lib import cls_RFI_Run as RFI

from backend.model_validation.vessel_detection.DVD_lib import cls_DVD001_TC as DVD001_TC
from backend.model_validation.gen_sof_req.GEN_SOF_lib import cls_GEN_SOF_Run as GEN_SOF




def main():

    RFI.test_main_RFI()
    time.sleep(2)

    DVD001_TC.tc_DVD001()
    time.sleep(2)

    GEN_SOF.test_main_GEN_SOF()

if __name__ == "__main__":
    main()