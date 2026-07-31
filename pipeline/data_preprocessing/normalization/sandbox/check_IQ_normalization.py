# This script is sued to check "z-score" normalization on all raw I/Q data:
# How to run the code: python check_IQ_normalization.py 

####################Z-score normalization for the full dataset#########################
#######################################################################################
import os
import numpy as np
import torch
from data_preprocessing.normalization.sandbox.read_l0_patch import read_l0_patch
from tqdm import tqdm
from sar_normalization import NormalizeL0IQTransform  # your transform
from dataset_generation_scripts.utils import get_config
cfg = get_config("DATA_PREPROC_PATH")

# Optional: you can save CSV or TXT
RESULTS_FILE = cfg["check_iq_normalization"]["output_path"]
DAT_FOLDER = cfg["check_iq_normalization"]["input_path"]  # ← path to the raw data

# Transform: z-score normalization
transform = NormalizeL0IQTransform()

# Get all .dat files
dat_files = sorted([
    f for f in os.listdir(DAT_FOLDER)
    if f.endswith(".dat")
])

with open(RESULTS_FILE, "w") as out_file:
    out_file.write("filename,mean_I,mean_Q,std_I,std_Q\n")

    for filename in tqdm(dat_files, desc="Processing .dat files"):
        full_path = os.path.join(DAT_FOLDER, filename)

        try:
            # Read raw complex matrix
            patch_complex = read_l0_patch(full_path)

            # Convert to (H, W, 2): real + imag
            iq = np.stack([patch_complex.real, patch_complex.imag], axis=-1).astype(np.float32)
            sample = {"iq": iq}

            # Apply z-score normalization
            sample_norm = transform(sample)
            iq_norm = sample_norm["iq"]  # shape (H, W, 2)
            iq_tensor = torch.tensor(iq_norm)

            # Compute stats
            mean = iq_tensor.mean(dim=(0, 1)).tolist()
            std = iq_tensor.std(dim=(0, 1)).tolist()

            # Save result
            out_file.write(f"{filename},{mean[0]:.6f},{mean[1]:.6f},{std[0]:.6f},{std[1]:.6f}\n")

        except Exception as e:
            print(f"[!] Error processing {filename}: {e}")
            continue

print(f"\n Finished. Results saved to: {RESULTS_FILE}")


