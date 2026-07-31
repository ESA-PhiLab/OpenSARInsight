import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from augmentations import get_full_sar_transform
import os


# Paths to input image and output image
img_path = "/app/OpenSAR/Repo/backend/flood_detection_baseline/data/train/images/DB_OPENSAR_FD_1_51_SLC_VH.tiff" # path to one example SLC image
output_dir = "/app/OpenSAR/Repo/backend/flood_detection_baseline/data/train" # path to output directory
os.makedirs(output_dir, exist_ok=True)

# Load a sample grayscale SAR image (normalized 0–1)
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
image_tensor = torch.tensor(image).unsqueeze(0)

# Apply SAR augmentation
augmenter = get_full_sar_transform()
augmented_tensor = augmenter(image_tensor)

# Convert to NumPy for visualization
original_np = image_tensor.squeeze(0).numpy()
augmented_np = augmented_tensor.squeeze(0).numpy()

# Plot and save the comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(original_np, cmap='gray')
axes[0].set_title("Original SAR Image")
axes[0].axis('off')

axes[1].imshow(augmented_np, cmap='gray')
axes[1].set_title("Augmented SAR Image")
axes[1].axis('off')

plt.tight_layout()

# Save to file
save_path = os.path.join(output_dir, "augmentation.png")
plt.savefig(save_path)
print("Augmentation visualization saved to: {save_path}")
