import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from utils.metrics import iou_score

# --- Configuration ---
model_path = "checkpoints/unet_epoch40.pth"  
image_path = "/path/to/kurosiwo_validation_sample_split/test/patches/DB_OPENSAR_FD_3_5711_SLC_VH.tiff"
mask_path = "/path/to/kurosiwo_validation_sample_split/test/masks/DB_OPENSAR_FD_3_5711_SLC_VH_mask.tiff"
output_path = "outputs/pred_vs_gt.png" 

input_channels = 1
output_classes = 1
image_size = (512, 512)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load and preprocess image ---
def load_grayscale_image(image_path, size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    image = image.astype(np.float32) / 255.0
    image = (image - 0.485) / 0.229  # Normalization for pretrained encoder
    tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    return tensor.to(device), image  # image returned for plotting

def load_mask(mask_path, size):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask at {mask_path}")
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.float32)
    return torch.tensor(mask).unsqueeze(0).unsqueeze(0).to(device), mask  # mask returned for plotting

# --- Load model ---
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=input_channels,
    classes=output_classes
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Run prediction ---
with torch.no_grad():
    input_tensor, orig_image = load_grayscale_image(image_path, image_size)
    mask_tensor, orig_mask = load_mask(mask_path, image_size)

    output = model(input_tensor)
    pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)

    
    pred_tensor = torch.tensor(pred_mask_bin).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    gt_tensor = mask_tensor.cpu()  # Already has shape (1, 1, H, W)
    iou = iou_score(pred_tensor, gt_tensor)
    print(f"IoU Score: {iou:.4f}")

# --- Save side-by-side comparison ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(orig_image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(orig_mask, cmap='gray')
plt.title("Ground Truth Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(pred_mask_bin, cmap='gray')
plt.title("Predicted Mask")
plt.axis("off")

os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.tight_layout()
plt.savefig(output_path)
print(f"Saved comparison to: {output_path}")
