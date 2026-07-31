import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from utils.augmentations import get_full_sar_transform


class UNetSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=None, augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size  # None means no resizing
        self.augment = augment 
        self.image_filenames = sorted([f for f in os.listdir(images_dir) if f.endswith('.tiff')])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.images_dir, image_name)
        mask_path = os.path.join(self.masks_dir, image_name.replace(".tiff", "_mask.tiff"))

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.image_size:
            image = cv2.resize(image, self.image_size)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        # Normalize image for pretrained encoder
        image = image.astype(np.float32) / 255.0
        image = (image - 0.485) / 0.229
        image = torch.tensor(image).unsqueeze(0).float()  # Shape: (1, H, W)

        # Normalize binary mask to [0, 1]
        mask = (mask > 127).astype(np.float32)
        mask_tensor = torch.tensor(mask).unsqueeze(0).float()

        return image, mask_tensor


def get_dataloader(images_dir, masks_dir, batch_size=8, image_size=None, shuffle=True, augment=False):
    dataset = UNetSegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=image_size,
        augment=augment
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def compute_pos_weight(masks_dir): # For class imbalance
    total_pos = 0
    total_neg = 0

    for filename in os.listdir(masks_dir):
        if filename.endswith("_mask.tiff"):
            mask_path = os.path.join(masks_dir, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Convert to binary
            binary = (mask > 127).astype(np.uint8)

            total_pos += np.sum(binary)
            total_neg += np.sum(1 - binary)

    pos_weight = total_neg / (total_pos + 1e-6)  # avoid division by zero
    return torch.tensor([pos_weight], dtype=torch.float32)



   
