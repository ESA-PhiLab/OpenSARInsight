import os
import torch
import numpy as np
from data_loader import UNetSegmentationDataset, get_dataloader, compute_pos_weight


def test_image_mask_pairing(images_dir, masks_dir):
    missing_masks = []
    for f in os.listdir(images_dir):
        if f.endswith(".tiff"):
            expected_mask = f.replace(".tiff", "_mask.tiff")
            if not os.path.exists(os.path.join(masks_dir, expected_mask)):
                missing_masks.append(f)
    assert not missing_masks, f"Missing masks for: {missing_masks}"
    print(" All images have corresponding masks.")

def test_tensor_value_ranges(image_tensor, mask_tensor):
    assert image_tensor.min() >= 0.0 and image_tensor.max() <= 1.0, " Image values not in [0,1]"
    assert set(torch.unique(mask_tensor).tolist()).issubset({0.0, 1.0}), " Mask contains non-binary values"
    print(" Image and mask tensors have valid value ranges.")

def test_augmentation_effect(images_dir, masks_dir):
    loader_aug = get_dataloader(images_dir, masks_dir, batch_size=1, image_size=(512, 512), augment=True)
    loader_no_aug = get_dataloader(images_dir, masks_dir, batch_size=1, image_size=(512, 512), augment=False)

    img_aug, _ = next(iter(loader_aug))
    img_plain, _ = next(iter(loader_no_aug))

    assert not torch.equal(img_aug, img_plain), " Augmentation not applied!"
    print(" Augmentation changes image as expected.")

def test_dataset_length(images_dir, masks_dir):
    dataset = UNetSegmentationDataset(images_dir, masks_dir)
    expected_len = len([f for f in os.listdir(images_dir) if f.endswith(".tiff")])
    assert len(dataset) == expected_len, " Dataset length mismatch!"
    print(" Dataset length matches number of images.")

def test_compute_pos_weight(masks_dir):
    pos_weight = compute_pos_weight(masks_dir)
    print(f" Computed pos_weight: {pos_weight.item():.4f}")
    assert pos_weight.item() > 0, " pos_weight should be > 0"
    assert not torch.isnan(pos_weight), " pos_weight should not be NaN"
    print(" pos_weight computation test passed.")


def run_all_tests(images_dir, masks_dir):
    test_image_mask_pairing(images_dir, masks_dir)
    test_dataset_length(images_dir, masks_dir)
    test_augmentation_effect(images_dir, masks_dir)
    test_compute_pos_weight(masks_dir)

    # Check one batch
    loader = get_dataloader(images_dir, masks_dir, batch_size=2, image_size=(512, 512), augment=True)
    images, masks = next(iter(loader))
    test_tensor_value_ranges(images, masks)

if __name__ == "__main__":
    images_dir = "/app/OpenSAR/Repo/backend/flood_detection_baseline/data/train/images" #path to image directory
    masks_dir = "/app/OpenSAR/Repo/backend/flood_detection_baseline/data/train/masks" #path to mask directory
    run_all_tests(images_dir, masks_dir)
