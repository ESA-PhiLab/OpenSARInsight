import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

def compute_pos_weight(masks_dir, masks_0_dir, images_dir, images_0_dir, max_rfi_images=None, max_clean_images=None):
    """
    Compute positive weight for imbalanced dataset using only the limited subset.
    
    Args:
        masks_dir: Directory with RFI masks (positive examples)
        masks_0_dir: Directory with clean masks (negative examples)
        images_dir: Directory with RFI images (to verify pairs exist)
        images_0_dir: Directory with clean images (to verify pairs exist)
        max_rfi_images: Maximum number of RFI masks to use (None = use all)
        max_clean_images: Maximum number of clean masks to use (None = use all)
    
    Returns:
        torch.Tensor: Positive weight scalar
    """
    total_positive_pixels = 0
    total_negative_pixels = 0
    total_masks = 0
    
    print("Computing class weights from limited mask subset...")
    
    # Count pixels in RFI masks (with limit)
    rfi_count = 0
    if os.path.exists(masks_dir) and os.path.exists(images_dir):
        # Get all image files first (these define what's available)
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tiff')])
        rfi_limit = max_rfi_images if max_rfi_images is not None else len(image_files)
        
        print(f"  Processing RFI masks (limit: {rfi_limit})...")
        
        for img_name in image_files:
            if rfi_count >= rfi_limit:
                break
                
            # Find corresponding mask
            mask_name = img_name.replace('_SLC_VH.tiff', '_MASK.png')
            mask_path = os.path.join(masks_dir, mask_name)
            
            if os.path.exists(mask_path):
                try:
                    mask = Image.open(mask_path).convert('L')
                    mask_array = np.array(mask)
                    
                    # Count positive (white) and negative (black) pixels
                    positive_pixels = np.sum(mask_array > 127)  # White pixels (RFI)
                    negative_pixels = np.sum(mask_array <= 127)  # Black pixels (clean)
                    
                    total_positive_pixels += positive_pixels
                    total_negative_pixels += negative_pixels
                    total_masks += 1
                    rfi_count += 1
                    
                    if rfi_count % 10 == 0:
                        print(f"    Processed {rfi_count} RFI masks...")
                    
                except Exception as e:
                    print(f"Error loading RFI mask {mask_name}: {e}")
    
    # Count pixels in clean masks (with limit)
    clean_count = 0
    if os.path.exists(masks_0_dir) and os.path.exists(images_0_dir):
        # Get all image files first (these define what's available)
        image_files = sorted([f for f in os.listdir(images_0_dir) if f.endswith('.tiff')])
        clean_limit = max_clean_images if max_clean_images is not None else len(image_files)
        
        print(f"  Processing clean masks (limit: {clean_limit})...")
        
        for img_name in image_files:
            if clean_count >= clean_limit:
                break
                
            # Find corresponding mask
            mask_name = img_name.replace('_SLC_VH.tiff', '_MASK.png')
            mask_path = os.path.join(masks_0_dir, mask_name)
            
            if os.path.exists(mask_path):
                try:
                    mask = Image.open(mask_path).convert('L')
                    mask_array = np.array(mask)
                    
                    # Count positive (white) and negative (black) pixels
                    positive_pixels = np.sum(mask_array > 127)  # White pixels (RFI)
                    negative_pixels = np.sum(mask_array <= 127)  # Black pixels (clean)
                    
                    total_positive_pixels += positive_pixels
                    total_negative_pixels += negative_pixels
                    total_masks += 1
                    clean_count += 1
                    
                    if clean_count % 10 == 0:
                        print(f"    Processed {clean_count} clean masks...")
                    
                except Exception as e:
                    print(f"Error loading clean mask {mask_name}: {e}")
    
    # Calculate positive weight
    if total_positive_pixels > 0:
        pos_weight = total_negative_pixels / total_positive_pixels
    else:
        pos_weight = 1.0  # Default if no positive pixels found
    '''
    print(f"Limited mask statistics:")
    print(f"  RFI masks used: {rfi_count}")
    print(f"  Clean masks used: {clean_count}")
    print(f"  Total masks processed: {total_masks}")
    print(f"  Total positive pixels (RFI): {total_positive_pixels:,}")
    print(f"  Total negative pixels (clean): {total_negative_pixels:,}")
    print(f"  Positive weight: {pos_weight:.3f}")
    '''
    return torch.tensor(pos_weight, dtype=torch.float32)

class UNetSegmentationDataset(Dataset):
    """
    Dataset for U-Net segmentation with control over RFI/clean image counts.
    """
    
    def __init__(self, images_dir, images_0_dir, masks_dir, masks_0_dir, 
                 image_size=(1510, 1510), augment=False, 
                 max_rfi_images=None, max_clean_images=None, balance_classes=False):
        """
        Args:
            images_dir: Directory with RFI images
            images_0_dir: Directory with clean images  
            masks_dir: Directory with RFI masks
            masks_0_dir: Directory with clean masks
            image_size: Target image size (height, width)
            augment: Whether to apply data augmentation
            max_rfi_images: Maximum number of RFI images to use (None = use all)
            max_clean_images: Maximum number of clean images to use (None = use all)
            balance_classes: If True, automatically balance RFI and clean image counts
        """
        self.image_size = image_size
        self.augment = augment
        
        # Store paths for use in compute_pos_weight
        self.images_dir = images_dir
        self.images_0_dir = images_0_dir
        self.masks_dir = masks_dir
        self.masks_0_dir = masks_0_dir
        self.max_rfi_images = max_rfi_images
        self.max_clean_images = max_clean_images
        
        # Basic transforms
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(image_size)
        
        # Debug: Print the parameters received
        print(f"Dataset parameters:")
        print(f"  max_rfi_images: {max_rfi_images}")
        print(f"  max_clean_images: {max_clean_images}")
        print(f"  balance_classes: {balance_classes}")
        
        # First pass: count available images
        available_rfi = self._count_available_images(images_dir, masks_dir)
        available_clean = self._count_available_images(images_0_dir, masks_0_dir)
        
        print(f"Available images: RFI={available_rfi}, Clean={available_clean}")
        
        # Determine actual counts to use
        if balance_classes:
            # Use equal numbers of both classes
            target_count = min(available_rfi, available_clean)
            if max_rfi_images is not None:
                target_count = min(target_count, max_rfi_images)
            if max_clean_images is not None:
                target_count = min(target_count, max_clean_images)
            
            actual_rfi_count = target_count
            actual_clean_count = target_count
            print(f"Balancing classes: using {target_count} of each class")
        else:
            # Use specified limits or all available
            if max_rfi_images is not None:
                actual_rfi_count = min(available_rfi, max_rfi_images)
            else:
                actual_rfi_count = available_rfi
                
            if max_clean_images is not None:
                actual_clean_count = min(available_clean, max_clean_images)
            else:
                actual_clean_count = available_clean
            
            print(f"Using specified limits: RFI={actual_rfi_count}, Clean={actual_clean_count}")
        
        # Load the images
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        
        # Load RFI images
        self._load_image_class(images_dir, masks_dir, actual_rfi_count, 1, "RFI")
        
        # Load clean images
        self._load_image_class(images_0_dir, masks_0_dir, actual_clean_count, 0, "Clean")
        
        # Print final statistics
        rfi_count = sum(self.labels)
        clean_count = len(self.labels) - rfi_count
        total = len(self.image_paths)
        
        print(f"\nFinal dataset:")
        print(f"  RFI images: {rfi_count}")
        print(f"  Clean images: {clean_count}")
        print(f"  Total: {total} image-mask pairs")
        if total > 0:
            print(f"  RFI ratio: {rfi_count/total:.2f}")
            if rfi_count > 0 and clean_count > 0:
                ratio = clean_count / rfi_count
                print(f"  Clean:RFI ratio: {ratio:.1f}:1")
    
    def _count_available_images(self, images_dir, masks_dir):
        """Count how many valid image-mask pairs are available."""
        if not (os.path.exists(images_dir) and os.path.exists(masks_dir)):
            return 0
        
        count = 0
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.tiff')]
        
        for img_name in image_files:
            mask_name = img_name.replace('_SLC_VH.tiff', '_MASK.png')
            mask_path = os.path.join(masks_dir, mask_name)
            if os.path.exists(mask_path):
                count += 1
        
        return count
    
    def _load_image_class(self, images_dir, masks_dir, max_count, label, class_name):
        """Load images from a specific class directory."""
        if not (os.path.exists(images_dir) and os.path.exists(masks_dir)):
            print(f"  {class_name} directories not found, skipping")
            return
        
        count = 0
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tiff')])
        
        print(f"  Loading {class_name} images (limit: {max_count})...")
        
        for img_name in image_files:
            if count >= max_count:
                print(f"    ✓ Reached limit of {max_count} {class_name} images")
                break
                
            img_path = os.path.join(images_dir, img_name)
            mask_name = img_name.replace('_SLC_VH.tiff', '_MASK.png')
            mask_path = os.path.join(masks_dir, mask_name)
            
            if os.path.exists(mask_path):
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
                self.labels.append(label)
                count += 1
                
                # Debug: Show progress every 20 images
                if count % 20 == 0:
                    print(f"    Loaded {count}/{max_count} {class_name} images...")
        
        print(f"  ✓ Final: Loaded {count} {class_name} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get a single image-mask pair."""
        # Load image and mask
        image = Image.open(self.image_paths[idx]).convert('L')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        
        # Resize to target size
        image = self.resize(image)
        mask = self.resize(mask)
        
        # Convert to tensors
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)
        
        # Ensure mask is binary (0 or 1)
        mask = (mask > 0.5).float()
        
        return image, mask
    
    def get_class_distribution(self):
        """Get distribution of RFI vs clean images."""
        rfi_count = sum(self.labels)
        clean_count = len(self.labels) - rfi_count
        return {'rfi': rfi_count, 'clean': clean_count}