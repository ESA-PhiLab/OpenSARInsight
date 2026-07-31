import torch
from torchvision.transforms import v2
import random

# Custom transform components
class AddSpeckleNoise:
    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, img):
        if torch.is_tensor(img):
            noise = torch.randn_like(img) * self.std
            return img + img * noise
        return img

class RandomBackscatterScale:
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, img):
        scale = torch.empty(1).uniform_(*self.scale_range).item()
        return img * scale


class AddGaussianNoise:
    """Adds additive Gaussian noise to the image."""
    def __init__(self, mean=0.0, std=0.03):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        noise = torch.randn_like(img) * self.std + self.mean
        return img + noise


class Clamp:
    def __init__(self, min=0.0, max=1.0):
        self.min = min
        self.max = max

    def __call__(self, img):
        return torch.clamp(img, self.min, self.max)

    

# Compose all SAR-specific augmentations (applied in order)
def get_full_sar_transform():
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),              # Normalize to [0, 1]
        RandomBackscatterScale(scale_range=(0.9, 1.1)),     # Simulate backscatter variation, too wide a range may distort signal intensity unrealistically
        AddSpeckleNoise(std=0.05),                          # Add speckle noise, typically between 0.05-0.15
        AddGaussianNoise(std=0.03),                         # Add gaussian noise, typically between 0.01-0.05
        Clamp(0.0, 1.0),                                    # Clamp the data
    ])


