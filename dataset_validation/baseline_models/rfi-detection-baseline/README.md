# RFI Detection and Mitigation Baseline

A U-Net based deep learning model for detecting Radio Frequency Interference (RFI) in SAR images.

## Project Structure

```
rfi_detection_and_mitigation_baseline/
├── config.yaml              # Training configuration
├── train.py                 # Main training script
├── models/
│   └── unet.py              # U-Net model
├── utils/
│   ├── data_loader.py       # Dataset loader
│   └── metrics.py           # Evaluation metrics
└── checkpoints/             # Saved models and logs
```

## Installation

```bash
pip install torch torchvision pillow pyyaml matplotlib numpy tqdm
```

## Configuration (config.yaml)

```yaml
# Data paths and limits
data:
  images_dir: "/path/to/rfi/images"       # RFI SAR SLC (.tiff)
  images_0_dir: "/path/to/clean/images"   # NO SAR SLC (.tiff)
  masks_dir: "/path/to/rfi/masks"         # RFI masks (.png)
  masks_0_dir: "/path/to/clean/masks"     # No RFI masks (.png)
  max_rfi_images: 70                      # Limit RFI samples
  max_clean_images: 140                   # Limit clean samples
  balance_classes: True                  # Auto-balance dataset
  val_split: 0.3                         # Validation split

# Training settings
training:
  num_epochs: 300
  learning_rate: 0.0001
  patience: 10                           # Early stopping
  use_augmentation: true                 # Data augmentation
  
# Hardware
device: "cuda"                           # cuda or cpu
```

### Key Options:
- **max_rfi_images/max_clean_images**: Limit dataset size (use 2:1 clean:RFI ratio)
- **balance_classes**: If true, uses equal numbers of RFI and clean images
- **use_augmentation**: Applies flips and rotations during training
- **patience**: Stops training if validation doesn't improve for N epochs

## Scripts and Notebooks

### train.py
Main training script. Loads data, trains U-Net model, saves best checkpoint.

### Notebooks 
- **RFI_inference.ipynb**: Load trained model and test on new images
- **RFI_exploratory_data_analysis.ipynb**: Visualize the patches and masks

## Training

1. **Set up data paths** in `config.yaml`
2. **Run training**:
   ```bash
   python train.py
   ```

## Outputs

Training creates:
- **checkpoints/RFI_Baseline_Model.pth**: Best model weights
- **checkpoints/training_curves.png**: Loss and metric plots  
- **checkpoints/training_metrics.txt**: Training summary

### Console Output:
```
Epoch 50/300
Train Loss: 0.1234 | Val Loss: 0.1456
Train IoU: 0.8234 | Val IoU: 0.7891
Train F1: 0.8567 | Val F1: 0.8123
★ New best model saved! Val IoU: 0.7891
```

## Data Format

Expected file naming:
- **Images**: `*_SLC_VH.tiff` (grayscale SAR images)
- **Masks**: `*_MASK.png` (binary masks: white=RFI, black=clean)

Each image must have a corresponding mask with matching filename.

## Inference

```python
import torch
from models.unet import UNet

# Load model
model = UNet(n_channels=1, n_classes=1)
checkpoint = torch.load('checkpoints/RFI_Baseline_Model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Use model.eval() and torch.no_grad() for inference
```