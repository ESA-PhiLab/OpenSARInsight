# Flood Detection Using U-Net on SAR SLC Data

This project implements a U-Net-based segmentation pipeline for detecting water bodies and flood events in SAR SLC images.

---

## Project Structure
```
├── utils/
│   ├── data_loader.py                      # Data loader script
│   ├── augmentations.py                    # SAR-safe data augmentations
│   ├── metrics.py                          # Dice & IoU metrics
├── checkpoints/             # Saved model weights
├── outputs/                 # Saved output binary masks after evaluation
├── config.yaml              # Config file for training setup
├── evaluation.py            # Evaluation code that loads the trained model (unet_epoch40.pth) and generate binary mask for an unseen data.
├── train.py                 # Main training script
├── requirements.txt
```

---

## Dataset

- Train: 876 patches with their corresponding binary masks (H0 samples:416, H1 samples:460).
- Val: 188 patches with their corresponding binary masks  (H0 samples:90, H1 samples:98).
- Test: 190 patches with their corresponding binary masks  (H0 samples:90, H1 samples:100).


## Data Format

- Images: grayscale SLC SAR images (`.tiff`) of shape `(1, H, W)`
- Masks: binary masks of same name with `_mask.tiff` suffix (values `0` or `1`)

---

## Configuration (`config.yaml`)

---

## How to Run

### 1. Train the Model
```bash
python train.py
```

---

## Output

- `checkpoints/unet_epoch*.pth`: Model weights
- `checkpoints/loss_curve.png`, `iou_curve.png`: Training curves

---

## Metrics

- **Loss**: BCEWithLogitsLoss with optional `pos_weight`
- **Validation Metric**: Intersection over Union (IoU)
- **Training Metric**: IoU (logged every epoch)

---

## Notes

- This setup is designed for binary segmentation (flood vs. non-flood)
- If your dataset is imbalanced, the `pos_weight` will help the model give more weight to flooded pixels


