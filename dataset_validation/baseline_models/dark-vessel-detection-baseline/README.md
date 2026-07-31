# Dark Vessel Detection Baseline

A YOLO-based baseline model for detecting dark vessels in SAR using the OpenSAR dataset.


## Project Structure

```
dark_vessel_detection_baseline/
├── README.md
├── constants.py                 # Project constants and paths
├── data/
│   ├── patches0/               # Positive samples (with vessels)
│   └── patches1/               # Negative samples (without vessels)
├── scripts/
│   ├── __init__.py
    ├── helper_functions.py
│   ├── data_to_yolo_format.py # Convert data to YOLO format
│   ├── yolo_preprocessing.py  # YOLO preprocessing utilities
│   └── train_yolo.py          # YOLO training script
├── notebooks/
│   ├── data_exploration.ipynb # Data analysis and exploration
│   ├── model_training.ipynb   # Model training workflow
│   └── evaluation.ipynb       # Model evaluation and visualization
├── checkpoints/
│   └── train/
│       └── weights/
│           ├── best.pt        # Best model weights
│           ├── best.onnx      # ONNX model export
│           └── last.pt        # Latest checkpoint
├── yolo_dataset/              # in a diff dir
│   ├── dataset.yaml           # YOLO dataset configuration
│   ├── images/                # Training/validation/test images
│   └── labels/                # YOLO format annotations
└── train.py                   # Run training pipeline
```

## Installation

1. Clone the repository:
```bash
git clone https://stash.deimos-space.com/projects/OPENSARI/repos/backend
cd dark_vessel_detection_baseline
```
## Quick Start

### 1. Data Preparation

Convert your SAR patches to YOLO format:

```python
from scripts.data_to_yolo_format import main
main()
```

Or run the script directly:
```bash
python scripts/data_to_yolo_format.py
```

### 2. Model Training

Train the YOLO model:
```bash
python scripts/train_yolo.py
```

### 3. Model Evaluation

Evaluate the trained model:

```python
from ultralytics import YOLO

# Load model
model = YOLO('checkpoints/train/weights/best.pt')

# Run validation
results = model.val(data='yolo_dataset/dataset.yaml')
print(f"mAP50: {results.box.map50}")
print(f"mAP50-95: {results.box.map}")
```

### 4. Visualization

Visualize predictions on test images:

```python
# See notebooks/evaluation.ipynb for detailed visualization code
from scripts.predictions import visualize_predictions

visualize_predictions(
    model=model,
    test_files=test_files,
    yolo_data_dir='yolo_dataset',
    conf_threshold=0.06
)
```

## Key Files

### Configuration
- **`constants.py`**: Contains all project paths and configuration parameters
- **`yolo_dataset/dataset.yaml`**: YOLO dataset configuration file

### Data Processing
- **`scripts/yolo_preprocessing.py`**: Core preprocessing utilities for YOLO format conversion
- **`scripts/data_to_yolo_format.py`**: Main script to convert SAR patches to YOLO format

### Training
- **`scripts/train_yolo.py`**: YOLO model training script with configurable parameters

### Notebooks
- **`notebooks/DVD_Exploratory_Data_Analysis.ipynb.ipynb`**: Analyze dataset distribution and characteristics
- **`DVD_Inference_Sample.ipynb`**: Model evaluation and result visualization

## Dataset Format

The project expects SAR image patches in the following structure:

```
data/
├── patches0/           # Positive samples (contain vessels)
│   ├── image1.jpg
│   ├── image1.txt     # YOLO format annotations
│   └── ...
└── patches1/           # Negative samples (no vessels)
    ├── image1.jpg
    └── ...
```

### YOLO Annotation Format
Each `.txt` file contains bounding box annotations in YOLO format:
```
class_id x_center y_center width height
```
All coordinates are normalized to [0, 1].

## Model Performance

The baseline model achieves the following metrics on the test set:

- **mAP@0.5**: 2.6%
- **mAP@0.5:0.95**: 0.6%
- **Precision**: 16%

## Usage Examples

### Custom Training Configuration

```python
from ultralytics import YOLO

# Initialize model
model = YOLO('yolov8n.pt')

# Custom training
results = model.train(
    data='yolo_dataset/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda',
    project='checkpoints',
    name='custom_training'
)
```

### Batch Prediction

```python
# Predict on multiple images
results = model.predict(
    source='yolo_dataset/images',
    conf=0.25,
    save=True,
    project='results',
    name='predictions'
)
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Ultralytics YOLOv8
- OpenCV
- Matplotlib
- NumPy
- PyYAML

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in training configuration
2. **Dataset Not Found**: Check paths in `constants.py` and `dataset.yaml`
3. **No Detections**: Try lower confidence thresholds (0.01-0.1)
4. **Missing Images**: Verify file paths and run data preprocessing again

### Tips

- Check the notebooks for  examples
- Review the evaluation metrics in `checkpoints/`
