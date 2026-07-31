from pathlib import Path

# Base paths
base_dataset_path = Path("/mnt/appide_nas/data_lake/AI4SAR/OpenSAR/data/use_cases/vessels/aresys_baseline/dvd/")

# Validate base path exists
if not base_dataset_path.exists():
    raise FileNotFoundError(f"Dataset path does not exist: {base_dataset_path}")

# Labels 
labels = base_dataset_path / "labels"

# Patches
patches = base_dataset_path / "patches"

# Patches without Detection 
patches0 = patches / "patches0_SLC_VH"

# Patches with Detection 
patches1 = patches / "patches1_SLC_VH"

# Yolo dir 
yolo_data_path = base_dataset_path / "yolo_dataset"

# Model Config
data_yaml=yolo_data_path / 'dataset.yaml'
epochs=400
batch=4
imgsz=512
patience=15
save_dir='checkpoints'