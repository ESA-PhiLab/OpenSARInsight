import os
import torch
from ultralytics import YOLO
from constants import (
    data_yaml,
    epochs,
    patience,
    save_dir,
    batch,
    imgsz,
)
from scripts import helper_functions

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(
    f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}"
)

# Only run this if the code above shows CUDA false and you have a GPU
# ! pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Check if GPU is available and display information
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(
        f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    )
    device = 0  # Use GPU
else:
    print("No GPU available, using CPU")
    device = "cpu"

# yolo nano
model = YOLO("yolov8n.pt")

# Training with GPU
try:
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,
        exist_ok=True,
        project=save_dir,
        device=device,  # Specify GPU (0) or CPU ('cpu')
    )
    print("Training completed successfully!")
except Exception as e:
    print(f"Training error: {e}")

# Evaluate on validation set
val_results = model.val()
print(f"mAP50: {val_results.box.map50}")
print(f"mAP50-95: {val_results.box.map}")

# Export model
# Save checkpoint (best.pt) to the save_dir after training
best_ckpt_path = os.path.join(save_dir, "train", "weights", "best.pt")
if os.path.exists(best_ckpt_path):
    print(f"Best checkpoint found at: {best_ckpt_path}")
else:
    print("Warning: Best checkpoint not found. Check training output directory.")

# Export model using the best checkpoint
model = YOLO(best_ckpt_path)
model.export(format="onnx")  # Export to ONNX format
print(f"Model exported to: {os.path.join(save_dir, 'train', 'weights', 'DVD_baseline_model.onnx')}")