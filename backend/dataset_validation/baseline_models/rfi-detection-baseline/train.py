import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from models.unet import UNet
from utils.data_loader import UNetSegmentationDataset, compute_pos_weight
from utils.metrics import iou_score, compute_precision_recall, compute_all_metrics
import time
from datetime import datetime, timedelta
import random

def load_config(config_path="config.yaml"):
    """Load training settings from YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def format_time(seconds):
    """Convert seconds to human readable format like '1:23:45'."""
    return str(timedelta(seconds=int(seconds)))

def simple_augment(images, masks):
    """
    Simple data augmentation for SAR images and masks.
    Applies the same random transformations to both image and mask.
    """
    batch_size = images.size(0)
    augmented_images = []
    augmented_masks = []
    
    for i in range(batch_size):
        img = images[i]
        mask = masks[i]
        
        # Random horizontal flip (50% chance)
        if random.random() > 0.5:
            img = torch.flip(img, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        
        # Random vertical flip (50% chance)
        if random.random() > 0.5:
            img = torch.flip(img, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        
        # Random 90-degree rotation (25% chance each: 0°, 90°, 180°, 270°)
        k = random.randint(0, 3)
        if k > 0:
            img = torch.rot90(img, k, dims=[-2, -1])
            mask = torch.rot90(mask, k, dims=[-2, -1])
        
        augmented_images.append(img)
        augmented_masks.append(mask)
    
    return torch.stack(augmented_images), torch.stack(augmented_masks)

def train_one_epoch(model, loader, criterion, optimizer, device, use_augment=True):
    """
    Train the model for one epoch with simple augmentation.
    
    Returns:
        avg_loss: Average training loss
        avg_iou: Average IoU score
        avg_precision: Average precision
        avg_recall: Average recall
        epoch_time: Time taken for this epoch
    """
    model.train()
    epoch_loss = 0.0
    epoch_iou = 0.0
    epoch_precision = 0.0
    epoch_recall = 0.0
    epoch_start_time = time.time()

    for images, masks in tqdm(loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)
        
        # Apply simple augmentation during training
        if use_augment:
            images, masks = simple_augment(images, masks)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)

        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            preds_binary = preds > 0.5
            
            # Calculate metrics using utils.metrics
            epoch_iou += iou_score(preds_binary, masks).item() * images.size(0)
            precision, recall = compute_precision_recall(preds, masks)
            epoch_precision += precision * images.size(0)
            epoch_recall += recall * images.size(0)

    epoch_time = time.time() - epoch_start_time
    dataset_size = len(loader.dataset)
    
    return (epoch_loss / dataset_size, 
            epoch_iou / dataset_size,
            epoch_precision / dataset_size,
            epoch_recall / dataset_size,
            epoch_time)

def validate(model, loader, criterion, device):
    """
    Validate the model on validation set (no augmentation).
    
    Returns:
        val_loss: Average validation loss
        val_iou: Average validation IoU
        val_precision: Average validation precision
        val_recall: Average validation recall
        val_time: Time taken for validation
    """
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    val_precision = 0.0
    val_recall = 0.0
    val_start_time = time.time()

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation", leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)

            preds = torch.sigmoid(outputs)
            preds_binary = preds > 0.5
            
            # Calculate metrics using utils.metrics
            val_iou += iou_score(preds_binary, masks).item() * images.size(0)
            precision, recall = compute_precision_recall(preds, masks)
            val_precision += precision * images.size(0)
            val_recall += recall * images.size(0)

    val_time = time.time() - val_start_time
    dataset_size = len(loader.dataset)
    
    return (val_loss / dataset_size,
            val_iou / dataset_size,
            val_precision / dataset_size,
            val_recall / dataset_size,
            val_time)

class EarlyStopping:
    """Stop training when validation loss stops improving."""
    
    def __init__(self, patience=20, min_delta=0, restore_best_weights=True):
        """
        Args:
            patience: How many epochs to wait before stopping
            min_delta: Minimum change to count as improvement
            restore_best_weights: Load best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        """Check if training should stop based on validation loss."""
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        """Save model weights when validation improves."""
        self.best_weights = model.state_dict()

def plot_training_curves(train_losses, val_losses, train_ious, val_ious, 
                        train_precisions, val_precisions, train_recalls, val_recalls,
                        output_dir="."):
    """Create and save training progress plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Create 2x2 subplot figure
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Loss curves
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot 2: IoU curves
    plt.subplot(2, 2, 2)
    plt.plot(train_ious, label='Train IoU', color='blue')
    plt.plot(val_ious, label='Val IoU', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.title('Training vs Validation IoU')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Precision curves
    plt.subplot(2, 2, 3)
    plt.plot(train_precisions, label='Train Precision', color='blue')
    plt.plot(val_precisions, label='Val Precision', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Training vs Validation Precision')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Recall curves
    plt.subplot(2, 2, 4)
    plt.plot(train_recalls, label='Train Recall', color='blue')
    plt.plot(val_recalls, label='Val Recall', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Training vs Validation Recall')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves_P1_230.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create F1-Score plot
    plt.figure(figsize=(10, 6))
    
    # Calculate F1 scores
    train_f1 = [2 * (p * r) / (p + r + 1e-8) for p, r in zip(train_precisions, train_recalls)]
    val_f1 = [2 * (p * r) / (p + r + 1e-8) for p, r in zip(val_precisions, val_recalls)]
    
    plt.plot(train_f1, label='Train F1-Score', color='blue')
    plt.plot(val_f1, label='Val F1-Score', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.title('Training vs Validation F1-Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'f1_score_curves_P1_230.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plots: training_curves_P1_230.png and f1_score_curves_P1_230.png")

def main():
    """Main training function that orchestrates the entire training process."""
    
    # Record overall training start time
    overall_start_time = time.time()
    start_datetime = datetime.now()
    
    print(f"Training started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # --- Load Configuration ---
    config = load_config("config.yaml")

    images_dir = config["data"]["images_dir"]
    images_0_dir = config["data"]["images_0_dir"]
    masks_dir = config["data"]["masks_dir"]
    masks_0_dir = config["data"]["masks_0_dir"]
    image_size = tuple(config["data"]["image_size"])
    batch_size = config["data"]["batch_size"]
    val_split = config["data"]["val_split"]

    num_epochs = config["training"]["num_epochs"]
    learning_rate = config["training"]["learning_rate"]
    checkpoint_dir = config["training"]["checkpoint_dir"]

    input_channels = config["model"]["input_channels"]
    output_classes = config["model"]["output_classes"]

    # Early stopping parameters
    patience = config["training"].get("patience", 10)
    min_delta = config["training"].get("min_delta", 0.001)

    # Augmentation setting
    use_augmentation = config["training"].get("use_augmentation", True)

    requested_device = config["device"]
    device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Data augmentation: {'Enabled' if use_augmentation else 'Disabled'}")

    # Dataset size control
    max_rfi_images = config["data"].get("max_rfi_images", None)
    max_clean_images = config["data"].get("max_clean_images", None)
    balance_classes = config["data"].get("balance_classes", True)
    
    # --- Load Dataset ---
    dataset_start_time = time.time()
    full_dataset = UNetSegmentationDataset(
        images_dir=images_dir,
        images_0_dir=images_0_dir,
        masks_dir=masks_dir,
        masks_0_dir=masks_0_dir,
        image_size=image_size,
        augment=False,
        max_rfi_images=max_rfi_images,
        max_clean_images=max_clean_images,
        balance_classes=balance_classes
    )
    dataset_load_time = time.time() - dataset_start_time
    print(f"Dataset loading time: {format_time(dataset_load_time)}")
    
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Initialize Model ---
    model_init_start_time = time.time()
    
    # Use limited dataset for pos_weight calculation - pass all directories
    pos_weight = compute_pos_weight(
        masks_dir, masks_0_dir, 
        images_dir, images_0_dir,
        max_rfi_images, max_clean_images
    ).to(device)

    model = UNet(n_channels=input_channels, n_classes=output_classes).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model_init_time = time.time() - model_init_start_time
    print(f"Model initialization time: {format_time(model_init_time)}")

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    # --- Training Loop ---
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    train_precisions = []
    val_precisions = []
    train_recalls = []
    val_recalls = []
    epoch_times = []
    
    best_val_iou = 0.0
    best_epoch = 0

    print(f"\nStarting training with early stopping (patience={patience})")
    print(f"Total epochs: {num_epochs}")
    print("-" * 60)

    training_start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Training with augmentation
        train_loss, train_iou, train_precision, train_recall, train_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_augmentation)
        
        # Validation (no augmentation)
        val_loss, val_iou, val_precision, val_recall, val_time = validate(
            model, val_loader, criterion, device)
        
        epoch_total_time = time.time() - epoch_start_time
        epoch_times.append(epoch_total_time)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)
        train_precisions.append(train_precision)
        val_precisions.append(val_precision)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)

        # Calculate F1 scores for display
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall + 1e-8)
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-8)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f}")
        print(f"Train Precision: {train_precision:.4f} | Val Precision: {val_precision:.4f}")
        print(f"Train Recall: {train_recall:.4f} | Val Recall: {val_recall:.4f}")
        print(f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
        print(f"Epoch time: {format_time(epoch_total_time)} (Train: {format_time(train_time)}, Val: {format_time(val_time)})")

        # Save best model based on validation IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch
            
            os.makedirs(checkpoint_dir, exist_ok=True)
            best_model_path = os.path.join(checkpoint_dir, "RFI_Baseline_Model_with_Augmentation_P1_230_P0_546.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'training_time': time.time() - overall_start_time,
            }, best_model_path)
            
            print(f"★ New best model saved! Val IoU: {val_iou:.4f}")

        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation IoU: {best_val_iou:.4f} at epoch {best_epoch}")
            break

    # Calculate final timing
    training_time = time.time() - training_start_time
    total_time = time.time() - overall_start_time
    end_datetime = datetime.now()

    # Final summary
    print(f"\n" + "="*60)
    print(f"Training completed!")
    print(f"Training started: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training ended: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {format_time(total_time)}")
    print(f"Best validation IoU: {best_val_iou:.4f} achieved at epoch {best_epoch}")

    # Create and save plots
    plot_training_curves(train_losses, val_losses, train_ious, val_ious,
                        train_precisions, val_precisions, train_recalls, val_recalls,
                        output_dir=checkpoint_dir)

    # Save comprehensive metrics
    metrics_path = os.path.join(checkpoint_dir, "training_metrics_P1_230_P0_546.txt")
    with open(metrics_path, "w") as f:
        f.write(f"RFI Detection Training Summary\n")
        f.write(f"="*50 + "\n")
        f.write(f"Training started: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training ended: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total training time: {format_time(total_time)}\n")
        f.write(f"Best validation IoU: {best_val_iou:.4f}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Data augmentation: {'Enabled' if use_augmentation else 'Disabled'}\n")
        f.write(f"Final metrics:\n")
        f.write(f"  Train Loss: {train_losses[-1]:.4f}\n")
        f.write(f"  Val Loss: {val_losses[-1]:.4f}\n")
        f.write(f"  Train IoU: {train_ious[-1]:.4f}\n")
        f.write(f"  Val IoU: {val_ious[-1]:.4f}\n")
        f.write(f"  Train Precision: {train_precisions[-1]:.4f}\n")
        f.write(f"  Val Precision: {val_precisions[-1]:.4f}\n")
        f.write(f"  Train Recall: {train_recalls[-1]:.4f}\n")
        f.write(f"  Val Recall: {val_recalls[-1]:.4f}\n")

    print(f"Training metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()