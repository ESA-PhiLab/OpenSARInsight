import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
from utils.data_loader import UNetSegmentationDataset
from utils.metrics import iou_score


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    epoch_iou = 0.0

    for images, masks in tqdm(loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)
        with torch.no_grad():
            preds = torch.sigmoid(outputs) > 0.5
            epoch_iou += iou_score(preds, masks).item() * images.size(0)

    return epoch_loss / len(loader.dataset), epoch_iou / len(loader.dataset)


def evaluate(model, loader, criterion, device, mode="Validation"):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc=mode, leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            preds = torch.sigmoid(outputs) > 0.5
            total_loss += loss.item() * images.size(0)
            total_iou += iou_score(preds, masks).item() * images.size(0)

    return total_loss / len(loader.dataset), total_iou / len(loader.dataset)


def plot_training_curves(train_losses, val_losses, train_ious, val_ious, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(train_ious, label='Train IoU', color='blue')
    plt.plot(val_ious, label='Val IoU', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Train & Validation IoU')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'iou_curve.png'))
    plt.close()


def main():
    config = load_config("config.yaml")

    train_images = config["data"]["train_images"]
    train_masks = config["data"]["train_masks"]
    val_images = config["data"]["val_images"]
    val_masks = config["data"]["val_masks"]
    test_images = config["data"]["test_images"]
    test_masks = config["data"]["test_masks"]

    image_size = tuple(config["data"]["image_size"])
    batch_size = config["data"]["batch_size"]

    num_epochs = config["training"]["num_epochs"]
    learning_rate = config["training"]["learning_rate"]
    checkpoint_dir = config["training"]["checkpoint_dir"]

    input_channels = config["model"]["input_channels"]
    output_classes = config["model"]["output_classes"]

    device = torch.device("cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = DataLoader(UNetSegmentationDataset(train_images, train_masks, image_size=image_size),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(UNetSegmentationDataset(val_images, val_masks, image_size=image_size),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(UNetSegmentationDataset(test_images, test_masks, image_size=image_size),
                             batch_size=batch_size, shuffle=False)

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=input_channels,
        classes=output_classes
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    train_losses, val_losses = [], []
    train_ious, val_ious = [], []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = evaluate(model, val_loader, criterion, device, mode="Validation")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f}")

        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"unet_epoch{epoch}.pth"))

    plot_training_curves(train_losses, val_losses, train_ious, val_ious, output_dir=checkpoint_dir)

    # Final evaluation on test set
    test_loss, test_iou = evaluate(model, test_loader, criterion, device, mode="Test")
    print(f"\nFinal Test Loss: {test_loss:.4f} | Test IoU: {test_iou:.4f}")


if __name__ == "__main__":
    main()