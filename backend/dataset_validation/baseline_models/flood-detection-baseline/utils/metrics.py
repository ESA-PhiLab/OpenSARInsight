import torch

def dice_coef(preds, targets, smooth=1e-6):
    """
    Computes the Dice coefficient between predicted and target masks.

    Args:
        preds (Tensor): Binary predicted masks (B, 1, H, W)
        targets (Tensor): Ground truth binary masks (B, 1, H, W)
        smooth (float): Smoothing factor to avoid division by zero

    Returns:
        float: Dice coefficient (0 to 1)
    """
    preds = preds.float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def iou_score(preds, targets, smooth=1e-6):
    """
    Computes the Intersection over Union (IoU) between predicted and target masks.

    Args:
        preds (Tensor): Binary predicted masks (B, 1, H, W)
        targets (Tensor): Ground truth binary masks (B, 1, H, W)
        smooth (float): Smoothing factor to avoid division by zero

    Returns:
        float: IoU score (0 to 1)
    """
    preds = preds.float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    total = (preds + targets).sum(dim=(1, 2, 3))
    union = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()