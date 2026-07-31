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

def compute_precision_recall(predictions, targets, threshold=0.5):
    """
    Calculate precision and recall for binary segmentation.
    
    Precision: How many predicted RFI pixels are actually RFI
    Recall: How many actual RFI pixels were correctly detected
    
    Args:
        predictions: Model outputs after sigmoid (0 to 1)
        targets: Ground truth masks (0 or 1)
        threshold: Threshold to convert predictions to binary
    
    Returns:
        precision, recall: Float values between 0 and 1
    """
    pred_binary = (predictions > threshold).float()
    target_binary = targets.float()
    
    # True positives, false positives, false negatives
    tp = (pred_binary * target_binary).sum()
    fp = (pred_binary * (1 - target_binary)).sum()
    fn = ((1 - pred_binary) * target_binary).sum()
    
    precision = tp / (tp + fp + 1e-8)  # Avoid division by zero
    recall = tp / (tp + fn + 1e-8)
    
    return precision.item(), recall.item()

def compute_f1_score(precision, recall):
    """
    Calculate F1-score from precision and recall.
    
    F1-score is the harmonic mean of precision and recall.
    It gives a single score that balances both metrics.
    
    Args:
        precision: Precision value (0 to 1)
        recall: Recall value (0 to 1)
    
    Returns:
        f1_score: F1-score value (0 to 1)
    """
    return 2 * (precision * recall) / (precision + recall + 1e-8)

def compute_specificity(predictions, targets, threshold=0.5):
    """
    Calculate specificity (true negative rate) for binary segmentation.
    
    Specificity: How many actual non-RFI pixels were correctly identified as non-RFI
    Useful for understanding how well the model avoids false alarms.
    
    Args:
        predictions: Model outputs after sigmoid (0 to 1)
        targets: Ground truth masks (0 or 1)
        threshold: Threshold to convert predictions to binary
    
    Returns:
        specificity: Float value between 0 and 1
    """
    pred_binary = (predictions > threshold).float()
    target_binary = targets.float()
    
    # True negatives and false positives
    tn = ((1 - pred_binary) * (1 - target_binary)).sum()
    fp = (pred_binary * (1 - target_binary)).sum()
    
    specificity = tn / (tn + fp + 1e-8)
    
    return specificity.item()

def compute_all_metrics(predictions, targets, threshold=0.5):
    """
    Compute all segmentation metrics in one function.
    
    This is a convenience function that calculates all metrics at once
    to avoid redundant computations.
    
    Args:
        predictions: Model outputs after sigmoid (0 to 1)
        targets: Ground truth masks (0 or 1)
        threshold: Threshold to convert predictions to binary
    
    Returns:
        dict: Dictionary containing all metrics
    """
    pred_binary = (predictions > threshold).float()
    
    # Compute IoU and Dice
    iou = iou_score(pred_binary, targets)
    dice = dice_coef(pred_binary, targets)
    
    # Compute precision, recall, specificity
    precision, recall = compute_precision_recall(predictions, targets, threshold)
    specificity = compute_specificity(predictions, targets, threshold)
    
    # Compute F1-score
    f1 = compute_f1_score(precision, recall)
    
    return {
        'iou': iou.item() if hasattr(iou, 'item') else iou,
        'dice': dice.item() if hasattr(dice, 'item') else dice,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1
    }