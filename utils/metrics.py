import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Combined BCE and Dice loss
class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss(weight=weight, reduction='mean' if size_average else 'sum')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)

        # Dice loss
        smooth = 1.0
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - ((2. * intersection + smooth) /
                         (inputs_flat.sum() + targets_flat.sum() + smooth))

        return 0.5 * bce_loss + 0.5 * dice_loss

# Evaluation metrics
def calculate_metrics(pred, target):
    # Detach tensors before converting to numpy
    pred = pred.detach().view(-1).cpu().numpy() > 0.5
    target = target.detach().view(-1).cpu().numpy() > 0.5

    # Calculate metrics
    precision = precision_score(target, pred, zero_division=1)
    recall = recall_score(target, pred, zero_division=1)
    f1 = f1_score(target, pred, zero_division=1)

    # Calculate Dice coefficient and IoU
    smooth = 1e-6
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    dice = (2. * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (union - intersection + smooth)

    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
