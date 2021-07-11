import torch
from torch import nn
import monai


class Loss(nn.Module):
    """Dice and crossentropy loss"""
    def __init__(self, include_background):
        super().__init__()
        self.dice = monai.losses.DiceLoss(
            include_background=include_background,
            to_onehot_y=True,
            softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        # target needs to have shape (B, D, H, W), Target from pipeline has shape (B, 1, D, H, W)
        dice = self.dice(y_pred, y_true)
        cross_entropy = self.cross_entropy(y_pred,
                                           torch.squeeze(y_true, dim=1).long())
        return dice + cross_entropy
