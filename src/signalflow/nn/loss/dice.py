"""Dice loss for multi-class classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for multi-class classification with severe class imbalance.

    Computes the soft Dice coefficient between predicted probabilities
    and one-hot encoded targets, averaged over classes. Directly optimizes
    the overlap between predictions and targets per class.

    Reference:
        Milletari et al., "V-Net: Fully Convolutional Neural Networks
        for Volumetric Medical Image Segmentation", 2016.

    Args:
        smooth: Smoothing factor for numerical stability. Default: 1.0.
        reduction: Reduction mode over classes ('mean', 'sum', 'none').
            Default: 'mean'.

    Example:
        >>> loss_fn = DiceLoss(smooth=1.0)
        >>> logits = torch.randn(32, 3)
        >>> targets = torch.randint(0, 3, (32,))
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if smooth <= 0:
            raise ValueError(f"smooth must be > 0, got {smooth}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")

        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute dice loss.

        Args:
            logits: Raw predictions [batch, num_classes] (before softmax).
            targets: Target class indices [batch] (long).

        Returns:
            Loss value (scalar if reduction='mean'/'sum', [num_classes] if 'none').
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=num_classes).float()

        # Per-class intersection and union over batch dimension
        intersection = (probs * targets_onehot).sum(dim=0)
        union = probs.sum(dim=0) + targets_onehot.sum(dim=0)

        # Per-class Dice coefficient
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
