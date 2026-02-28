"""Focal loss for multi-class classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for multi-class classification with class imbalance.

    Reduces loss for well-classified examples, focusing training on hard
    misclassified samples. Particularly useful for financial signal classification
    where the neutral class dominates.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        alpha: Per-class weight tensor or list. Shape: [num_classes].
            Typical usage: inverse class frequency. None for uniform weights.
        gamma: Focusing parameter. Higher values down-weight easy examples more.
            gamma=0 reduces to standard cross-entropy. Default: 2.0.
        label_smoothing: Label smoothing factor in [0, 1). Default: 0.0.
        reduction: Reduction mode ('mean', 'sum', 'none'). Default: 'mean'.

    Example:
        >>> loss_fn = FocalLoss(gamma=2.0)
        >>> logits = torch.randn(32, 3)
        >>> targets = torch.randint(0, 3, (32,))
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        alpha: torch.Tensor | list[float] | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if not 0 <= label_smoothing < 1:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")

        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, list):
                alpha = torch.FloatTensor(alpha)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw predictions [batch, num_classes] (before softmax).
            targets: Target class indices [batch] (long).

        Returns:
            Loss value (scalar if reduction='mean'/'sum', [batch] if 'none').
        """
        num_classes = logits.shape[1]
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()

        # One-hot encode targets
        targets_onehot = F.one_hot(targets, num_classes=num_classes).float()

        # Apply label smoothing
        if self.label_smoothing > 0:
            targets_onehot = (1 - self.label_smoothing) * targets_onehot + self.label_smoothing / num_classes

        # Focal modulation: (1 - p_t)^gamma
        # p_t is the probability assigned to the true class
        p_t = (p * targets_onehot).sum(dim=1)
        focal_weight = (1 - p_t) ** self.gamma

        # Cross-entropy per sample: -sum(target * log_p)
        ce = -(targets_onehot * log_p).sum(dim=1)

        # Apply focal weight
        loss = focal_weight * ce

        # Apply class weights
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
