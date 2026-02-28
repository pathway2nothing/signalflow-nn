"""Symmetric cross-entropy loss for noisy label robustness."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricCrossEntropyLoss(nn.Module):
    """Symmetric cross-entropy loss robust to noisy labels.

    Combines standard cross-entropy with reverse cross-entropy.
    The reverse term penalizes the model for being confident about
    wrong labels, providing robustness to label noise -- common in
    auto-generated trading signals.

    Reference:
        Wang et al., "Symmetric Cross Entropy for Robust Learning
        with Noisy Labels", ICCV 2019.

    Args:
        alpha: Weight for standard cross-entropy term. Default: 1.0.
        beta: Weight for reverse cross-entropy term. Default: 1.0.
        num_classes: Number of classes (required for one-hot encoding).
        reduction: Reduction mode ('mean', 'sum', 'none'). Default: 'mean'.

    Example:
        >>> loss_fn = SymmetricCrossEntropyLoss(alpha=1.0, beta=0.5, num_classes=3)
        >>> logits = torch.randn(32, 3)
        >>> targets = torch.randint(0, 3, (32,))
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        num_classes: int = 3,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")

        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute symmetric cross-entropy loss.

        Args:
            logits: Raw predictions [batch, num_classes] (before softmax).
            targets: Target class indices [batch] (long).

        Returns:
            Loss value (scalar if reduction='mean'/'sum', [batch] if 'none').

        Raises:
            ValueError: If logits num_classes doesn't match self.num_classes.
        """
        if logits.shape[1] != self.num_classes:
            raise ValueError(f"logits has {logits.shape[1]} classes, expected {self.num_classes}")

        # Standard cross-entropy term
        ce = F.cross_entropy(logits, targets, reduction="none")

        # Reverse cross-entropy term
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        targets_clipped = torch.clamp(targets_onehot, min=1e-4, max=1.0)
        rce = -(probs * torch.log(targets_clipped)).sum(dim=1)

        loss = self.alpha * ce + self.beta * rce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
