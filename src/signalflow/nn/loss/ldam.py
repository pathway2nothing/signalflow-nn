"""Label-Distribution-Aware Margin loss for long-tailed distributions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin loss for long-tailed distributions.

    Enforces larger margins for minority classes based on class frequency.
    Classes with fewer samples get larger margins, encouraging the model
    to learn more robust decision boundaries for rare classes.

    Reference:
        Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware
        Margin Loss", NeurIPS 2019.

    Args:
        cls_num_list: Number of training samples per class.
            E.g., [5000, 1000, 200] for 3 imbalanced classes.
        max_m: Maximum margin value. Default: 0.5.
        s: Logit scaling factor. Default: 30.0.
        weight: Optional per-class weight tensor for additional reweighting.
        reduction: Reduction mode ('mean', 'sum', 'none'). Default: 'mean'.

    Example:
        >>> cls_num_list = [5000, 1000, 200]
        >>> loss_fn = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30.0)
        >>> logits = torch.randn(32, 3)
        >>> targets = torch.randint(0, 3, (32,))
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        cls_num_list: list[int],
        max_m: float = 0.5,
        s: float = 30.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if len(cls_num_list) < 2:
            raise ValueError(f"cls_num_list must have at least 2 classes, got {len(cls_num_list)}")
        if any(n <= 0 for n in cls_num_list):
            raise ValueError(f"All class counts must be > 0, got {cls_num_list}")
        if max_m <= 0:
            raise ValueError(f"max_m must be > 0, got {max_m}")
        if s <= 0:
            raise ValueError(f"s must be > 0, got {s}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")

        self.s = s
        self.reduction = reduction

        # Compute per-class margins: delta_j = C / n_j^(1/4)
        m_list = torch.FloatTensor(cls_num_list).pow(-0.25)
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer("m_list", m_list)

        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute LDAM loss.

        Args:
            logits: Raw predictions [batch, num_classes] (before softmax).
            targets: Target class indices [batch] (long).

        Returns:
            Loss value (scalar if reduction='mean'/'sum', [batch] if 'none').
        """
        num_classes = logits.shape[1]

        # Per-sample margin for the target class
        batch_m = self.m_list[targets]

        # Subtract margin only from the target class logit
        targets_onehot = F.one_hot(targets, num_classes=num_classes).float()
        logits_m = logits - batch_m.unsqueeze(1) * targets_onehot

        # Scale logits
        logits_m = logits_m * self.s

        return F.cross_entropy(logits_m, targets, weight=self.weight, reduction=self.reduction)
