"""
Loss functions for temporal classification models.

Specialized losses for common challenges in financial signal classification:
- Class imbalance (FocalLoss, DiceLoss, LDAMLoss)
- Noisy labels (SymmetricCrossEntropyLoss)

All losses follow the interface:
    __init__(**kwargs)
    forward(logits: Tensor[batch, num_classes], targets: Tensor[batch]) -> Tensor
"""

from signalflow.nn.loss.dice import DiceLoss
from signalflow.nn.loss.focal import FocalLoss
from signalflow.nn.loss.ldam import LDAMLoss
from signalflow.nn.loss.symmetric_ce import SymmetricCrossEntropyLoss

__all__ = [
    "DiceLoss",
    "FocalLoss",
    "LDAMLoss",
    "SymmetricCrossEntropyLoss",
]
