"""
Classification and regression heads for temporal models.

Supports:
- Hard classification (2-5 classes)
- Soft labeling (regression/distribution prediction)
- Various architectures (MLP, Attention, Residual, Ordinal)

All heads follow the interface:
    __init__(input_size: int, num_classes: int, **kwargs)
    forward(x: Tensor[batch, input_size]) -> Tensor[batch, num_classes]
"""

from signalflow.nn.head.mlp_head import MLPClassifierHead
from signalflow.nn.head.linear_head import LinearClassifierHead
from signalflow.nn.head.residual_head import ResidualClassifierHead
from signalflow.nn.head.attention_head import AttentionClassifierHead
from signalflow.nn.head.distribution_head import DistributionHead
from signalflow.nn.head.ordinal_head import OrdinalRegressionHead
from signalflow.nn.head.confidence_head import ClassificationWithConfidenceHead

__all__ = [
    "MLPClassifierHead",
    "LinearClassifierHead",
    "ResidualClassifierHead",
    "AttentionClassifierHead",
    "DistributionHead",
    "OrdinalRegressionHead",
    "ClassificationWithConfidenceHead",
]