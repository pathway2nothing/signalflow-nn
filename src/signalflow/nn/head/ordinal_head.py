# signalflow.nn/heads/ordinal_head.py
"""Ordinal regression head for ordered class predictions."""

from typing import Literal

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F

from signalflow import sf_component, SfTorchModuleMixin


@sf_component(name="head/soft/ordinal")
class OrdinalRegressionHead(nn.Module, SfTorchModuleMixin):
    """Ordinal regression head for ordered classes.
    
    Treats classes as ordered (e.g., strong_fall < fall < neutral < rise < strong_rise).
    Uses cumulative link model: P(Y > k) = sigmoid(score - threshold_k)
    
    Better than standard classification when classes have natural ordering:
    - 2 classes: fall, rise
    - 3 classes: fall, neutral, rise
    - 5 classes: strong_fall, fall, neutral, rise, strong_rise
    
    The model learns:
    1. A single score f(x) for each input
    2. K-1 thresholds that separate the K classes
    
    Args:
        input_size: Size of encoder output.
        num_classes: Number of ordered classes (2-5).
        hidden_sizes: Hidden layer dimensions. Default: [128].
        dropout: Dropout probability. Default: 0.2.
        
    Example:
        >>> # 5 ordered classes
        >>> head = OrdinalRegressionHead(input_size=256, num_classes=5)
        >>> x = torch.randn(32, 256)
        >>> probs = head(x)  # [32, 5], ordered probabilities
        >>> 
        >>> # Get predictions
        >>> preds = probs.argmax(dim=1)
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        
        if hidden_sizes is None:
            hidden_sizes = [128]
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, 1))
        self.feature_net = nn.Sequential(*layers)
        
        init_thresholds = torch.linspace(-2, 2, num_classes - 1)
        self.thresholds = nn.Parameter(init_thresholds)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class probabilities.
        
        Args:
            x: Input tensor [batch, input_size].
            
        Returns:
            Probabilities tensor [batch, num_classes].
        """
        score = self.feature_net(x)  # [batch, 1]
        
        thresholds = self._get_ordered_thresholds()  # [num_classes - 1]
        
        cum_probs = torch.sigmoid(score - thresholds)  # [batch, num_classes - 1]
        
        batch_size = x.shape[0]
        ones = torch.ones(batch_size, 1, device=x.device)
        zeros = torch.zeros(batch_size, 1, device=x.device)
        
        extended_cum = torch.cat([ones, cum_probs, zeros], dim=1)  # [batch, num_classes + 1]
        
        probs = extended_cum[:, :-1] - extended_cum[:, 1:]  # [batch, num_classes]
        
        return probs
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get logits (log of probabilities) for compatibility.
        
        Note: These are not true logits, but log(probs) for CrossEntropyLoss.
        """
        probs = self.forward(x)
        return torch.log(probs + 1e-8)
    
    def get_score(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw latent score (useful for analysis)."""
        return self.feature_net(x)
    
    def _get_ordered_thresholds(self) -> torch.Tensor:
        """Ensure thresholds are strictly increasing."""
        deltas = F.softplus(self.thresholds)
        ordered = torch.cumsum(deltas, dim=0) - deltas.sum() / 2
        return ordered
    
    @classmethod
    def default_params(cls) -> dict:
        """Default parameters."""
        return {
            "hidden_sizes": [128],
            "dropout": 0.2,
        }
    
    @classmethod
    def tune(cls, trial: optuna.Trial, model_size: Literal["small", "medium", "large"] = "small") -> dict:
        """Optuna hyperparameter search space."""
        size_config = {
            "small": {"hidden_range": (64, 128), "max_layers": 2},
            "medium": {"hidden_range": (128, 256), "max_layers": 2},
            "large": {"hidden_range": (256, 512), "max_layers": 3},
        }
        
        config = size_config[model_size]
        
        num_layers = trial.suggest_int("head_num_layers", 1, config["max_layers"])
        hidden_sizes = [
            trial.suggest_int(f"head_hidden_{i}", *config["hidden_range"])
            for i in range(num_layers)
        ]
        
        return {
            "hidden_sizes": hidden_sizes,
            "dropout": trial.suggest_float("head_dropout", 0.1, 0.4),
        }


class OrdinalCrossEntropyLoss(nn.Module):
    """Cross-entropy loss for ordinal regression.
    
    Works with OrdinalRegressionHead output (probabilities).
    
    Example:
        >>> head = OrdinalRegressionHead(input_size=256, num_classes=5)
        >>> loss_fn = OrdinalCrossEntropyLoss()
        >>> 
        >>> probs = head(x)
        >>> loss = loss_fn(probs, targets)
    """
    
    def __init__(self, reduction: str = "mean", **kwargs):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss.
        
        Args:
            probs: Predicted probabilities [batch, num_classes].
            targets: Target class indices [batch].
            
        Returns:
            Loss value.
        """
        log_probs = torch.log(probs + 1e-8)
        loss = F.nll_loss(log_probs, targets, reduction=self.reduction)
        return loss