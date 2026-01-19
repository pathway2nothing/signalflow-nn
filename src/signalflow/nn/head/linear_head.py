"""Simple linear classification head."""

from typing import Literal

import optuna
import torch
import torch.nn as nn

from signalflow import sf_component, SfTorchModuleMixin


@sf_component(name="head/cls/linear")
class LinearClassifierHead(nn.Module, SfTorchModuleMixin):
    """Simple linear classification head.
    
    Just a single linear layer: Linear(input_size, num_classes).
    Useful as baseline or when encoder is already very expressive.
    
    Args:
        input_size: Size of encoder output.
        num_classes: Number of output classes.
        bias: Whether to include bias term. Default: True.
        
    Example:
        >>> head = LinearClassifierHead(input_size=256, num_classes=3)
        >>> x = torch.randn(32, 256)
        >>> logits = head(x)  # [32, 3]
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        self.linear = nn.Linear(input_size, num_classes, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch, input_size].
            
        Returns:
            Logits tensor [batch, num_classes].
        """
        return self.linear(x)
    
    @classmethod
    def default_params(cls) -> dict:
        """Default parameters."""
        return {
            "bias": True,
        }
    
    @classmethod
    def tune(cls, trial: optuna.Trial, model_size: Literal["small", "medium", "large"] = "small") -> dict:
        """Optuna hyperparameter search space (minimal for linear)."""
        return {
            "bias": trial.suggest_categorical("head_bias", [True, False]),
        }