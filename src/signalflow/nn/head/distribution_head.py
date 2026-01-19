# signalflow.nn/heads/distribution_head.py
"""Soft labeling head that outputs probability distributions."""

from typing import Literal

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F

from signalflow import sf_component, SfTorchModuleMixin


@sf_component(name="head/soft/distribution")
class DistributionHead(nn.Module, SfTorchModuleMixin):
    """Predicts probability distribution over classes (soft labels).
    
    Output is a valid probability distribution (sums to 1).
    Use with KL-divergence or cross-entropy loss against soft targets.
    
    Soft labeling allows:
    - Expressing uncertainty: [0.4, 0.4, 0.2] instead of hard [1, 0, 0]
    - Label smoothing: [0.9, 0.05, 0.05]
    - Continuous target distributions from regression-based labelers
    
    Args:
        input_size: Size of encoder output.
        num_classes: Number of output classes (2-5).
        hidden_sizes: Hidden layer dimensions. Default: [128].
        dropout: Dropout probability. Default: 0.2.
        temperature: Softmax temperature for calibration. Default: 1.0.
            - T > 1: Softer distributions (more uncertainty)
            - T < 1: Sharper distributions (more confident)
        output_type: Output format. Default: "prob".
            - "prob": Probabilities via softmax
            - "log_prob": Log probabilities via log_softmax (for NLLLoss)
            - "logits": Raw logits (no transformation)
        
    Example:
        >>> head = DistributionHead(input_size=256, num_classes=3)
        >>> x = torch.randn(32, 256)
        >>> probs = head(x)  # [32, 3], sums to 1
        >>> 
        >>> # Soft target labels
        >>> soft_targets = torch.tensor([[0.7, 0.2, 0.1], ...])
        >>> loss = F.kl_div(probs.log(), soft_targets, reduction='batchmean')
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.2,
        temperature: float = 1.0,
        output_type: Literal["prob", "log_prob", "logits"] = "prob",
        **kwargs,
    ):
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128]
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.temperature = temperature
        self.output_type = output_type
        
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, num_classes))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch, input_size].
            
        Returns:
            Output tensor [batch, num_classes].
            Format depends on output_type parameter.
        """
        logits = self.mlp(x) / self.temperature
        
        if self.output_type == "log_prob":
            return F.log_softmax(logits, dim=-1)
        elif self.output_type == "prob":
            return F.softmax(logits, dim=-1)
        else:  
            return logits
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw logits (for compatibility with standard losses)."""
        return self.mlp(x) / self.temperature
    
    @classmethod
    def default_params(cls) -> dict:
        """Default parameters."""
        return {
            "hidden_sizes": [128],
            "dropout": 0.2,
            "temperature": 1.0,
            "output_type": "prob",
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
            "temperature": trial.suggest_float("head_temperature", 0.5, 2.0),
            "output_type": "prob",  
        }