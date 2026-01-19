"""MLP-based classification head."""

from typing import Literal

import optuna
import torch
import torch.nn as nn

from signalflow import sf_component, SfTorchModuleMixin


@sf_component(name="head/cls/mlp")
class MLPClassifierHead(nn.Module, SfTorchModuleMixin):
    """Multi-layer perceptron classification head.
    
    Architecture: Linear -> Activation -> Dropout -> ... -> Linear(num_classes)
    
    Args:
        input_size: Size of encoder output.
        num_classes: Number of output classes (2-5 typical).
        hidden_sizes: List of hidden layer dimensions. Default: [].
        dropout: Dropout probability. Default: 0.2.
        activation: Activation function name. Default: "relu".
        
    Example:
        >>> head = MLPClassifierHead(
        ...     input_size=256,
        ...     num_classes=3,
        ...     hidden_sizes=[128, 64],
        ...     dropout=0.3,
        ... )
        >>> x = torch.randn(32, 256)
        >>> logits = head(x)  # [32, 3]
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.2,
        activation: Literal["relu", "gelu", "silu", "tanh"] = "relu",
        **kwargs,
    ):
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = []
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                self._get_activation(activation),
                nn.Dropout(dropout),
            ])
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch, input_size].
            
        Returns:
            Logits tensor [batch, num_classes].
        """
        return self.classifier(x)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation module by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(name, nn.ReLU())
    
    @classmethod
    def default_params(cls) -> dict:
        """Default parameters for quick instantiation."""
        return {
            "hidden_sizes": [128],
            "dropout": 0.2,
            "activation": "relu",
        }
    
    @classmethod
    def tune(cls, trial: optuna.Trial, model_size: Literal["small", "medium", "large"] = "small") -> dict:
        """Optuna hyperparameter search space."""
        size_config = {
            "small": {"hidden_range": (32, 128), "max_layers": 2},
            "medium": {"hidden_range": (64, 256), "max_layers": 3},
            "large": {"hidden_range": (128, 512), "max_layers": 4},
        }
        
        config = size_config[model_size]
        
        num_layers = trial.suggest_int("head_num_layers", 0, config["max_layers"])
        hidden_sizes = []
        
        for i in range(num_layers):
            size = trial.suggest_int(f"head_hidden_{i}", *config["hidden_range"])
            hidden_sizes.append(size)
        
        return {
            "hidden_sizes": hidden_sizes,
            "dropout": trial.suggest_float("head_dropout", 0.1, 0.5),
            "activation": trial.suggest_categorical("head_activation", ["relu", "gelu", "silu"]),
        }