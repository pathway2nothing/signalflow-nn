import torch
import torch.nn as nn
from signalflow.core import sf_component, SfTorchModuleMixin
import optuna

@sf_component(name="head/cls/mlp")
class MLPClassifierHead(nn.Module, SfTorchModuleMixin):
    """MLP-based classification head"""
    
    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = []
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Build layers
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                self._get_activation(activation),
                nn.Dropout(dropout)
            ])
            current_size = hidden_size
        
        # Final layer
        layers.append(nn.Linear(current_size, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_size]
            
        Returns:
            logits: [batch, num_classes]
        """
        return self.classifier(x)
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(name, nn.ReLU())
    
    @classmethod
    def default_params(cls) -> dict:
        return {
            "hidden_sizes": [128],
            "dropout": 0.2,
            "activation": "relu",
        }
    
    @classmethod
    def tune(cls, trial: optuna.Trial, model_size: str = "small") -> dict:
        size_config = {
            "small": {"hidden_range": (32, 128), "max_layers": 2},
            "medium": {"hidden_range": (64, 256), "max_layers": 3},
            "large": {"hidden_range": (128, 512), "max_layers": 4}
        }
        
        config = size_config[model_size]
        
        num_layers = trial.suggest_int("num_hidden_layers", 0, config["max_layers"])
        hidden_sizes = []
        
        for i in range(num_layers):
            size = trial.suggest_int(
                f"hidden_size_{i}", 
                *config["hidden_range"]
            )
            hidden_sizes.append(size)
        
        return {
            "hidden_sizes": hidden_sizes,
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "activation": trial.suggest_categorical(
                "activation", 
                ["relu", "gelu", "silu"]
            ),
        }