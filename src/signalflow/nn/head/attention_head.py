# signalflow.nn/heads/attention_head.py
"""Self-attention based classification head."""

from typing import Literal

import optuna
import torch
import torch.nn as nn

from signalflow import sf_component, SfTorchModuleMixin


@sf_component(name="head/cls/attention")
class AttentionClassifierHead(nn.Module, SfTorchModuleMixin):
    """Self-attention based classification head.
    
    Applies self-attention then pools to produce classification logits.
    Useful when encoder outputs sequence embeddings or to add extra expressiveness.
    
    Args:
        input_size: Size of encoder output.
        num_classes: Number of output classes.
        num_heads: Number of attention heads. Default: 4.
        hidden_dim: Hidden dimension. Default: 128.
        dropout: Dropout probability. Default: 0.2.
        
    Note:
        If input is 2D [batch, features], treats as single-token sequence.
        If input is 3D [batch, seq, features], applies attention pooling.
        
    Example:
        >>> head = AttentionClassifierHead(
        ...     input_size=256,
        ...     num_classes=3,
        ...     num_heads=4,
        ... )
        >>> x = torch.randn(32, 256)  # 2D input
        >>> logits = head(x)  # [32, 3]
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,
        num_heads: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Ensure hidden_dim is divisible by num_heads
        if hidden_dim % num_heads != 0:
            hidden_dim = (hidden_dim // num_heads) * num_heads
        
        self.input_proj = nn.Linear(input_size, hidden_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch, input_size] or [batch, seq, input_size].
            
        Returns:
            Logits tensor [batch, num_classes].
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_size]
        
        x = self.input_proj(x)  # [batch, seq, hidden_dim]
        
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        
        x = x.mean(dim=1)  # [batch, hidden_dim]
        
        return self.classifier(x)
    
    @classmethod
    def default_params(cls) -> dict:
        """Default parameters."""
        return {
            "num_heads": 4,
            "hidden_dim": 128,
            "dropout": 0.2,
        }
    
    @classmethod
    def tune(cls, trial: optuna.Trial, model_size: Literal["small", "medium", "large"] = "small") -> dict:
        """Optuna hyperparameter search space."""
        size_config = {
            "small": {"dim_range": (64, 128), "heads": [2, 4]},
            "medium": {"dim_range": (128, 256), "heads": [4, 8]},
            "large": {"dim_range": (256, 512), "heads": [4, 8, 16]},
        }
        
        config = size_config[model_size]
        
        num_heads = trial.suggest_categorical("head_num_heads", config["heads"])
        base_dim = trial.suggest_int("head_hidden_base", config["dim_range"][0] // num_heads, config["dim_range"][1] // num_heads)
        hidden_dim = base_dim * num_heads
        
        return {
            "num_heads": num_heads,
            "hidden_dim": hidden_dim,
            "dropout": trial.suggest_float("head_dropout", 0.1, 0.4),
        }