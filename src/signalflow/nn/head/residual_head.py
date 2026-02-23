# signalflow.nn/heads/residual_head.py
import torch
import torch.nn as nn

from signalflow import SfTorchModuleMixin, sf_component


class ResidualBlock(nn.Module):
    """Single residual block: x + MLP(x)."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


@sf_component(name="head/cls/residual")
class ResidualClassifierHead(nn.Module, SfTorchModuleMixin):
    """Residual MLP classification head.

    Uses skip connections for better gradient flow.
    Architecture: Input -> Projection -> ResBlock x N -> Linear(num_classes)

    Args:
        input_size: Size of encoder output.
        num_classes: Number of output classes.
        hidden_dim: Hidden dimension (same for all blocks). Default: 128.
        num_blocks: Number of residual blocks. Default: 2.
        dropout: Dropout probability. Default: 0.2.

    Example:
        >>> head = ResidualClassifierHead(
        ...     input_size=256,
        ...     num_classes=3,
        ...     hidden_dim=128,
        ...     num_blocks=2,
        ... )
        >>> x = torch.randn(32, 256)
        >>> logits = head(x)  # [32, 3]
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        self.input_proj = nn.Linear(input_size, hidden_dim) if input_size != hidden_dim else nn.Identity()

        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])

        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, input_size].

        Returns:
            Logits tensor [batch, num_classes].
        """
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x)

    @classmethod
    def default_params(cls) -> dict:
        """Default parameters."""
        return {
            "hidden_dim": 128,
            "num_blocks": 2,
            "dropout": 0.2,
        }

    @classmethod
    def search_space(cls, model_size: str = "small") -> dict:
        """Hyperparameter search space."""
        size_config = {
            "small": {"dim_range": (64, 128), "blocks_range": (1, 2)},
            "medium": {"dim_range": (128, 256), "blocks_range": (2, 3)},
            "large": {"dim_range": (256, 512), "blocks_range": (2, 4)},
        }

        config = size_config[model_size]

        return {
            "hidden_dim": {"type": "int", "low": config["dim_range"][0], "high": config["dim_range"][1]},
            "num_blocks": {"type": "int", "low": config["blocks_range"][0], "high": config["blocks_range"][1]},
            "dropout": {"type": "float", "low": 0.1, "high": 0.4},
        }
