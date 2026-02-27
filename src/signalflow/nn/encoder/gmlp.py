"""gMLP encoder for temporal features.

Based on: Liu et al. "Pay Attention to MLPs" (Google, 2021).
arXiv:2105.08050. Uses Spatial Gating Unit (SGU) instead of
self-attention for token mixing.
"""

import torch
import torch.nn as nn
from signalflow.core import feature

from signalflow import SfTorchModuleMixin


class SpatialGatingUnit(nn.Module):
    """Spatial Gating Unit (SGU).

    Splits input channels in half, applies a linear projection along
    the sequence dimension to one half, then uses it to gate the other.

    Args:
        d_model: Input feature dimension (will be split in half).
        seq_len: Sequence length for spatial projection weights.
    """

    def __init__(self, d_model: int, seq_len: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model // 2)
        self.spatial_proj = nn.Linear(seq_len, seq_len)
        # Initialize gate bias to 1 for stable training start
        nn.init.ones_(self.spatial_proj.bias)
        nn.init.normal_(self.spatial_proj.weight, std=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model)

        Returns:
            Output of shape (batch, seq_len, d_model // 2)
        """
        # Split along feature dim
        u, v = x.chunk(2, dim=-1)  # each [B, seq, d_model//2]

        # Spatial gating: project along sequence dimension
        v = self.norm(v)
        v = v.transpose(1, 2)  # [B, d_model//2, seq]
        v = self.spatial_proj(v)  # [B, d_model//2, seq]
        v = v.transpose(1, 2)  # [B, seq, d_model//2]

        return u * v


class gMLPBlock(nn.Module):
    """Single gMLP block: channel projection + spatial gating unit.

    Args:
        d_model: Input/output feature dimension.
        d_ffn: Hidden dimension in channel projection.
        seq_len: Sequence length for SGU.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj_in = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        self.channel_proj_out = nn.Sequential(
            nn.Linear(d_ffn // 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm residual connection.

        Args:
            x: Input of shape (batch, seq_len, d_model)

        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)
        x = self.channel_proj_in(x)  # [B, seq, d_ffn]
        x = self.sgu(x)  # [B, seq, d_ffn//2]
        x = self.channel_proj_out(x)  # [B, seq, d_model]
        return x + residual


@feature("encoder/gmlp")
class gMLPEncoder(nn.Module, SfTorchModuleMixin):
    """gMLP encoder for sequence processing.

    Replaces self-attention with Spatial Gating Units (SGU) for
    efficient token mixing. Achieves competitive performance with
    Transformers while being more parameter-efficient.

    Args:
        input_size: Number of input features per timestep.
        seq_len: Input sequence length (needed for SGU weights).
        d_model: Embedding dimension.
        d_ffn: Hidden FFN dimension (should be even for SGU split).
        num_layers: Number of gMLP blocks.
        dropout: Dropout rate.

    Example:
        >>> encoder = gMLPEncoder(input_size=10, seq_len=60, d_model=128)
        >>> x = torch.randn(32, 60, 10)  # (batch, seq_len, features)
        >>> out = encoder(x)  # (32, 128)
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        d_model: int = 128,
        d_ffn: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.input_projection = nn.Linear(input_size, d_model)

        self.blocks = nn.Sequential(*[gMLPBlock(d_model, d_ffn, seq_len, dropout) for _ in range(num_layers)])

        self.norm = nn.LayerNorm(d_model)
        self._output_size = d_model

    @property
    def output_size(self) -> int:
        """Output embedding size."""
        return self._output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Output tensor of shape (batch, output_size)
        """
        x = self.input_projection(x)  # [B, seq, d_model]
        x = self.blocks(x)  # [B, seq, d_model]
        x = self.norm(x)
        x = x.mean(dim=1)  # [B, d_model]
        return x

    @classmethod
    def default_params(cls) -> dict:
        """Default parameters for gMLP encoder."""
        return {
            "input_size": 10,
            "seq_len": 60,
            "d_model": 128,
            "d_ffn": 256,
            "num_layers": 4,
            "dropout": 0.1,
        }

    @classmethod
    def search_space(cls, model_size: str = "small") -> dict:
        """Hyperparameter search space.

        Args:
            model_size: Size variant ('small', 'medium', 'large').

        Returns:
            Dictionary of hyperparameters (fixed values or spec dicts).
        """
        size_config = {
            "small": {"d_model": [64, 128], "d_ffn": [128, 256], "layers": (2, 4)},
            "medium": {"d_model": [128, 256], "d_ffn": [256, 512], "layers": (4, 6)},
            "large": {"d_model": [256, 512], "d_ffn": [512, 1024], "layers": (6, 8)},
        }

        config = size_config[model_size]

        return {
            "input_size": 10,
            "seq_len": 60,
            "d_model": {"type": "categorical", "choices": config["d_model"]},
            "d_ffn": {"type": "categorical", "choices": config["d_ffn"]},
            "num_layers": {"type": "int", "low": config["layers"][0], "high": config["layers"][1]},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        }
