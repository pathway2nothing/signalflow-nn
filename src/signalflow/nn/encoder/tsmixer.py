"""TSMixer encoder for temporal features.

Based on: Google "TSMixer: An All-MLP Architecture for Time Series
Forecasting" (2023). arXiv:2303.06053.
"""

import torch
import torch.nn as nn

from signalflow import SfTorchModuleMixin, sf_component


class MixerBlock(nn.Module):
    """Single TSMixer block with time-mixing and feature-mixing MLPs.

    Time-mixing operates along the temporal dimension (shared across features).
    Feature-mixing operates along the feature dimension (shared across time).
    Both use pre-norm residual connections.

    Args:
        seq_len: Sequence length (needed for time-mixing weights).
        input_size: Number of features/channels.
        expansion_factor: MLP expansion in feature-mixing.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        seq_len: int,
        input_size: int,
        expansion_factor: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Time-mixing: operates on temporal dimension
        self.time_norm = nn.LayerNorm(input_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Feature-mixing: operates on feature dimension
        self.feature_norm = nn.LayerNorm(input_size)
        self.feature_mlp = nn.Sequential(
            nn.Linear(input_size, input_size * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_size * expansion_factor, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, input_size)

        Returns:
            Output of shape (batch, seq_len, input_size)
        """
        # Time-mixing: [B, seq, feat] -> transpose -> MLP on seq dim -> transpose back
        residual = x
        x_norm = self.time_norm(x)
        x_time = x_norm.transpose(1, 2)  # [B, feat, seq]
        x_time = self.time_mlp(x_time)  # [B, feat, seq]
        x = residual + x_time.transpose(1, 2)  # [B, seq, feat]

        # Feature-mixing: MLP on feature dim
        residual = x
        x = residual + self.feature_mlp(self.feature_norm(x))

        return x


@sf_component(name="encoder/tsmixer")
class TSMixerEncoder(nn.Module, SfTorchModuleMixin):
    """TSMixer encoder for sequence processing.

    All-MLP architecture that alternates time-mixing and feature-mixing
    operations. Efficient O(L + C) complexity vs O(L*C) for dense models.

    Args:
        input_size: Number of input features (channels) per timestep.
        seq_len: Input sequence length (required for time-mixing weights).
        d_model: Output embedding dimension.
        num_layers: Number of mixer blocks.
        expansion_factor: MLP expansion factor in feature-mixing.
        dropout: Dropout rate.

    Example:
        >>> encoder = TSMixerEncoder(input_size=10, seq_len=60, d_model=128)
        >>> x = torch.randn(32, 60, 10)  # (batch, seq_len, features)
        >>> out = encoder(x)  # (32, 128)
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        d_model: int = 128,
        num_layers: int = 4,
        expansion_factor: int = 2,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size

        self.blocks = nn.Sequential(
            *[MixerBlock(seq_len, input_size, expansion_factor, dropout) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(input_size)
        self.output_projection = nn.Linear(input_size, d_model)
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
        x = self.blocks(x)  # [B, seq_len, input_size]
        x = self.norm(x)
        x = x.mean(dim=1)  # [B, input_size]
        x = self.output_projection(x)  # [B, d_model]
        return x

    @classmethod
    def default_params(cls) -> dict:
        """Default parameters for TSMixer encoder."""
        return {
            "input_size": 10,
            "seq_len": 60,
            "d_model": 128,
            "num_layers": 4,
            "expansion_factor": 2,
            "dropout": 0.2,
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
            "small": {"d_model": [64, 128], "layers": (2, 4), "expansion": [2, 3]},
            "medium": {"d_model": [128, 256], "layers": (4, 6), "expansion": [2, 3, 4]},
            "large": {"d_model": [256, 512], "layers": (6, 8), "expansion": [2, 3, 4]},
        }

        config = size_config[model_size]

        return {
            "input_size": 10,
            "seq_len": 60,
            "d_model": {"type": "categorical", "choices": config["d_model"]},
            "num_layers": {"type": "int", "low": config["layers"][0], "high": config["layers"][1]},
            "expansion_factor": {"type": "categorical", "choices": config["expansion"]},
            "dropout": {"type": "float", "low": 0.1, "high": 0.5},
        }
