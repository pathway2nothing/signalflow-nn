"""Transformer encoder for temporal features."""

import math

import torch
import torch.nn as nn

from signalflow import SfTorchModuleMixin, sf_component


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


@sf_component(name="encoder/transformer")
class TransformerEncoder(nn.Module, SfTorchModuleMixin):
    """Transformer encoder for sequence processing.

    Standard Transformer encoder with sinusoidal positional encoding
    and CLS token or mean pooling for fixed-size output.

    Args:
        input_size: Number of input features per timestep.
        d_model: Embedding dimension.
        nhead: Number of attention heads.
        num_layers: Number of TransformerEncoderLayer blocks.
        dim_feedforward: Feed-forward network hidden dimension.
        dropout: Dropout rate.
        activation: Activation in FFN ('relu' or 'gelu').
        pooling: Pooling strategy ('cls' for CLS token, 'mean' for average).

    Example:
        >>> encoder = TransformerEncoder(input_size=10, d_model=128)
        >>> x = torch.randn(32, 60, 10)  # (batch, seq_len, features)
        >>> out = encoder(x)  # (32, 128)
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        pooling: str = "cls",
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.pooling = pooling

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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
        x = self.pos_encoding(x)

        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+seq, d_model]

        x = self.transformer(x)  # [B, seq', d_model]
        x = self.norm(x)

        x = x[:, 0] if self.pooling == "cls" else x.mean(dim=1)  # [B, d_model]

        return x

    @classmethod
    def default_params(cls) -> dict:
        """Default parameters for Transformer encoder."""
        return {
            "input_size": 10,
            "d_model": 128,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "activation": "gelu",
            "pooling": "cls",
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
            "small": {"d_model": [64, 128], "layers": (2, 3), "ffn": [128, 256]},
            "medium": {"d_model": [128, 256], "layers": (3, 5), "ffn": [256, 512]},
            "large": {"d_model": [256, 512], "layers": (4, 6), "ffn": [512, 1024]},
        }

        config = size_config[model_size]

        return {
            "input_size": 10,
            "d_model": {"type": "categorical", "choices": config["d_model"]},
            "nhead": {"type": "categorical", "choices": [2, 4, 8]},
            "num_layers": {"type": "int", "low": config["layers"][0], "high": config["layers"][1]},
            "dim_feedforward": {"type": "categorical", "choices": config["ffn"]},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "activation": "gelu",
            "pooling": {"type": "categorical", "choices": ["cls", "mean"]},
        }
