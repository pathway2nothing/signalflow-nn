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

        if self.pooling == "cls":
            x = x[:, 0]  # [B, d_model]
        else:
            x = x.mean(dim=1)  # [B, d_model]

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
    def tune(cls, trial, model_size: str = "small") -> dict:
        """Optuna hyperparameter search space.

        Args:
            trial: Optuna trial object.
            model_size: Size variant ('small', 'medium', 'large').

        Returns:
            Dictionary of hyperparameters.
        """
        size_config = {
            "small": {"d_model": [64, 128], "layers": (2, 3), "ffn": [128, 256]},
            "medium": {"d_model": [128, 256], "layers": (3, 5), "ffn": [256, 512]},
            "large": {"d_model": [256, 512], "layers": (4, 6), "ffn": [512, 1024]},
        }

        config = size_config[model_size]
        d_model = trial.suggest_categorical("transformer_d_model", config["d_model"])
        nhead = trial.suggest_categorical("transformer_nhead", [2, 4, 8])
        # Ensure d_model is divisible by nhead
        while d_model % nhead != 0:
            nhead = nhead // 2

        return {
            "input_size": 10,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": trial.suggest_int("transformer_num_layers", *config["layers"]),
            "dim_feedforward": trial.suggest_categorical("transformer_ffn", config["ffn"]),
            "dropout": trial.suggest_float("transformer_dropout", 0.0, 0.5),
            "activation": "gelu",
            "pooling": trial.suggest_categorical("transformer_pooling", ["cls", "mean"]),
        }
