"""ConvTran encoder for temporal features.

Based on: Foumani et al. "ConvTran: Improving Position Encoding of
Transformers for Multivariate Time Series Classification" (2024).
arXiv:2304.03754.

Hybrid CNN + Transformer: uses convolutional position encoding instead
of sinusoidal/learnable positional embeddings.
"""

import torch
import torch.nn as nn
from signalflow.core import feature

from signalflow import SfTorchModuleMixin


class ConvPositionalEncoding(nn.Module):
    """Convolutional positional encoding.

    Uses depthwise 1D convolution to inject position information
    instead of explicit positional embeddings. This approach is
    shift-equivariant and adapts to sequence length.

    Args:
        d_model: Embedding dimension.
        kernel_size: Kernel size for position conv (odd recommended).
    """

    def __init__(self, d_model: int, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add convolutional position encoding.

        Args:
            x: Input of shape (batch, seq_len, d_model)

        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        # [B, seq, d] -> [B, d, seq] -> conv -> [B, d, seq] -> [B, seq, d]
        pos = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x + pos


class ConvTranBlock(nn.Module):
    """ConvTran block: conv position encoding + multi-head attention + FFN.

    Args:
        d_model: Embedding dimension.
        nhead: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout rate.
        conv_kernel_size: Kernel size for position encoding conv.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        conv_kernel_size: int = 7,
    ):
        super().__init__()

        self.conv_pe = ConvPositionalEncoding(d_model, conv_kernel_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm residual connections.

        Args:
            x: Input of shape (batch, seq_len, d_model)

        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        # Convolutional position encoding
        x = self.conv_pe(x)

        # Multi-head attention with pre-norm
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + self.dropout1(attn_out)

        # FFN with pre-norm
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x


@feature("encoder/convtran")
class ConvTranEncoder(nn.Module, SfTorchModuleMixin):
    """ConvTran encoder for sequence processing.

    Hybrid CNN + Transformer that uses convolutional position encoding
    instead of sinusoidal/learnable embeddings. The conv-based PE is
    shift-equivariant and better captures local temporal structure.

    Args:
        input_size: Number of input features per timestep.
        d_model: Embedding dimension.
        nhead: Number of attention heads.
        num_layers: Number of ConvTran blocks.
        dim_feedforward: FFN hidden dimension.
        conv_kernel_size: Kernel size for positional encoding conv.
        dropout: Dropout rate.

    Example:
        >>> encoder = ConvTranEncoder(input_size=10, d_model=128)
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
        conv_kernel_size: int = 7,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.input_projection = nn.Linear(input_size, d_model)

        self.blocks = nn.Sequential(
            *[ConvTranBlock(d_model, nhead, dim_feedforward, dropout, conv_kernel_size) for _ in range(num_layers)]
        )

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
        """Default parameters for ConvTran encoder."""
        return {
            "input_size": 10,
            "d_model": 128,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 256,
            "conv_kernel_size": 7,
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
            "conv_kernel_size": {"type": "categorical", "choices": [3, 5, 7, 11]},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        }
