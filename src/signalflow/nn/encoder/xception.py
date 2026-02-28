"""XceptionTime encoder for temporal features.

Based on: Rahimian et al. "XceptionTime: Independent Time-Window
XceptionBased Architecture for Hand Gesture Classification" (2020).
Adapted from Chollet "Xception" (2017) for 1D time series.
"""

import torch
import torch.nn as nn
from signalflow.core import register

from signalflow import SfTorchModuleMixin


class SeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution.

    Depthwise conv (per-channel) followed by pointwise conv (1x1).
    Much more parameter-efficient than standard convolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for depthwise conv.
        stride: Stride for depthwise conv.
        padding: Padding mode or size.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str | int = "same",
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class XceptionBlock(nn.Module):
    """Xception block: two separable convolutions with residual connection.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for separable convolutions.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 39,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.sep_conv1 = SeparableConv1d(in_channels, out_channels, kernel_size)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.sep_conv2 = SeparableConv1d(out_channels, out_channels, kernel_size)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.sep_conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.sep_conv2(out))
        out = self.relu(out + residual)
        return out


@register("encoder/xception")
class XceptionTimeEncoder(nn.Module, SfTorchModuleMixin):
    """XceptionTime encoder for sequence processing.

    Uses depthwise separable convolutions instead of standard convolutions,
    providing similar expressiveness with significantly fewer parameters.
    Suitable for time series of varying lengths due to adaptive pooling.

    Args:
        input_size: Number of input features per timestep.
        num_filters: Base number of filters (doubled each block).
        num_blocks: Number of Xception blocks.
        kernel_size: Kernel size for separable convolutions.
        dropout: Dropout rate.

    Example:
        >>> encoder = XceptionTimeEncoder(input_size=10, num_filters=64)
        >>> x = torch.randn(32, 60, 10)  # (batch, seq_len, features)
        >>> out = encoder(x)  # (32, 256) with default 3 blocks
    """

    def __init__(
        self,
        input_size: int,
        num_filters: int = 64,
        num_blocks: int = 3,
        kernel_size: int = 39,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size

        # Initial projection
        self.stem = nn.Sequential(
            nn.Conv1d(input_size, num_filters, 1, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
        )

        # Xception blocks with progressive channel doubling
        blocks: list[nn.Module] = []
        in_ch = num_filters
        for i in range(num_blocks):
            out_ch = num_filters * (2**i)
            blocks.append(XceptionBlock(in_ch, out_ch, kernel_size, dropout))
            in_ch = out_ch

        self.blocks = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self._output_size = in_ch

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
        x = x.transpose(1, 2)  # [B, input_size, seq_len]
        x = self.stem(x)  # [B, num_filters, seq_len]
        x = self.blocks(x)  # [B, out_channels, seq_len]
        x = self.global_pool(x)  # [B, out_channels, 1]
        x = x.squeeze(-1)  # [B, out_channels]
        return x

    @classmethod
    def default_params(cls) -> dict:
        """Default parameters for XceptionTime encoder."""
        return {
            "input_size": 10,
            "num_filters": 64,
            "num_blocks": 3,
            "kernel_size": 39,
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
            "small": {"filters": (32, 64), "blocks": (2, 3)},
            "medium": {"filters": (64, 128), "blocks": (3, 4)},
            "large": {"filters": (128, 256), "blocks": (3, 5)},
        }

        config = size_config[model_size]

        return {
            "input_size": 10,
            "num_filters": {"type": "int", "low": config["filters"][0], "high": config["filters"][1]},
            "num_blocks": {"type": "int", "low": config["blocks"][0], "high": config["blocks"][1]},
            "kernel_size": {"type": "categorical", "choices": [15, 25, 39, 51]},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        }
