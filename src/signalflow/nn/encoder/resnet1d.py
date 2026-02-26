"""ResNet-1D encoder for temporal features.

Adapted from: He et al. "Deep Residual Learning for Image Recognition"
(2015) for 1D time series data.
"""

import torch
import torch.nn as nn

from signalflow import SfTorchModuleMixin
from signalflow.core import feature


class ResidualBlock1d(nn.Module):
    """Residual block for 1D sequences.

    Two Conv1d layers with BatchNorm and ReLU, plus a skip connection.
    If input and output channels differ, a 1x1 conv adjusts the shortcut.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Stride for the first conv (used for downsampling).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


@feature("encoder/resnet1d")
class ResNet1dEncoder(nn.Module, SfTorchModuleMixin):
    """ResNet-1D encoder for sequence processing.

    Adapted ResNet architecture for 1D temporal data with residual blocks,
    progressive channel doubling, and stride-based downsampling.

    Args:
        input_size: Number of input features per timestep.
        base_filters: Number of filters in the first stage.
        num_blocks: List of block counts per stage. Each stage doubles
            the channel count and downsamples with stride=2.
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.

    Example:
        >>> encoder = ResNet1dEncoder(input_size=10, base_filters=64)
        >>> x = torch.randn(32, 60, 10)  # (batch, seq_len, features)
        >>> out = encoder(x)  # (32, 256) with default 3 stages
    """

    def __init__(
        self,
        input_size: int,
        base_filters: int = 64,
        num_blocks: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        if num_blocks is None:
            num_blocks = [2, 2, 2]

        self.input_size = input_size

        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv1d(input_size, base_filters, 7, stride=2, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
        )

        # Build stages
        stages: list[nn.Module] = []
        in_channels = base_filters
        for stage_idx, n_blocks in enumerate(num_blocks):
            out_channels = base_filters * (2**stage_idx)
            for block_idx in range(n_blocks):
                stride = 2 if block_idx == 0 and stage_idx > 0 else 1
                stages.append(ResidualBlock1d(in_channels, out_channels, kernel_size, stride, dropout))
                in_channels = out_channels

        self.stages = nn.Sequential(*stages)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self._output_size = in_channels

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
        x = x.transpose(1, 2)  # [batch, input_size, seq_len]
        x = self.stem(x)  # [batch, base_filters, seq_len/2]
        x = self.stages(x)  # [batch, out_channels, seq_len']
        x = self.global_pool(x)  # [batch, out_channels, 1]
        x = x.squeeze(-1)  # [batch, out_channels]
        return x

    @classmethod
    def default_params(cls) -> dict:
        """Default parameters for ResNet-1D encoder."""
        return {
            "input_size": 10,
            "base_filters": 64,
            "num_blocks": [2, 2, 2],
            "kernel_size": 3,
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
            "small": {"filters": (32, 64), "blocks_options": [[1, 1], [2, 2], [2, 2, 2]]},
            "medium": {"filters": (64, 128), "blocks_options": [[2, 2, 2], [3, 3, 3], [2, 2, 2, 2]]},
            "large": {"filters": (64, 128), "blocks_options": [[3, 3, 3], [3, 4, 6, 3], [2, 2, 2, 2, 2]]},
        }

        config = size_config[model_size]

        return {
            "input_size": 10,
            "base_filters": {"type": "int", "low": config["filters"][0], "high": config["filters"][1]},
            "resnet1d_blocks_idx": {"type": "int", "low": 0, "high": len(config["blocks_options"]) - 1},
            "blocks_options": config["blocks_options"],
            "kernel_size": {"type": "categorical", "choices": [3, 5, 7]},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        }
