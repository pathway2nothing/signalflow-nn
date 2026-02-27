"""InceptionTime encoder for temporal features.

Based on: Fawaz et al. "InceptionTime: Finding AlexNet for
Time Series Classification" (2019). arXiv:1909.04939.
"""

import torch
import torch.nn as nn
from signalflow.core import feature

from signalflow import SfTorchModuleMixin


class InceptionModule(nn.Module):
    """Single inception module with multi-scale convolutions.

    Parallel branches with different kernel sizes capture patterns
    at multiple temporal scales. A bottleneck 1x1 conv reduces
    dimensionality before the learned-kernel branches.

    Args:
        in_channels: Number of input channels.
        num_filters: Number of filters per branch.
        kernel_sizes: Kernel sizes for parallel conv branches.
        bottleneck_channels: Bottleneck dimension (1x1 conv).
    """

    def __init__(
        self,
        in_channels: int,
        num_filters: int = 32,
        kernel_sizes: list[int] | None = None,
        bottleneck_channels: int = 32,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [10, 20, 40]

        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)

        self.conv_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(bottleneck_channels, num_filters, k, padding="same", bias=False),
                    nn.BatchNorm1d(num_filters),
                )
                for k in kernel_sizes
            ]
        )

        self.maxpool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, num_filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_filters),
        )

        self.activation = nn.ReLU()
        self.out_channels = num_filters * (len(kernel_sizes) + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bottleneck = self.bottleneck(x)
        conv_outs = [branch(x_bottleneck) for branch in self.conv_branches]
        pool_out = self.maxpool_branch(x)
        out = torch.cat([*conv_outs, pool_out], dim=1)
        return self.activation(out)


class InceptionResidualBlock(nn.Module):
    """Stack of inception modules with a residual shortcut connection.

    Args:
        in_channels: Number of input channels.
        num_filters: Number of filters per branch in each inception module.
        kernel_sizes: Kernel sizes for parallel conv branches.
        bottleneck_channels: Bottleneck dimension.
        depth: Number of inception modules in this block.
    """

    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        kernel_sizes: list[int],
        bottleneck_channels: int,
        depth: int = 3,
    ):
        super().__init__()

        modules: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(depth):
            module = InceptionModule(current_channels, num_filters, kernel_sizes, bottleneck_channels)
            modules.append(module)
            current_channels = module.out_channels

        self.inception_modules = nn.ModuleList(modules)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, current_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(current_channels),
        )
        self.activation = nn.ReLU()
        self.out_channels = current_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = x
        for module in self.inception_modules:
            out = module(out)
        out = out + self.shortcut(residual)
        return self.activation(out)


@feature("encoder/inception")
class InceptionTimeEncoder(nn.Module, SfTorchModuleMixin):
    """InceptionTime encoder for sequence processing.

    Multi-scale inception modules with residual connections capture
    temporal patterns at different scales simultaneously. Each inception
    module applies parallel convolutions with different kernel sizes,
    concatenates results, and uses bottleneck layers for efficiency.

    Args:
        input_size: Number of input features per timestep.
        num_filters: Number of filters per branch in each inception module.
        kernel_sizes: Kernel sizes for the parallel conv branches.
        bottleneck_channels: Bottleneck dimension (1x1 conv before branches).
        num_blocks: Number of inception residual blocks.
        depth_per_block: Number of inception modules per residual block.
        dropout: Dropout rate after global pooling.

    Example:
        >>> encoder = InceptionTimeEncoder(input_size=10, num_filters=32)
        >>> x = torch.randn(32, 60, 10)  # (batch, seq_len, features)
        >>> out = encoder(x)  # (32, 128) with default 3 kernels + 1 pool = 4 branches * 32
    """

    def __init__(
        self,
        input_size: int,
        num_filters: int = 32,
        kernel_sizes: list[int] | None = None,
        bottleneck_channels: int = 32,
        num_blocks: int = 2,
        depth_per_block: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [10, 20, 40]

        self.input_size = input_size

        blocks: list[nn.Module] = []
        in_channels = input_size
        for _ in range(num_blocks):
            block = InceptionResidualBlock(
                in_channels=in_channels,
                num_filters=num_filters,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
                depth=depth_per_block,
            )
            blocks.append(block)
            in_channels = block.out_channels

        self.blocks = nn.ModuleList(blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
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
        for block in self.blocks:
            x = block(x)  # [batch, out_channels, seq_len]
        x = self.global_pool(x)  # [batch, out_channels, 1]
        x = x.squeeze(-1)  # [batch, out_channels]
        x = self.dropout(x)
        return x

    @classmethod
    def default_params(cls) -> dict:
        """Default parameters for InceptionTime encoder."""
        return {
            "input_size": 10,
            "num_filters": 32,
            "kernel_sizes": [10, 20, 40],
            "bottleneck_channels": 32,
            "num_blocks": 2,
            "depth_per_block": 3,
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
            "small": {
                "filters": (16, 32),
                "blocks": (1, 2),
                "depth": (2, 3),
                "bottleneck": (16, 32),
                "kernels_options": [[10, 20, 40], [5, 11, 21]],
            },
            "medium": {
                "filters": (32, 64),
                "blocks": (2, 3),
                "depth": (3, 4),
                "bottleneck": (32, 64),
                "kernels_options": [[10, 20, 40], [5, 11, 21]],
            },
            "large": {
                "filters": (64, 128),
                "blocks": (2, 4),
                "depth": (3, 6),
                "bottleneck": (32, 64),
                "kernels_options": [[10, 20, 40], [20, 40, 80]],
            },
        }

        config = size_config[model_size]

        return {
            "input_size": 10,
            "num_filters": {"type": "int", "low": config["filters"][0], "high": config["filters"][1]},
            "inception_kernels_idx": {"type": "categorical", "choices": [0, 1]},
            "kernels_options": config["kernels_options"],
            "bottleneck_channels": {"type": "int", "low": config["bottleneck"][0], "high": config["bottleneck"][1]},
            "num_blocks": {"type": "int", "low": config["blocks"][0], "high": config["blocks"][1]},
            "depth_per_block": {"type": "int", "low": config["depth"][0], "high": config["depth"][1]},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        }
