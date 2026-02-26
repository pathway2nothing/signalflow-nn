"""OmniScale CNN encoder for temporal features.

Based on: Tang et al. "Omni-Scale CNNs: a simple and effective kernel
size configuration for time series classification" (2022).
ICLR 2022.
"""

import torch
import torch.nn as nn

from signalflow import SfTorchModuleMixin
from signalflow.core import feature


class OmniScaleBlock(nn.Module):
    """OmniScale block with multiple parallel receptive field sizes.

    Uses a set of Conv1d layers with geometrically increasing kernel sizes
    and max-pools their outputs to capture patterns at all scales.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels per scale.
        receptive_field_sizes: List of kernel sizes to use.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        receptive_field_sizes: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if receptive_field_sizes is None:
            receptive_field_sizes = [3, 5, 7, 11, 15, 21]

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, k, padding="same", bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                )
                for k in receptive_field_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: apply all scales, element-wise max across scales.

        Args:
            x: Input of shape (batch, in_channels, seq_len)

        Returns:
            Output of shape (batch, out_channels, seq_len)
        """
        # Apply each scale conv and stack
        scale_outputs = torch.stack([conv(x) for conv in self.convs], dim=0)  # [n_scales, B, C, L]
        # Element-wise max across scales
        out = scale_outputs.max(dim=0).values  # [B, C, L]
        return self.dropout(out)


@feature("encoder/omniscale")
class OmniScaleCNNEncoder(nn.Module, SfTorchModuleMixin):
    """OmniScale CNN encoder for sequence processing.

    Uses multiple parallel convolutions with different kernel sizes and
    takes element-wise maximum across scales. This captures patterns at
    all temporal scales without needing to choose a specific kernel size.

    Args:
        input_size: Number of input features per timestep.
        num_filters: Number of filters per scale per block.
        num_blocks: Number of OmniScale blocks.
        receptive_field_sizes: List of kernel sizes (receptive fields).
        dropout: Dropout rate.

    Example:
        >>> encoder = OmniScaleCNNEncoder(input_size=10, num_filters=64)
        >>> x = torch.randn(32, 60, 10)  # (batch, seq_len, features)
        >>> out = encoder(x)  # (32, 64)
    """

    def __init__(
        self,
        input_size: int,
        num_filters: int = 64,
        num_blocks: int = 3,
        receptive_field_sizes: list[int] | None = None,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        if receptive_field_sizes is None:
            receptive_field_sizes = [3, 5, 7, 11, 15, 21]

        self.input_size = input_size

        blocks: list[nn.Module] = []
        in_ch = input_size
        for _ in range(num_blocks):
            blocks.append(OmniScaleBlock(in_ch, num_filters, receptive_field_sizes, dropout))
            in_ch = num_filters

        self.blocks = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self._output_size = num_filters

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
        x = self.blocks(x)  # [B, num_filters, seq_len]
        x = self.global_pool(x)  # [B, num_filters, 1]
        x = x.squeeze(-1)  # [B, num_filters]
        return x

    @classmethod
    def default_params(cls) -> dict:
        """Default parameters for OmniScale CNN encoder."""
        return {
            "input_size": 10,
            "num_filters": 64,
            "num_blocks": 3,
            "receptive_field_sizes": [3, 5, 7, 11, 15, 21],
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
        rf_options = [
            [3, 5, 7, 11],
            [3, 5, 7, 11, 15, 21],
            [3, 7, 15, 31, 63],
        ]

        return {
            "input_size": 10,
            "num_filters": {"type": "int", "low": config["filters"][0], "high": config["filters"][1]},
            "num_blocks": {"type": "int", "low": config["blocks"][0], "high": config["blocks"][1]},
            "omniscale_rf_idx": {"type": "int", "low": 0, "high": 2},
            "rf_options": rf_options,
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        }
