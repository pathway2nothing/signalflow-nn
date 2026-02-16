"""Temporal Convolutional Network (TCN) encoder for temporal features.

Based on: Bai et al. "An Empirical Evaluation of Generic Convolutional
and Recurrent Networks for Sequence Modeling" (2018).
"""

import torch
import torch.nn as nn

from signalflow import SfTorchModuleMixin, sf_component


class Chomp1d(nn.Module):
    """Remove extra right-side padding for causal convolution."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Single TCN residual block with dilated causal convolutions.

    Architecture: 2x (Conv1d dilated + WeightNorm + ReLU + Dropout) + residual.

    Args:
        n_inputs: Number of input channels.
        n_outputs: Number of output channels.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor for causal convolutions.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


@sf_component(name="encoder/tcn")
class TCNEncoder(nn.Module, SfTorchModuleMixin):
    """Temporal Convolutional Network encoder for sequence processing.

    Uses dilated causal convolutions with exponentially increasing dilation
    rates (1, 2, 4, 8, ...) and residual connections to capture long-range
    temporal dependencies efficiently.

    Receptive field = 1 + 2 * (kernel_size - 1) * sum(2^i for i in range(num_blocks)).

    Args:
        input_size: Number of input features per timestep.
        num_channels: List of channel counts for each temporal block.
            Length determines the network depth.
        kernel_size: Kernel size for all temporal blocks.
        dropout: Dropout rate.

    Example:
        >>> encoder = TCNEncoder(input_size=10, num_channels=[64, 64, 64])
        >>> x = torch.randn(32, 60, 10)  # (batch, seq_len, features)
        >>> out = encoder(x)  # (32, 64)
    """

    def __init__(
        self,
        input_size: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        if num_channels is None:
            num_channels = [64, 64, 64]

        self.input_size = input_size

        blocks: list[nn.Module] = []
        in_channels = input_size
        for i, out_channels in enumerate(num_channels):
            dilation = 2**i
            blocks.append(
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.network = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self._output_size = num_channels[-1]

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
        x = self.network(x)  # [batch, num_channels[-1], seq_len]
        x = self.global_pool(x)  # [batch, num_channels[-1], 1]
        x = x.squeeze(-1)  # [batch, num_channels[-1]]
        return x

    @classmethod
    def default_params(cls) -> dict:
        """Default parameters for TCN encoder."""
        return {
            "input_size": 10,
            "num_channels": [64, 64, 64],
            "kernel_size": 3,
            "dropout": 0.1,
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
            "small": {"max_blocks": 3, "channels": (32, 64), "kernels": [3, 5]},
            "medium": {"max_blocks": 4, "channels": (64, 128), "kernels": [3, 5, 7]},
            "large": {"max_blocks": 6, "channels": (128, 256), "kernels": [3, 5, 7]},
        }

        config = size_config[model_size]
        num_blocks = trial.suggest_int("tcn_num_blocks", 2, config["max_blocks"])
        channel_size = trial.suggest_int("tcn_channel_size", *config["channels"])

        return {
            "input_size": 10,
            "num_channels": [channel_size] * num_blocks,
            "kernel_size": trial.suggest_categorical("tcn_kernel_size", config["kernels"]),
            "dropout": trial.suggest_float("tcn_dropout", 0.0, 0.5),
        }
