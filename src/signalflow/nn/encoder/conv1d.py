"""Conv1D encoder for temporal features."""

import torch
import torch.nn as nn
from signalflow.core import feature

from signalflow import SfTorchModuleMixin


@feature("encoder/conv1d")
class Conv1dEncoder(nn.Module, SfTorchModuleMixin):
    """Stacked 1D convolutional encoder for sequence processing.

    Processes temporal sequences through Conv1d -> BatchNorm -> Activation -> Dropout
    blocks, followed by global average pooling to produce fixed-size embeddings.

    Args:
        input_size: Number of input features per timestep.
        num_filters: List of filter counts for each conv block.
        kernel_sizes: Kernel size(s); int for uniform, list for per-layer.
        dropout: Dropout rate between layers.
        use_pooling: Whether to add MaxPool1d(2) between blocks.
        activation: Activation function name ('relu', 'gelu', 'silu').

    Example:
        >>> encoder = Conv1dEncoder(input_size=10, num_filters=[64, 128])
        >>> x = torch.randn(32, 60, 10)  # (batch, seq_len, features)
        >>> out = encoder(x)  # (32, 128)
    """

    def __init__(
        self,
        input_size: int,
        num_filters: list[int] | None = None,
        kernel_sizes: list[int] | int = 3,
        dropout: float = 0.1,
        use_pooling: bool = False,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__()

        if num_filters is None:
            num_filters = [64, 128]

        self.input_size = input_size

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(num_filters)

        if len(kernel_sizes) != len(num_filters):
            raise ValueError(
                f"kernel_sizes length ({len(kernel_sizes)}) must match num_filters length ({len(num_filters)})"
            )

        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
        act_cls = act_fn.get(activation, nn.ReLU)

        layers: list[nn.Module] = []
        in_channels = input_size
        for filters, kernel in zip(num_filters, kernel_sizes, strict=False):
            layers.append(nn.Conv1d(in_channels, filters, kernel, padding=kernel // 2))
            layers.append(nn.BatchNorm1d(filters))
            layers.append(act_cls())
            layers.append(nn.Dropout(dropout))
            if use_pooling:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = filters

        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self._output_size = num_filters[-1]

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
        x = self.conv_layers(x)  # [batch, num_filters[-1], seq_len']
        x = self.global_pool(x)  # [batch, num_filters[-1], 1]
        x = x.squeeze(-1)  # [batch, num_filters[-1]]
        return x

    @classmethod
    def default_params(cls) -> dict:
        """Default parameters for Conv1d encoder."""
        return {
            "input_size": 10,
            "num_filters": [64, 128],
            "kernel_sizes": 3,
            "dropout": 0.1,
            "use_pooling": False,
            "activation": "relu",
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
            "small": {"max_layers": 2, "filters": (32, 64), "kernels": [3, 5]},
            "medium": {"max_layers": 3, "filters": (64, 128), "kernels": [3, 5, 7]},
            "large": {"max_layers": 4, "filters": (128, 256), "kernels": [3, 5, 7]},
        }

        config = size_config[model_size]

        return {
            "input_size": 10,
            "conv1d_num_layers": {"type": "int", "low": 1, "high": config["max_layers"]},
            "conv1d_filters": {"type": "int", "low": config["filters"][0], "high": config["filters"][1]},
            "kernel_sizes": {"type": "categorical", "choices": config["kernels"]},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "use_pooling": {"type": "categorical", "choices": [False, True]},
            "activation": {"type": "categorical", "choices": ["relu", "gelu", "silu"]},
        }
