"""XCM (eXplainable Convolutional Module) encoder for temporal features.

Based on: Fauvel et al. "XCM: An Explainable Convolutional Neural Network
for Multivariate Time Series Classification" (2021). arXiv:2009.04796.
"""

import torch
import torch.nn as nn
from signalflow.core import feature

from signalflow import SfTorchModuleMixin


@feature("encoder/xcm")
class XCMEncoder(nn.Module, SfTorchModuleMixin):
    """XCM encoder for sequence processing.

    Separates temporal and spatial (feature) learning into two parallel
    branches, then combines them. This explicit separation enables
    interpretability through attention-like weight inspection.

    - Temporal branch: Conv1d along time axis (shared across features)
    - Spatial branch: Conv1d along feature axis (shared across timesteps)
    - Combination: concatenation + 1x1 conv fusion

    Args:
        input_size: Number of input features per timestep.
        seq_len: Input sequence length (needed for spatial branch).
        num_filters_time: Filters for temporal convolution.
        num_filters_space: Filters for spatial convolution.
        kernel_size_time: Kernel size for temporal conv.
        kernel_size_space: Kernel size for spatial conv.
        d_model: Output embedding dimension.
        dropout: Dropout rate.

    Example:
        >>> encoder = XCMEncoder(input_size=10, seq_len=60, d_model=128)
        >>> x = torch.randn(32, 60, 10)  # (batch, seq_len, features)
        >>> out = encoder(x)  # (32, 128)
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        num_filters_time: int = 64,
        num_filters_space: int = 64,
        kernel_size_time: int = 5,
        kernel_size_space: int = 3,
        d_model: int = 128,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size

        # Temporal branch: learns temporal patterns per feature
        # Input: [B, input_size, seq_len] -> Conv1d along time
        self.time_branch = nn.Sequential(
            nn.Conv1d(input_size, num_filters_time, kernel_size_time, padding="same", bias=False),
            nn.BatchNorm1d(num_filters_time),
            nn.ReLU(),
            nn.Conv1d(num_filters_time, num_filters_time, kernel_size_time, padding="same", bias=False),
            nn.BatchNorm1d(num_filters_time),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # [B, num_filters_time, 1]
        )

        # Spatial branch: learns feature interactions per timestep
        # Input: [B, seq_len, input_size] -> Conv1d along features
        self.space_branch = nn.Sequential(
            nn.Conv1d(seq_len, num_filters_space, kernel_size_space, padding="same", bias=False),
            nn.BatchNorm1d(num_filters_space),
            nn.ReLU(),
            nn.Conv1d(num_filters_space, num_filters_space, kernel_size_space, padding="same", bias=False),
            nn.BatchNorm1d(num_filters_space),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # [B, num_filters_space, 1]
        )

        # Fusion
        combined_size = num_filters_time + num_filters_space
        self.fusion = nn.Sequential(
            nn.Linear(combined_size, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self._output_size = d_model

    @property
    def output_size(self) -> int:
        """Output embedding size."""
        return self._output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with separate temporal and spatial branches.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Output tensor of shape (batch, output_size)
        """
        # Temporal branch: [B, seq, feat] -> [B, feat, seq] -> pool -> [B, filters_t]
        x_time = x.transpose(1, 2)  # [B, input_size, seq_len]
        x_time = self.time_branch(x_time).squeeze(-1)  # [B, num_filters_time]

        # Spatial branch: [B, seq, feat] -> treat seq as channels -> pool -> [B, filters_s]
        x_space = self.space_branch(x).squeeze(-1)  # [B, num_filters_space]

        # Fuse
        combined = torch.cat([x_time, x_space], dim=1)  # [B, filters_t + filters_s]
        return self.fusion(combined)  # [B, d_model]

    @classmethod
    def default_params(cls) -> dict:
        """Default parameters for XCM encoder."""
        return {
            "input_size": 10,
            "seq_len": 60,
            "num_filters_time": 64,
            "num_filters_space": 64,
            "kernel_size_time": 5,
            "kernel_size_space": 3,
            "d_model": 128,
            "dropout": 0.1,
        }

    @classmethod
    def search_space(cls, model_size: str = "small") -> dict:
        """Hyperparameter search space.

        Args:
            model_size: Size variant ('small', 'medium', 'large').

        Returns:
            Dictionary of hyperparameter specs.
        """
        size_config = {
            "small": {"filters": (32, 64), "d_model": [64, 128]},
            "medium": {"filters": (64, 128), "d_model": [128, 256]},
            "large": {"filters": (128, 256), "d_model": [256, 512]},
        }

        config = size_config[model_size]

        return {
            "input_size": 10,
            "seq_len": 60,
            "num_filters_time": {"type": "int", "low": config["filters"][0], "high": config["filters"][1]},
            "num_filters_space": {"type": "int", "low": config["filters"][0], "high": config["filters"][1]},
            "kernel_size_time": {"type": "categorical", "choices": [3, 5, 7]},
            "kernel_size_space": {"type": "categorical", "choices": [3, 5]},
            "d_model": {"type": "categorical", "choices": config["d_model"]},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        }
