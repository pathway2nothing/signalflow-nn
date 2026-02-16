"""PatchTST encoder for temporal features.

Based on: Nie et al. "A Time Series is Worth 64 Words: Long-term
Forecasting with Transformers" (2023). arXiv:2211.14730.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from signalflow import SfTorchModuleMixin, sf_component


class PatchEmbedding(nn.Module):
    """Split sequence into patches and project to embedding dimension.

    Args:
        patch_len: Length of each patch.
        patch_stride: Stride between patches.
        d_model: Embedding dimension for each patch.
    """

    def __init__(self, patch_len: int, patch_stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.projection = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create and embed patches.

        Args:
            x: Input of shape (batch, seq_len)

        Returns:
            Patch embeddings of shape (batch, num_patches, d_model)
        """
        # Pad if needed so at least one patch fits
        seq_len = x.size(1)
        if seq_len < self.patch_len:
            x = F.pad(x, (0, self.patch_len - seq_len))

        # Unfold into patches: [batch, num_patches, patch_len]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        return self.projection(patches)  # [batch, num_patches, d_model]


@sf_component(name="encoder/patchtst")
class PatchTSTEncoder(nn.Module, SfTorchModuleMixin):
    """Patch-based Time Series Transformer encoder.

    Splits each channel independently into patches, embeds them,
    and processes through a shared Transformer encoder. Channel
    independence reduces overfitting and enables efficient processing.

    Args:
        input_size: Number of input features (channels) per timestep.
        d_model: Embedding dimension.
        patch_len: Length of each patch.
        patch_stride: Stride between patches.
        nhead: Number of attention heads.
        num_layers: Number of Transformer layers.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout rate.

    Example:
        >>> encoder = PatchTSTEncoder(input_size=10, d_model=128, patch_len=16, patch_stride=8)
        >>> x = torch.randn(32, 60, 10)  # (batch, seq_len, features)
        >>> out = encoder(x)  # (32, 128)
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        patch_len: int = 16,
        patch_stride: int = 8,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.patch_len = patch_len
        self.patch_stride = patch_stride

        self.patch_embedding = PatchEmbedding(patch_len, patch_stride, d_model)

        # Learnable positional encoding for patches
        max_patches = 512
        self.pos_encoding = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)
        self.pos_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
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
        """Forward pass with channel-independent patching.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Output tensor of shape (batch, output_size)
        """
        batch_size, seq_len, num_channels = x.shape

        # Channel independence: process each channel separately
        # [B, seq, C] -> [B, C, seq] -> [B*C, seq]
        x = x.transpose(1, 2).reshape(batch_size * num_channels, seq_len)

        # Create patch embeddings: [B*C, num_patches, d_model]
        x = self.patch_embedding(x)
        num_patches = x.size(1)

        # Add positional encoding
        x = x + self.pos_encoding[:, :num_patches]
        x = self.pos_dropout(x)

        # Transformer: [B*C, num_patches, d_model]
        x = self.transformer(x)
        x = self.norm(x)

        # Mean pooling over patches: [B*C, d_model]
        x = x.mean(dim=1)

        # Aggregate channels: [B, C, d_model] -> mean -> [B, d_model]
        x = x.view(batch_size, num_channels, -1).mean(dim=1)

        return x

    @classmethod
    def default_params(cls) -> dict:
        """Default parameters for PatchTST encoder."""
        return {
            "input_size": 10,
            "d_model": 128,
            "patch_len": 16,
            "patch_stride": 8,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 256,
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
            "small": {"d_model": [64, 128], "layers": (2, 3), "ffn": [128, 256]},
            "medium": {"d_model": [128, 256], "layers": (3, 5), "ffn": [256, 512]},
            "large": {"d_model": [256, 512], "layers": (4, 6), "ffn": [512, 1024]},
        }

        config = size_config[model_size]
        d_model = trial.suggest_categorical("patchtst_d_model", config["d_model"])
        nhead = trial.suggest_categorical("patchtst_nhead", [2, 4, 8])
        while d_model % nhead != 0:
            nhead = nhead // 2

        patch_len = trial.suggest_categorical("patchtst_patch_len", [8, 16, 32])

        return {
            "input_size": 10,
            "d_model": d_model,
            "patch_len": patch_len,
            "patch_stride": patch_len // 2,
            "nhead": nhead,
            "num_layers": trial.suggest_int("patchtst_num_layers", *config["layers"]),
            "dim_feedforward": trial.suggest_categorical("patchtst_ffn", config["ffn"]),
            "dropout": trial.suggest_float("patchtst_dropout", 0.0, 0.5),
        }
