"""iTransformer encoder for time series.

iTransformer (Inverted Transformer) from ICLR 2024 inverts the attention
mechanism - instead of attending across time (which loses locality),
it attends across features/variables while keeping temporal structure.

Based on: "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
https://arxiv.org/abs/2310.06625
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from signalflow import SfTorchModuleMixin, sf_component


class InvertedMultiHeadAttention(nn.Module):
    """Multi-head attention over features (variables) instead of time.

    Standard Transformer: attention(Q, K, V) where Q, K, V are (B, T, D)
    iTransformer: attention over features, so Q, K, V are (B, D, T)
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(seq_len, seq_len)
        self.k_proj = nn.Linear(seq_len, seq_len)
        self.v_proj = nn.Linear(seq_len, seq_len)
        self.out_proj = nn.Linear(seq_len, seq_len)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, n_vars, seq_len)
               where n_vars is the number of features/variables

        Returns:
            Output tensor of shape (batch, n_vars, seq_len)
        """
        batch, n_vars, seq_len = x.shape

        # Project queries, keys, values
        # Shape: (B, n_vars, seq_len)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        # (B, n_vars, seq_len) -> (B, n_heads, n_vars // n_heads, seq_len)
        # Note: We attend over variables, not time
        q = q.view(batch, self.n_heads, n_vars // self.n_heads, seq_len)
        k = k.view(batch, self.n_heads, n_vars // self.n_heads, seq_len)
        v = v.view(batch, self.n_heads, n_vars // self.n_heads, seq_len)

        # Attention weights over features
        # (B, n_heads, n_vars_per_head, seq_len) @ (B, n_heads, seq_len, n_vars_per_head)
        # -> (B, n_heads, n_vars_per_head, n_vars_per_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        # (B, n_heads, n_vars_per_head, n_vars_per_head) @ (B, n_heads, n_vars_per_head, seq_len)
        # -> (B, n_heads, n_vars_per_head, seq_len)
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.view(batch, n_vars, seq_len)

        return self.out_proj(out)


class VariableAttention(nn.Module):
    """Simplified variable attention for iTransformer.

    Attends across variables (features) at each time step,
    capturing cross-variate dependencies.
    """

    def __init__(
        self,
        n_vars: int,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, n_vars, d_model)

        Returns:
            Output tensor of shape (batch, n_vars, d_model)
        """
        batch, n_vars, d_model = x.shape

        # Project
        q = self.q_proj(x)  # (B, V, D)
        k = self.k_proj(x)  # (B, V, D)
        v = self.v_proj(x)  # (B, V, D)

        # Reshape for multi-head attention
        q = q.view(batch, n_vars, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, V, D/H)
        k = k.view(batch, n_vars, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, V, D/H)
        v = v.view(batch, n_vars, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, V, D/H)

        # Attention over variables
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, V, V)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = torch.matmul(attn, v)  # (B, H, V, D/H)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch, n_vars, d_model)

        return self.out_proj(out)


class FeedForward(nn.Module):
    """Feed-forward network."""

    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class iTransformerBlock(nn.Module):
    """Single iTransformer block."""

    def __init__(
        self,
        n_vars: int,
        d_model: int,
        n_heads: int = 8,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = VariableAttention(
            n_vars=n_vars,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connections."""
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


@sf_component(name="encoder/itransformer")
class iTransformerEncoder(nn.Module, SfTorchModuleMixin):
    """iTransformer encoder for time series classification.

    Unlike standard Transformers that attend over time, iTransformer
    inverts the attention to attend over features/variables. This
    preserves temporal locality and captures cross-variate dependencies.

    Key insight: For time series, features at different variables but
    same time step are often more related than same variable at
    different time steps.

    Args:
        input_size: Number of input features per timestep (n_vars)
        seq_len: Sequence length
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer blocks
        d_ff: Feed-forward dimension (default: 4 * d_model)
        dropout: Dropout rate
        pool: Pooling method ('last', 'mean', 'max', 'cls')

    Example:
        >>> encoder = iTransformerEncoder(input_size=10, seq_len=60, d_model=64)
        >>> x = torch.randn(32, 60, 10)  # (batch, seq_len, features)
        >>> out = encoder(x)  # (32, 64)
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int = 60,
        d_model: int = 64,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int | None = None,
        dropout: float = 0.1,
        pool: str = "mean",
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size  # n_vars
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff or d_model * 4
        self.dropout_rate = dropout
        self.pool = pool

        # Embedding: project each variable's time series to d_model
        # Input: (B, T, V) -> we treat each variable as a token
        # Embed: (B, V, T) -> (B, V, D) via linear projection
        self.embed = nn.Linear(seq_len, d_model)

        # CLS token for classification
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer blocks
        n_vars = input_size + (1 if pool == "cls" else 0)
        self.layers = nn.ModuleList(
            [
                iTransformerBlock(
                    n_vars=n_vars,
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=self.d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # Final projection
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
            Output tensor of shape (batch, d_model)
        """
        batch, _seq_len, _n_vars = x.shape

        # Transpose: (B, T, V) -> (B, V, T)
        x = x.transpose(1, 2)

        # Embed each variable's time series
        # (B, V, T) -> (B, V, D)
        x = self.embed(x)

        # Add CLS token if using cls pooling
        if self.pool == "cls":
            cls_tokens = self.cls_token.expand(batch, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x)

        # Final norm
        x = self.norm(x)

        # Pool to fixed size output
        if self.pool == "cls":
            return x[:, 0, :]  # CLS token
        elif self.pool == "mean":
            return x.mean(dim=1)
        elif self.pool == "max":
            return x.max(dim=1).values
        elif self.pool == "last":
            return x[:, -1, :]
        else:
            return x.mean(dim=1)

    @classmethod
    def default_params(cls) -> dict:
        """Default parameters for iTransformer encoder."""
        return {
            "input_size": 10,
            "seq_len": 60,
            "d_model": 64,
            "n_heads": 8,
            "n_layers": 3,
            "d_ff": None,
            "dropout": 0.1,
            "pool": "mean",
        }

    @classmethod
    def tune(cls, trial, model_size: str = "small") -> dict:
        """Optuna hyperparameter search space.

        Args:
            trial: Optuna trial object
            model_size: Size variant ('small', 'medium', 'large')

        Returns:
            Dictionary of hyperparameters
        """
        size_config = {
            "small": {"d_model": (32, 64), "n_layers": (2, 3), "n_heads": (4, 8)},
            "medium": {"d_model": (64, 128), "n_layers": (3, 4), "n_heads": (8, 8)},
            "large": {"d_model": (128, 256), "n_layers": (4, 6), "n_heads": (8, 16)},
        }

        config = size_config[model_size]
        d_model = trial.suggest_int("d_model", *config["d_model"], step=16)

        # Ensure n_heads divides d_model
        n_heads = trial.suggest_categorical("n_heads", [h for h in [4, 8, 16] if d_model % h == 0])

        return {
            "input_size": 10,  # Fixed, depends on features
            "seq_len": 60,  # Fixed, depends on data
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": trial.suggest_int("n_layers", *config["n_layers"]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "pool": trial.suggest_categorical("pool", ["mean", "cls"]),
        }
