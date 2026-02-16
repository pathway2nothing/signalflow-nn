"""Mamba encoder for efficient sequence modeling.

Mamba is a State Space Model (SSM) that achieves O(T) complexity
for sequence modeling, compared to O(T^2) for Transformers.

Based on: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
https://arxiv.org/abs/2312.00752
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from signalflow import SfTorchModuleMixin, sf_component


class SelectiveSSM(nn.Module):
    """Selective State Space Model (S6) block.

    The core of Mamba - a data-dependent SSM that selectively
    propagates or forgets information along the sequence.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = dt_rank

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Conv layer for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias,
        )

        # SSM parameters
        # x_proj: project x to (delta, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)

        # dt_proj: project dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt projection to preserve variance
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A parameter (learnable)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Input projection: split into x and z (gating)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Conv1d for local context
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.conv1d(x)[:, :, :seq_len]  # Causal padding
        x = x.transpose(1, 2)  # (B, L, D)

        # Activation
        x = F.silu(x)

        # SSM
        y = self._ssm(x)

        # Gating
        z = F.silu(z)
        y = y * z

        # Output projection
        return self.out_proj(y)

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Selective State Space Model.

        Args:
            x: Input tensor of shape (batch, seq_len, d_inner)

        Returns:
            Output tensor of shape (batch, seq_len, d_inner)
        """
        batch, seq_len, _ = x.shape
        d_state = self.d_state

        # Compute data-dependent parameters
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        delta, B, C = x_dbl.split([self.dt_rank, d_state, d_state], dim=-1)

        # Compute delta
        delta = F.softplus(self.dt_proj(delta))  # (B, L, D)

        # Get A from log
        A = -torch.exp(self.A_log.float())  # (D, N)

        # Discretize A and B
        # A_bar = exp(delta * A)
        # B_bar = (exp(delta * A) - I) * inv(A) * B
        # For simplicity, use first-order approximation

        # Scan (sequential for now, can be parallelized with associative scan)
        y = self._scan(x, delta, A, B, C)

        # Add D * x (skip connection)
        y = y + x * self.D

        return y

    def _scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Scan operation for SSM.

        This is the sequential version. For production, use
        associative scan for O(log T) parallel complexity.
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # Initialize hidden state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)

        ys = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (B, D)
            delta_t = delta[:, t, :]  # (B, D)
            B_t = B[:, t, :]  # (B, N)
            C_t = C[:, t, :]  # (B, N)

            # Discretize
            delta_A = torch.exp(delta_t.unsqueeze(-1) * A)  # (B, D, N)
            delta_B = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (B, D, N)

            # Update hidden state
            h = delta_A * h + delta_B * x_t.unsqueeze(-1)  # (B, D, N)

            # Compute output
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)  # (B, D)
            ys.append(y_t)

        return torch.stack(ys, dim=1)


class MambaBlock(nn.Module):
    """Single Mamba block with residual connection and normalization."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        return x + self.dropout(self.mamba(self.norm(x)))


@sf_component(name="encoder/mamba")
class MambaEncoder(nn.Module, SfTorchModuleMixin):
    """Mamba encoder for efficient temporal sequence modeling.

    Mamba is a State Space Model that achieves O(T) complexity
    compared to O(T^2) for Transformers, making it efficient
    for long sequences.

    Args:
        input_size: Number of input features per timestep
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Local convolution kernel size
        expand: Expansion factor for inner dimension
        n_layers: Number of Mamba blocks
        dropout: Dropout rate
        pool: Pooling method ('last', 'mean', 'max')

    Example:
        >>> encoder = MambaEncoder(input_size=10, d_model=64, n_layers=4)
        >>> x = torch.randn(32, 120, 10)  # (batch, seq_len, features)
        >>> out = encoder(x)  # (32, 64)
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
        dropout: float = 0.1,
        pool: str = "last",
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.n_layers = n_layers
        self.dropout_rate = dropout
        self.pool = pool

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)

        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Final normalization
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
        # Project input
        x = self.input_proj(x)

        # Apply Mamba blocks
        for layer in self.layers:
            x = layer(x)

        # Final norm
        x = self.norm(x)

        # Pool to fixed size output
        if self.pool == "last":
            return x[:, -1, :]
        elif self.pool == "mean":
            return x.mean(dim=1)
        elif self.pool == "max":
            return x.max(dim=1).values
        else:
            return x[:, -1, :]

    @classmethod
    def default_params(cls) -> dict:
        """Default parameters for Mamba encoder."""
        return {
            "input_size": 10,
            "d_model": 64,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "n_layers": 4,
            "dropout": 0.1,
            "pool": "last",
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
            "small": {"d_model": (32, 64), "n_layers": (2, 4)},
            "medium": {"d_model": (64, 128), "n_layers": (4, 6)},
            "large": {"d_model": (128, 256), "n_layers": (6, 8)},
        }

        config = size_config[model_size]

        return {
            "input_size": 10,  # Fixed, depends on features
            "d_model": trial.suggest_int("d_model", *config["d_model"], step=16),
            "d_state": trial.suggest_int("d_state", 8, 32, step=8),
            "d_conv": trial.suggest_int("d_conv", 2, 6),
            "expand": trial.suggest_int("expand", 1, 4),
            "n_layers": trial.suggest_int("n_layers", *config["n_layers"]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "pool": trial.suggest_categorical("pool", ["last", "mean"]),
        }
