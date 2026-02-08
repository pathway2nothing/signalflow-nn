"""Shared fixtures for signalflow-nn tests."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import torch


@pytest.fixture
def num_features() -> int:
    return 5


@pytest.fixture
def num_pairs() -> int:
    return 3


@pytest.fixture
def num_timestamps() -> int:
    return 200


@pytest.fixture
def feature_cols(num_features: int) -> list[str]:
    return [f"feature_{i}" for i in range(num_features)]


@pytest.fixture
def pair_names(num_pairs: int) -> list[str]:
    return [f"PAIR_{i}" for i in range(num_pairs)]


@pytest.fixture
def base_timestamps(num_timestamps: int) -> list[datetime]:
    start = datetime(2024, 1, 1)
    return [start + timedelta(hours=i) for i in range(num_timestamps)]


@pytest.fixture
def features_df(
    pair_names: list[str],
    base_timestamps: list[datetime],
    feature_cols: list[str],
) -> pl.DataFrame:
    """Sample features DataFrame: [pair, timestamp, feature_0..feature_N]."""
    rng = np.random.default_rng(42)
    rows = []
    for pair in pair_names:
        for ts in base_timestamps:
            row = {"pair": pair, "timestamp": ts}
            for col in feature_cols:
                row[col] = rng.standard_normal()
            rows.append(row)

    return pl.DataFrame(rows).sort(["pair", "timestamp"])


@pytest.fixture
def signals_df(
    pair_names: list[str],
    base_timestamps: list[datetime],
    num_timestamps: int,
) -> pl.DataFrame:
    """Sample signals DataFrame: [pair, timestamp, label]."""
    rng = np.random.default_rng(42)
    rows = []
    for pair in pair_names:
        # pick ~20% of timestamps as signal timestamps, but skip first 80 to ensure enough history
        signal_indices = rng.choice(range(80, num_timestamps), size=num_timestamps // 5, replace=False)
        for idx in sorted(signal_indices):
            rows.append(
                {
                    "pair": pair,
                    "timestamp": base_timestamps[idx],
                    "label": int(rng.integers(0, 3)),
                }
            )

    return pl.DataFrame(rows).sort(["pair", "timestamp"])


@pytest.fixture
def sample_batch(num_features: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample batch: (x, y) for model forward pass."""
    batch_size = 8
    seq_len = 60
    x = torch.randn(batch_size, seq_len, num_features)
    y = torch.randint(0, 3, (batch_size,))
    return x, y


@pytest.fixture
def encoder_input(num_features: int) -> torch.Tensor:
    """Encoder input tensor [batch, seq_len, features]."""
    return torch.randn(8, 60, num_features)


@pytest.fixture
def head_input() -> torch.Tensor:
    """Head input tensor [batch, input_size]."""
    return torch.randn(8, 64)
