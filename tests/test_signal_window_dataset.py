"""Tests for SignalWindowDataset."""

import numpy as np
import polars as pl
import pytest
import torch

from signalflow.nn.data.signal_window_dataset import SignalWindowDataset


class TestSignalWindowDatasetInit:
    def test_creates_dataset(self, features_df, signals_df, feature_cols):
        dataset = SignalWindowDataset(
            features_df=features_df,
            signals_df=signals_df,
            window_size=10,
            feature_cols=feature_cols,
        )
        assert len(dataset) > 0

    def test_auto_detects_feature_cols(self, features_df, signals_df):
        dataset = SignalWindowDataset(
            features_df=features_df,
            signals_df=signals_df,
            window_size=10,
        )
        assert len(dataset.feature_cols) > 0
        assert "pair" not in dataset.feature_cols
        assert "timestamp" not in dataset.feature_cols

    def test_window_size_stored(self, features_df, signals_df, feature_cols):
        dataset = SignalWindowDataset(
            features_df=features_df,
            signals_df=signals_df,
            window_size=30,
            feature_cols=feature_cols,
        )
        assert dataset.window_size == 30


class TestWindowCreation:
    def test_window_shape(self, features_df, signals_df, feature_cols, num_features):
        window_size = 10
        dataset = SignalWindowDataset(
            features_df=features_df,
            signals_df=signals_df,
            window_size=window_size,
            feature_cols=feature_cols,
        )
        x, y = dataset[0]
        assert x.shape == (window_size, num_features)
        assert y.shape == ()

    def test_returns_tensors(self, features_df, signals_df, feature_cols):
        dataset = SignalWindowDataset(
            features_df=features_df,
            signals_df=signals_df,
            window_size=10,
            feature_cols=feature_cols,
        )
        x, y = dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.dtype == torch.float32
        assert y.dtype == torch.long

    def test_all_windows_valid(self, features_df, signals_df, feature_cols, num_features):
        window_size = 10
        dataset = SignalWindowDataset(
            features_df=features_df,
            signals_df=signals_df,
            window_size=window_size,
            feature_cols=feature_cols,
        )
        for i in range(len(dataset)):
            x, y = dataset[i]
            assert x.shape == (window_size, num_features)
            assert not torch.isnan(x).any()


class TestDilatedStride:
    def test_dilated_window(self, features_df, signals_df, feature_cols, num_features):
        window_size = 10
        window_timeframe = 3
        dataset = SignalWindowDataset(
            features_df=features_df,
            signals_df=signals_df,
            window_size=window_size,
            window_timeframe=window_timeframe,
            feature_cols=feature_cols,
        )
        if len(dataset) > 0:
            x, y = dataset[0]
            assert x.shape == (window_size, num_features)

    def test_larger_timeframe_fewer_windows(self, features_df, signals_df, feature_cols):
        ds_small = SignalWindowDataset(
            features_df=features_df,
            signals_df=signals_df,
            window_size=10,
            window_timeframe=1,
            feature_cols=feature_cols,
        )
        ds_large = SignalWindowDataset(
            features_df=features_df,
            signals_df=signals_df,
            window_size=10,
            window_timeframe=5,
            feature_cols=feature_cols,
        )
        # larger timeframe requires more history, so fewer valid windows
        assert len(ds_large) <= len(ds_small)


class TestEdgeCases:
    def test_no_matching_pairs(self, signals_df, feature_cols):
        empty_features = pl.DataFrame(
            {
                "pair": ["UNKNOWN"],
                "timestamp": [signals_df["timestamp"][0]],
                **{col: [0.0] for col in feature_cols},
            }
        )
        dataset = SignalWindowDataset(
            features_df=empty_features,
            signals_df=signals_df,
            window_size=10,
            feature_cols=feature_cols,
        )
        assert len(dataset) == 0

    def test_insufficient_history(self, features_df, signals_df, feature_cols):
        # Very large window that exceeds available history
        dataset = SignalWindowDataset(
            features_df=features_df,
            signals_df=signals_df,
            window_size=500,
            feature_cols=feature_cols,
        )
        assert len(dataset) == 0

    def test_valid_signal_indices_tracked(self, features_df, signals_df, feature_cols):
        dataset = SignalWindowDataset(
            features_df=features_df,
            signals_df=signals_df,
            window_size=10,
            feature_cols=feature_cols,
        )
        assert len(dataset.valid_signal_indices) == len(dataset)
        # All indices should be within signal range
        for idx in dataset.valid_signal_indices:
            assert 0 <= idx < signals_df.height

    def test_nan_in_features_replaced(self, signals_df, feature_cols):
        rng = np.random.default_rng(42)
        rows = []
        for pair in ["PAIR_0"]:
            for i in range(200):
                from datetime import datetime, timedelta

                ts = datetime(2024, 1, 1) + timedelta(hours=i)
                row = {"pair": pair, "timestamp": ts}
                for col in feature_cols:
                    row[col] = float("nan") if i % 10 == 0 else rng.standard_normal()
                rows.append(row)
        df_with_nans = pl.DataFrame(rows)

        # Filter signals to only PAIR_0
        sigs = signals_df.filter(pl.col("pair") == "PAIR_0")
        if sigs.height == 0:
            pytest.skip("No signals for PAIR_0")

        dataset = SignalWindowDataset(
            features_df=df_with_nans,
            signals_df=sigs,
            window_size=10,
            feature_cols=feature_cols,
        )
        if len(dataset) > 0:
            x, _ = dataset[0]
            assert not torch.isnan(x).any()
