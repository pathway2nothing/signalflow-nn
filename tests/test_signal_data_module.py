"""Tests for SignalDataModule."""

import polars as pl
import pytest

from signalflow.nn.data.signal_data_module import SignalDataModule
from signalflow.nn.data.ts_preprocessor import ScalerConfig, TimeSeriesPreprocessor


@pytest.fixture
def preprocessor():
    return TimeSeriesPreprocessor(
        default_config=ScalerConfig(method="standard", scope="group"),
        group_col="pair",
    )


@pytest.fixture
def data_module(features_df, signals_df, preprocessor, feature_cols):
    return SignalDataModule(
        features_df=features_df,
        signals_df=signals_df,
        preprocessor=preprocessor,
        window_size=10,
        window_timeframe=1,
        train_val_test_split=(0.7, 0.15, 0.15),
        split_strategy="temporal",
        batch_size=4,
        num_workers=0,
        feature_cols=feature_cols,
    )


class TestSignalDataModuleInit:
    def test_creates_module(self, data_module):
        assert data_module is not None
        assert data_module.window_size == 10
        assert data_module.batch_size == 4

    def test_auto_detects_feature_cols(self, features_df, signals_df, preprocessor):
        dm = SignalDataModule(
            features_df=features_df,
            signals_df=signals_df,
            preprocessor=preprocessor,
            window_size=10,
            num_workers=0,
        )
        assert dm.feature_cols is not None
        assert "pair" not in dm.feature_cols
        assert "timestamp" not in dm.feature_cols


class TestTemporalSplit:
    def test_setup_creates_splits(self, data_module):
        data_module.setup()
        assert data_module.train_signals is not None
        assert data_module.val_signals is not None
        assert data_module.test_signals is not None

    def test_split_sizes(self, data_module):
        data_module.setup()
        total = len(data_module.train_signals) + len(data_module.val_signals) + len(data_module.test_signals)
        assert total == len(data_module.signals_df)

    def test_temporal_ordering(self, data_module):
        data_module.setup()
        train_max = data_module.train_signals["timestamp"].max()
        val_min = data_module.val_signals["timestamp"].min()
        assert train_max <= val_min

    def test_processed_features_created(self, data_module):
        data_module.setup()
        assert data_module.processed_features is not None
        assert isinstance(data_module.processed_features, pl.DataFrame)


class TestRandomSplit:
    def test_random_split(self, features_df, signals_df, preprocessor, feature_cols):
        dm = SignalDataModule(
            features_df=features_df,
            signals_df=signals_df,
            preprocessor=preprocessor,
            window_size=10,
            split_strategy="random",
            batch_size=4,
            num_workers=0,
            feature_cols=feature_cols,
        )
        dm.setup()
        total = len(dm.train_signals) + len(dm.val_signals) + len(dm.test_signals)
        assert total == len(signals_df)


class TestPairSplit:
    def test_pair_split(self, features_df, signals_df, preprocessor, feature_cols):
        dm = SignalDataModule(
            features_df=features_df,
            signals_df=signals_df,
            preprocessor=preprocessor,
            window_size=10,
            split_strategy="pair",
            batch_size=4,
            num_workers=0,
            feature_cols=feature_cols,
        )
        dm.setup()
        # No pair overlap between splits
        train_pairs = set(dm.train_signals["pair"].unique().to_list())
        val_pairs = set(dm.val_signals["pair"].unique().to_list())
        test_pairs = set(dm.test_signals["pair"].unique().to_list())
        assert train_pairs.isdisjoint(val_pairs)
        assert train_pairs.isdisjoint(test_pairs)


class TestDataLoaders:
    def test_train_dataloader(self, data_module):
        data_module.setup()
        dl = data_module.train_dataloader()
        batch = next(iter(dl))
        x, y = batch
        assert x.ndim == 3  # [batch, seq_len, features]
        assert y.ndim == 1  # [batch]

    def test_val_dataloader(self, data_module):
        data_module.setup()
        dl = data_module.val_dataloader()
        batch = next(iter(dl))
        x, y = batch
        assert x.ndim == 3

    def test_test_dataloader(self, data_module):
        data_module.setup()
        dl = data_module.test_dataloader()
        batch = next(iter(dl))
        x, y = batch
        assert x.ndim == 3

    def test_setup_idempotent(self, data_module):
        data_module.setup()
        train_len = len(data_module.train_signals)
        data_module.setup()  # second call should not change anything
        assert len(data_module.train_signals) == train_len


class TestInvalidStrategy:
    def test_unknown_strategy_raises(self, features_df, signals_df, preprocessor, feature_cols):
        dm = SignalDataModule(
            features_df=features_df,
            signals_df=signals_df,
            preprocessor=preprocessor,
            window_size=10,
            split_strategy="unknown",
            num_workers=0,
            feature_cols=feature_cols,
        )
        with pytest.raises(ValueError, match="Unknown split_strategy"):
            dm.setup()
