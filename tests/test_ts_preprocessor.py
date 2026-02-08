"""Tests for TimeSeriesPreprocessor."""

import pickle
import tempfile

import numpy as np
import polars as pl
import pytest

from signalflow.nn.data.ts_preprocessor import ScalerConfig, TimeSeriesPreprocessor


@pytest.fixture
def preprocessor() -> TimeSeriesPreprocessor:
    return TimeSeriesPreprocessor(
        default_config=ScalerConfig(method="standard", scope="group"),
        group_col="asset_id",
    )


@pytest.fixture
def ts_df() -> pl.DataFrame:
    """DataFrame with asset_id for preprocessor tests."""
    rng = np.random.default_rng(42)
    rows = []
    for asset in ["BTC", "ETH"]:
        for i in range(100):
            rows.append(
                {
                    "asset_id": asset,
                    "timestamp": i,
                    "close": rng.standard_normal() * 10 + (100 if asset == "BTC" else 50),
                    "volume": abs(rng.standard_normal() * 1000),
                }
            )
    return pl.DataFrame(rows)


class TestScalerConfig:
    def test_default_values(self):
        config = ScalerConfig()
        assert config.method == "robust"
        assert config.scope == "group"

    def test_custom_values(self):
        config = ScalerConfig(method="standard", scope="global")
        assert config.method == "standard"
        assert config.scope == "global"


class TestTimeSeriesPreprocessorInit:
    def test_default_init(self):
        prep = TimeSeriesPreprocessor()
        assert prep.fill_strategy == "forward"
        assert prep.group_col == "asset_id"
        assert prep.time_col == "timestamp"
        assert prep.fitted_params == {}

    def test_custom_init(self):
        prep = TimeSeriesPreprocessor(
            fill_strategy="zero",
            group_col="pair",
            time_col="ts",
        )
        assert prep.fill_strategy == "zero"
        assert prep.group_col == "pair"
        assert prep.time_col == "ts"


class TestFitTransform:
    def test_fit_stores_feature_names(self, preprocessor, ts_df):
        preprocessor.fit(ts_df, feature_cols=["close", "volume"])
        assert preprocessor.feature_names == ["close", "volume"]

    def test_fit_creates_params(self, preprocessor, ts_df):
        preprocessor.fit(ts_df, feature_cols=["close", "volume"])
        assert "close" in preprocessor.fitted_params
        assert "volume" in preprocessor.fitted_params

    def test_fit_grouped_standard(self, ts_df):
        prep = TimeSeriesPreprocessor(
            default_config=ScalerConfig(method="standard", scope="group"),
            group_col="asset_id",
        )
        prep.fit(ts_df, feature_cols=["close"])
        params = prep.fitted_params["close"]
        assert "stats_df" in params
        stats = params["stats_df"]
        assert "close_mean" in stats.columns
        assert "close_std" in stats.columns

    def test_fit_grouped_robust(self, ts_df):
        prep = TimeSeriesPreprocessor(
            default_config=ScalerConfig(method="robust", scope="group"),
            group_col="asset_id",
        )
        prep.fit(ts_df, feature_cols=["close"])
        params = prep.fitted_params["close"]
        assert "stats_df" in params
        stats = params["stats_df"]
        assert "close_median" in stats.columns
        assert "close_iqr" in stats.columns

    def test_fit_grouped_minmax(self, ts_df):
        prep = TimeSeriesPreprocessor(
            default_config=ScalerConfig(method="minmax", scope="group"),
            group_col="asset_id",
        )
        prep.fit(ts_df, feature_cols=["close"])
        params = prep.fitted_params["close"]
        assert "stats_df" in params
        stats = params["stats_df"]
        assert "close_min" in stats.columns
        assert "close_max" in stats.columns

    def test_transform_returns_dataframe(self, preprocessor, ts_df):
        preprocessor.fit(ts_df, feature_cols=["close", "volume"])
        result = preprocessor.transform(ts_df)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == ts_df.shape

    def test_transform_standard_scaling(self, ts_df):
        prep = TimeSeriesPreprocessor(
            default_config=ScalerConfig(method="standard", scope="group"),
            group_col="asset_id",
        )
        prep.fit(ts_df, feature_cols=["close"])
        result = prep.transform(ts_df)
        # After standard scaling per group, mean should be ~0
        for asset in ["BTC", "ETH"]:
            scaled = result.filter(pl.col("asset_id") == asset)["close"]
            assert abs(scaled.mean()) < 0.5  # approximate zero-mean

    def test_transform_preserves_non_feature_cols(self, preprocessor, ts_df):
        preprocessor.fit(ts_df, feature_cols=["close"])
        result = preprocessor.transform(ts_df)
        assert "asset_id" in result.columns
        assert "timestamp" in result.columns

    def test_fit_invalid_scope_raises(self, ts_df):
        prep = TimeSeriesPreprocessor(
            default_config=ScalerConfig(method="standard", scope="invalid"),
            group_col="asset_id",
        )
        with pytest.raises(ValueError, match="Unknown scope"):
            prep.fit(ts_df, feature_cols=["close"])


class TestNanHandling:
    def test_forward_fill(self, ts_df):
        df_with_nan = ts_df.with_columns(
            pl.when(pl.col("close") > 100).then(None).otherwise(pl.col("close")).alias("close")
        )
        prep = TimeSeriesPreprocessor(
            default_config=ScalerConfig(method="standard", scope="group"),
            group_col="asset_id",
            fill_strategy="forward",
        )
        prep.fit(df_with_nan, feature_cols=["close"])
        result = prep.transform(df_with_nan)
        assert result["close"].null_count() == 0

    def test_zero_fill(self, ts_df):
        df_with_nan = ts_df.with_columns(
            pl.when(pl.col("close") > 100).then(None).otherwise(pl.col("close")).alias("close")
        )
        prep = TimeSeriesPreprocessor(
            default_config=ScalerConfig(method="standard", scope="group"),
            group_col="asset_id",
            fill_strategy="zero",
        )
        prep.fit(df_with_nan, feature_cols=["close"])
        result = prep.transform(df_with_nan)
        assert result["close"].null_count() == 0


class TestSaveLoad:
    def test_save_load_roundtrip(self, preprocessor, ts_df):
        preprocessor.fit(ts_df, feature_cols=["close", "volume"])

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            preprocessor.save(f.name)
            loaded = TimeSeriesPreprocessor.load(f.name)

        assert loaded.feature_names == preprocessor.feature_names
        assert loaded.fill_strategy == preprocessor.fill_strategy
        assert set(loaded.fitted_params.keys()) == set(preprocessor.fitted_params.keys())

    def test_loaded_preprocessor_transforms(self, preprocessor, ts_df):
        preprocessor.fit(ts_df, feature_cols=["close"])

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            preprocessor.save(f.name)
            loaded = TimeSeriesPreprocessor.load(f.name)

        original = preprocessor.transform(ts_df)
        from_loaded = loaded.transform(ts_df)
        assert original["close"].to_list() == from_loaded["close"].to_list()


class TestFeatureConfigs:
    def test_per_feature_config(self, ts_df):
        prep = TimeSeriesPreprocessor(
            feature_configs={
                "close": ScalerConfig(method="standard", scope="group"),
                "volume": ScalerConfig(method="minmax", scope="group"),
            },
            default_config=ScalerConfig(method="robust", scope="group"),
            group_col="asset_id",
        )
        prep.fit(ts_df, feature_cols=["close", "volume"])
        assert prep.fitted_params["close"]["config"].method == "standard"
        assert prep.fitted_params["volume"]["config"].method == "minmax"
