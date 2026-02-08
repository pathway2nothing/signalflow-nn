"""Tests for TemporalValidator."""

import pickle
import tempfile
from pathlib import Path

import polars as pl
import pytest
import torch

from signalflow.nn.validator.temporal_validator import TemporalValidator
from signalflow.nn.data.ts_preprocessor import ScalerConfig, TimeSeriesPreprocessor


@pytest.fixture
def validator(num_features):
    return TemporalValidator(
        encoder_type="encoder/lstm",
        encoder_params={
            "input_size": num_features,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
        },
        head_type="head/cls/mlp",
        head_params={"hidden_sizes": [32], "dropout": 0.1},
        window_size=10,
        window_timeframe=1,
        num_classes=3,
        batch_size=4,
        max_epochs=1,
        num_workers=0,
    )


class TestTemporalValidatorInit:
    def test_creates_validator(self, validator):
        assert validator is not None
        assert validator.encoder_type == "encoder/lstm"
        assert validator.window_size == 10
        assert validator.num_classes == 3

    def test_model_not_created_on_init(self, validator):
        assert validator.model is None

    def test_setup_model(self, validator):
        validator._setup_model()
        assert validator.model is not None

    def test_infer_input_size(self, validator, features_df):
        size = validator._infer_input_size(features_df)
        # features_df has pair, timestamp + feature_cols
        assert size > 0

    def test_infer_input_size_with_feature_cols(self, num_features):
        v = TemporalValidator(
            encoder_type="encoder/lstm",
            encoder_params={"input_size": num_features, "hidden_size": 32},
            feature_cols=["a", "b", "c"],
        )
        dummy_df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        assert v._infer_input_size(dummy_df) == 3


class TestSaveLoad:
    def test_save_load_roundtrip(self, validator):
        validator._setup_model()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        validator.save(path)
        loaded = TemporalValidator.load(path)

        assert loaded.encoder_type == validator.encoder_type
        assert loaded.window_size == validator.window_size
        assert loaded.num_classes == validator.num_classes
        assert loaded.model is not None

    def test_save_without_model(self, validator):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        validator.save(path)
        loaded = TemporalValidator.load(path)
        assert loaded.model is None

    def test_save_with_preprocessor(self, validator):
        prep = TimeSeriesPreprocessor(
            default_config=ScalerConfig(method="standard", scope="group"),
            group_col="pair",
        )
        validator.preprocessor = prep
        validator._setup_model()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        validator.save(path)
        loaded = TemporalValidator.load(path)
        assert loaded.preprocessor is not None

    def test_save_creates_parent_dirs(self, validator):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "model.pkl"
            validator.save(path)
            assert path.exists()


class TestModelStateDict:
    def test_model_weights_preserved(self, validator):
        validator._setup_model()
        original_params = {name: param.clone() for name, param in validator.model.named_parameters()}

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        validator.save(path)
        loaded = TemporalValidator.load(path)

        for name, param in loaded.model.named_parameters():
            assert torch.equal(param, original_params[name]), f"Mismatch in {name}"
