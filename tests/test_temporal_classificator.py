"""Tests for TemporalClassificator (Lightning module)."""

import pytest
import torch

from signalflow.nn.model.temporal_classificator import TemporalClassificator, TrainingConfig


@pytest.fixture
def model(num_features):
    return TemporalClassificator(
        encoder_type="encoder/lstm",
        encoder_params={
            "input_size": num_features,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
        },
        head_type="head/cls/mlp",
        head_params={"hidden_sizes": [32], "dropout": 0.1},
        num_classes=3,
        training_config=TrainingConfig(learning_rate=1e-3),
    )


class TestTrainingConfig:
    def test_default_values(self):
        config = TrainingConfig()
        assert config.learning_rate == 1e-3
        assert config.optimizer == "adamw"
        assert config.scheduler == "reduce_on_plateau"

    def test_to_dict(self):
        config = TrainingConfig(learning_rate=5e-4)
        d = config.to_dict()
        assert d["learning_rate"] == 5e-4
        assert "optimizer" in d
        assert "scheduler" in d

    def test_from_dict(self):
        d = {"learning_rate": 2e-4, "optimizer": "adam"}
        config = TrainingConfig.from_dict(d)
        assert config.learning_rate == 2e-4
        assert config.optimizer == "adam"
        # defaults for unspecified keys
        assert config.scheduler == "reduce_on_plateau"

    def test_from_dict_ignores_unknown(self):
        d = {"learning_rate": 1e-3, "unknown_key": 42}
        config = TrainingConfig.from_dict(d)
        assert config.learning_rate == 1e-3


class TestTemporalClassificatorInit:
    def test_creates_model(self, model):
        assert model.encoder is not None
        assert model.head is not None

    def test_default_head(self, num_features):
        model = TemporalClassificator(
            encoder_type="encoder/lstm",
            encoder_params={"input_size": num_features, "hidden_size": 32},
            head_type=None,
            num_classes=3,
        )
        # Default head should be nn.Linear
        assert isinstance(model.head, torch.nn.Linear)

    def test_gru_encoder(self, num_features):
        model = TemporalClassificator(
            encoder_type="encoder/gru",
            encoder_params={"input_size": num_features, "hidden_size": 32},
            num_classes=3,
        )
        assert model.encoder is not None


class TestForwardPass:
    def test_forward_shape(self, model, sample_batch):
        x, y = sample_batch
        logits = model(x)
        assert logits.shape == (x.shape[0], 3)

    def test_forward_output_type(self, model, sample_batch):
        x, _ = sample_batch
        logits = model(x)
        assert logits.dtype == torch.float32


class TestTrainingStep:
    def test_training_step(self, model, sample_batch):
        loss = model.training_step(sample_batch, batch_idx=0)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_validation_step(self, model, sample_batch):
        loss = model.validation_step(sample_batch, batch_idx=0)
        assert loss.ndim == 0

    def test_test_step(self, model, sample_batch):
        loss = model.test_step(sample_batch, batch_idx=0)
        assert loss.ndim == 0

    def test_predict_step(self, model, sample_batch):
        probs = model.predict_step(sample_batch, batch_idx=0)
        assert probs.shape == (sample_batch[0].shape[0], 3)
        # Probabilities should sum to ~1
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestConfigureOptimizers:
    def test_adamw_reduce_on_plateau(self, num_features):
        model = TemporalClassificator(
            encoder_type="encoder/lstm",
            encoder_params={"input_size": num_features, "hidden_size": 32},
            num_classes=3,
            training_config=TrainingConfig(optimizer="adamw", scheduler="reduce_on_plateau"),
        )
        result = model.configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_adam_cosine(self, num_features):
        model = TemporalClassificator(
            encoder_type="encoder/lstm",
            encoder_params={"input_size": num_features, "hidden_size": 32},
            num_classes=3,
            training_config=TrainingConfig(optimizer="adam", scheduler="cosine"),
        )
        result = model.configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_sgd_no_scheduler(self, num_features):
        model = TemporalClassificator(
            encoder_type="encoder/lstm",
            encoder_params={"input_size": num_features, "hidden_size": 32},
            num_classes=3,
            training_config=TrainingConfig(optimizer="sgd", scheduler="none"),
        )
        result = model.configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" not in result

    def test_unknown_optimizer_raises(self, num_features):
        model = TemporalClassificator(
            encoder_type="encoder/lstm",
            encoder_params={"input_size": num_features, "hidden_size": 32},
            num_classes=3,
            training_config=TrainingConfig(),
        )
        model._training_config.optimizer = "invalid"
        with pytest.raises(ValueError, match="Unknown optimizer"):
            model.configure_optimizers()

    def test_unknown_scheduler_raises(self, num_features):
        model = TemporalClassificator(
            encoder_type="encoder/lstm",
            encoder_params={"input_size": num_features, "hidden_size": 32},
            num_classes=3,
            training_config=TrainingConfig(),
        )
        model._training_config.scheduler = "invalid"
        with pytest.raises(ValueError, match="Unknown scheduler"):
            model.configure_optimizers()


class TestClassWeights:
    def test_with_class_weights(self, num_features, sample_batch):
        model = TemporalClassificator(
            encoder_type="encoder/lstm",
            encoder_params={"input_size": num_features, "hidden_size": 32},
            num_classes=3,
            class_weights=[1.0, 2.0, 1.5],
        )
        loss = model.training_step(sample_batch, batch_idx=0)
        assert loss.item() > 0


class TestDefaultParams:
    def test_default_params(self):
        params = TemporalClassificator.default_params()
        assert "encoder_type" in params
        assert "encoder_params" in params
        assert "head_type" in params
        assert "num_classes" in params


class TestFromConfig:
    def test_from_config(self, num_features):
        model = TemporalClassificator.from_config(
            encoder_type="encoder/gru",
            encoder_params={"input_size": num_features, "hidden_size": 64},
            head_type="head/cls/mlp",
            head_params={"hidden_sizes": [32]},
            num_classes=3,
            training_config={"learning_rate": 5e-4},
        )
        assert model._training_config.learning_rate == 5e-4

    def test_from_config_defaults(self, num_features):
        model = TemporalClassificator.from_config(
            encoder_type="encoder/lstm",
            encoder_params={"input_size": num_features, "hidden_size": 32},
        )
        assert model._num_classes == 3


class TestGradientFlow:
    def test_backward_pass(self, model, sample_batch):
        loss = model.training_step(sample_batch, batch_idx=0)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
