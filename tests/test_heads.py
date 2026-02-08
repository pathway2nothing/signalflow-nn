"""Tests for all 7 classification/regression heads."""

import pytest
import torch

from signalflow.nn.head.linear_head import LinearClassifierHead
from signalflow.nn.head.mlp_head import MLPClassifierHead
from signalflow.nn.head.attention_head import AttentionClassifierHead
from signalflow.nn.head.residual_head import ResidualClassifierHead
from signalflow.nn.head.ordinal_head import OrdinalRegressionHead, OrdinalCrossEntropyLoss
from signalflow.nn.head.distribution_head import DistributionHead
from signalflow.nn.head.confidence_head import ClassificationWithConfidenceHead, ConfidenceLoss


INPUT_SIZE = 64
NUM_CLASSES = 3
BATCH_SIZE = 8


@pytest.fixture
def x():
    return torch.randn(BATCH_SIZE, INPUT_SIZE)


class TestLinearClassifierHead:
    def test_forward_shape(self, x):
        head = LinearClassifierHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
        out = head(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_default_params(self):
        params = LinearClassifierHead.default_params()
        assert "bias" in params

    def test_no_bias(self, x):
        head = LinearClassifierHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES, bias=False)
        out = head(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)


class TestMLPClassifierHead:
    def test_forward_shape(self, x):
        head = MLPClassifierHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES, hidden_sizes=[128, 64])
        out = head(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_no_hidden_layers(self, x):
        head = MLPClassifierHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES, hidden_sizes=[])
        out = head(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_default_params(self):
        params = MLPClassifierHead.default_params()
        assert "hidden_sizes" in params
        assert "dropout" in params

    def test_activations(self, x):
        for act in ["relu", "gelu", "silu", "tanh"]:
            head = MLPClassifierHead(
                input_size=INPUT_SIZE,
                num_classes=NUM_CLASSES,
                hidden_sizes=[32],
                activation=act,
            )
            out = head(x)
            assert out.shape == (BATCH_SIZE, NUM_CLASSES)


class TestAttentionClassifierHead:
    def test_forward_2d(self, x):
        head = AttentionClassifierHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES, num_heads=4, hidden_dim=64)
        out = head(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_forward_3d(self):
        x_3d = torch.randn(BATCH_SIZE, 10, INPUT_SIZE)
        head = AttentionClassifierHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES, num_heads=4, hidden_dim=64)
        out = head(x_3d)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_hidden_dim_adjusted_for_heads(self, x):
        # hidden_dim=65 is not divisible by num_heads=4, should be adjusted
        head = AttentionClassifierHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES, num_heads=4, hidden_dim=65)
        out = head(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_default_params(self):
        params = AttentionClassifierHead.default_params()
        assert "num_heads" in params
        assert "hidden_dim" in params


class TestResidualClassifierHead:
    def test_forward_shape(self, x):
        head = ResidualClassifierHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES, hidden_dim=64, num_blocks=2)
        out = head(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_identity_projection(self, x):
        # When input_size == hidden_dim, should use Identity
        head = ResidualClassifierHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES, hidden_dim=INPUT_SIZE)
        out = head(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_default_params(self):
        params = ResidualClassifierHead.default_params()
        assert "hidden_dim" in params
        assert "num_blocks" in params


class TestOrdinalRegressionHead:
    def test_forward_shape(self, x):
        head = OrdinalRegressionHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
        out = head(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_output_is_probability(self, x):
        head = OrdinalRegressionHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
        probs = head(x)
        # Probabilities should sum to ~1
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(BATCH_SIZE), atol=1e-4)
        # All values should be non-negative
        assert (probs >= 0).all()

    def test_num_classes_validation(self):
        with pytest.raises(ValueError, match="num_classes must be >= 2"):
            OrdinalRegressionHead(input_size=INPUT_SIZE, num_classes=1)

    def test_5_classes(self, x):
        head = OrdinalRegressionHead(input_size=INPUT_SIZE, num_classes=5)
        out = head(x)
        assert out.shape == (BATCH_SIZE, 5)

    def test_get_logits(self, x):
        head = OrdinalRegressionHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
        logits = head.get_logits(x)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_get_score(self, x):
        head = OrdinalRegressionHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
        score = head.get_score(x)
        assert score.shape == (BATCH_SIZE, 1)

    def test_default_params(self):
        params = OrdinalRegressionHead.default_params()
        assert "hidden_sizes" in params


class TestOrdinalCrossEntropyLoss:
    def test_loss_computation(self):
        probs = torch.softmax(torch.randn(BATCH_SIZE, NUM_CLASSES), dim=1)
        targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
        loss_fn = OrdinalCrossEntropyLoss()
        loss = loss_fn(probs, targets)
        assert loss.ndim == 0
        assert loss.item() > 0


class TestDistributionHead:
    def test_forward_prob(self, x):
        head = DistributionHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES, output_type="prob")
        out = head(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)
        sums = out.sum(dim=1)
        assert torch.allclose(sums, torch.ones(BATCH_SIZE), atol=1e-5)

    def test_forward_log_prob(self, x):
        head = DistributionHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES, output_type="log_prob")
        out = head(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)
        assert (out <= 0).all()  # log probs are negative

    def test_forward_logits(self, x):
        head = DistributionHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES, output_type="logits")
        out = head(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_temperature_scaling(self, x):
        head_low = DistributionHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES, temperature=0.1)
        head_high = DistributionHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES, temperature=10.0)
        # Lower temperature -> sharper distribution (higher max prob)
        out_low = head_low(x)
        out_high = head_high(x)
        # high temperature should be closer to uniform
        assert out_high.std(dim=1).mean() < out_low.std(dim=1).mean()

    def test_get_logits(self, x):
        head = DistributionHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
        logits = head.get_logits(x)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_default_params(self):
        params = DistributionHead.default_params()
        assert "temperature" in params
        assert "output_type" in params


class TestClassificationWithConfidenceHead:
    def test_forward_shape(self, x):
        head = ClassificationWithConfidenceHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
        logits, confidence = head(x)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
        assert confidence.shape == (BATCH_SIZE, 1)

    def test_confidence_range(self, x):
        head = ClassificationWithConfidenceHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
        _, confidence = head(x)
        # Confidence should be in [0, 1] due to sigmoid
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()

    def test_get_logits(self, x):
        head = ClassificationWithConfidenceHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
        logits = head.get_logits(x)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_predict_with_confidence(self, x):
        head = ClassificationWithConfidenceHead(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
        preds, confs, mask = head.predict_with_confidence(x, threshold=0.5)
        assert preds.shape == (BATCH_SIZE,)
        assert confs.shape == (BATCH_SIZE,)
        assert mask.shape == (BATCH_SIZE,)
        assert mask.dtype == torch.bool

    def test_default_params(self):
        params = ClassificationWithConfidenceHead.default_params()
        assert "hidden_sizes" in params


class TestConfidenceLoss:
    def test_loss_computation(self):
        logits = torch.randn(BATCH_SIZE, NUM_CLASSES)
        confidence = torch.sigmoid(torch.randn(BATCH_SIZE, 1))
        targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

        loss_fn = ConfidenceLoss(alpha=0.5)
        loss = loss_fn(logits, confidence, targets)
        assert loss.ndim == 0
        assert loss.item() > 0


class TestGradientFlow:
    """Test that gradients flow through all heads."""

    @pytest.mark.parametrize(
        "head_cls,kwargs",
        [
            (LinearClassifierHead, {}),
            (MLPClassifierHead, {"hidden_sizes": [32]}),
            (AttentionClassifierHead, {"num_heads": 2, "hidden_dim": 32}),
            (ResidualClassifierHead, {"hidden_dim": 32, "num_blocks": 1}),
            (OrdinalRegressionHead, {"hidden_sizes": [32]}),
            (DistributionHead, {"hidden_sizes": [32]}),
            (ClassificationWithConfidenceHead, {"hidden_sizes": [32]}),
        ],
    )
    def test_gradient_flows(self, head_cls, kwargs):
        head = head_cls(input_size=INPUT_SIZE, num_classes=NUM_CLASSES, **kwargs)
        x = torch.randn(4, INPUT_SIZE, requires_grad=True)
        out = head(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
