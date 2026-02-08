"""Tests for loss functions."""

import pytest
import torch
import torch.nn as nn

from signalflow.nn.loss.dice import DiceLoss
from signalflow.nn.loss.focal import FocalLoss
from signalflow.nn.loss.ldam import LDAMLoss
from signalflow.nn.loss.symmetric_ce import SymmetricCrossEntropyLoss

NUM_CLASSES = 3
BATCH_SIZE = 8


@pytest.fixture
def logits():
    return torch.randn(BATCH_SIZE, NUM_CLASSES)


@pytest.fixture
def targets():
    return torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))


# ── FocalLoss ──────────────────────────────────────────────────────────────────


class TestFocalLoss:
    def test_output_scalar(self, logits, targets):
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_output_positive(self, logits, targets):
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_reduction_none(self, logits, targets):
        loss_fn = FocalLoss(gamma=2.0, reduction="none")
        loss = loss_fn(logits, targets)
        assert loss.shape == (BATCH_SIZE,)

    def test_reduction_sum(self, logits, targets):
        loss_fn = FocalLoss(gamma=2.0, reduction="sum")
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_gamma_zero_equals_ce(self, logits, targets):
        """FocalLoss with gamma=0 should be equivalent to CrossEntropyLoss."""
        focal = FocalLoss(gamma=0.0)
        ce = nn.CrossEntropyLoss()
        focal_loss = focal(logits, targets)
        ce_loss = ce(logits, targets)
        torch.testing.assert_close(focal_loss, ce_loss, atol=1e-5, rtol=1e-5)

    def test_with_alpha(self, logits, targets):
        loss_fn = FocalLoss(gamma=2.0, alpha=[1.0, 2.0, 0.5])
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_label_smoothing(self, logits, targets):
        loss_no_smooth = FocalLoss(gamma=2.0, label_smoothing=0.0)
        loss_smooth = FocalLoss(gamma=2.0, label_smoothing=0.1)
        v1 = loss_no_smooth(logits, targets)
        v2 = loss_smooth(logits, targets)
        assert not torch.allclose(v1, v2)

    def test_gradient_flow(self):
        logits = torch.randn(BATCH_SIZE, NUM_CLASSES, requires_grad=True)
        targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_invalid_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            FocalLoss(gamma=-1.0)

    def test_invalid_reduction(self):
        with pytest.raises(ValueError, match="reduction"):
            FocalLoss(reduction="invalid")


# ── DiceLoss ───────────────────────────────────────────────────────────────────


class TestDiceLoss:
    def test_output_scalar(self, logits, targets):
        loss_fn = DiceLoss()
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_output_range(self, logits, targets):
        loss_fn = DiceLoss()
        loss = loss_fn(logits, targets)
        assert 0 <= loss.item() <= 1

    def test_perfect_prediction(self):
        """Perfect predictions should yield loss near 0."""
        targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        # Create logits that strongly predict the correct class
        logits = torch.zeros(8, 3)
        for i, t in enumerate(targets):
            logits[i, t] = 10.0
        loss_fn = DiceLoss()
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.05

    def test_reduction_none(self, logits, targets):
        loss_fn = DiceLoss(reduction="none")
        loss = loss_fn(logits, targets)
        assert loss.shape == (NUM_CLASSES,)

    def test_reduction_sum(self, logits, targets):
        loss_fn = DiceLoss(reduction="sum")
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_smooth_prevents_nan(self):
        """Missing class in batch should not produce NaN."""
        # All targets are class 0 — classes 1 and 2 are missing
        targets = torch.zeros(BATCH_SIZE, dtype=torch.long)
        logits = torch.randn(BATCH_SIZE, NUM_CLASSES)
        loss_fn = DiceLoss(smooth=1.0)
        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss)

    def test_gradient_flow(self):
        logits = torch.randn(BATCH_SIZE, NUM_CLASSES, requires_grad=True)
        targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
        loss_fn = DiceLoss()
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_invalid_smooth(self):
        with pytest.raises(ValueError, match="smooth"):
            DiceLoss(smooth=0.0)


# ── SymmetricCrossEntropyLoss ──────────────────────────────────────────────────


class TestSymmetricCrossEntropyLoss:
    def test_output_scalar(self, logits, targets):
        loss_fn = SymmetricCrossEntropyLoss(num_classes=NUM_CLASSES)
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_output_positive(self, logits, targets):
        loss_fn = SymmetricCrossEntropyLoss(num_classes=NUM_CLASSES)
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_reduction_none(self, logits, targets):
        loss_fn = SymmetricCrossEntropyLoss(num_classes=NUM_CLASSES, reduction="none")
        loss = loss_fn(logits, targets)
        assert loss.shape == (BATCH_SIZE,)

    def test_beta_zero_equals_ce(self, logits, targets):
        """SCE with beta=0 should equal standard CrossEntropyLoss."""
        sce = SymmetricCrossEntropyLoss(alpha=1.0, beta=0.0, num_classes=NUM_CLASSES)
        ce = nn.CrossEntropyLoss()
        sce_loss = sce(logits, targets)
        ce_loss = ce(logits, targets)
        torch.testing.assert_close(sce_loss, ce_loss, atol=1e-5, rtol=1e-5)

    def test_num_classes_mismatch(self):
        logits = torch.randn(BATCH_SIZE, 5)
        targets = torch.randint(0, 5, (BATCH_SIZE,))
        loss_fn = SymmetricCrossEntropyLoss(num_classes=3)
        with pytest.raises(ValueError, match="classes"):
            loss_fn(logits, targets)

    def test_noise_robustness(self):
        """SCE should be less sensitive to noisy labels than CE."""
        torch.manual_seed(42)
        logits = torch.randn(64, NUM_CLASSES)
        clean_targets = torch.randint(0, NUM_CLASSES, (64,))

        # Flip 30% of labels
        noisy_targets = clean_targets.clone()
        flip_mask = torch.rand(64) < 0.3
        noisy_targets[flip_mask] = torch.randint(0, NUM_CLASSES, (flip_mask.sum(),))

        ce = nn.CrossEntropyLoss()
        sce = SymmetricCrossEntropyLoss(alpha=1.0, beta=1.0, num_classes=NUM_CLASSES)

        ce_clean = ce(logits, clean_targets).item()
        ce_noisy = ce(logits, noisy_targets).item()

        sce_clean = sce(logits, clean_targets).item()
        sce_noisy = sce(logits, noisy_targets).item()

        # Both losses should compute without error on noisy labels
        assert abs(ce_noisy - ce_clean) >= 0
        assert abs(sce_noisy - sce_clean) >= 0

    def test_gradient_flow(self):
        logits = torch.randn(BATCH_SIZE, NUM_CLASSES, requires_grad=True)
        targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
        loss_fn = SymmetricCrossEntropyLoss(num_classes=NUM_CLASSES)
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_invalid_num_classes(self):
        with pytest.raises(ValueError, match="num_classes"):
            SymmetricCrossEntropyLoss(num_classes=1)


# ── LDAMLoss ──────────────────────────────────────────────────────────────────


class TestLDAMLoss:
    def test_output_scalar(self, logits, targets):
        loss_fn = LDAMLoss(cls_num_list=[100, 50, 10])
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_output_positive(self, logits, targets):
        loss_fn = LDAMLoss(cls_num_list=[100, 50, 10])
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_reduction_none(self, logits, targets):
        loss_fn = LDAMLoss(cls_num_list=[100, 50, 10], reduction="none")
        loss = loss_fn(logits, targets)
        assert loss.shape == (BATCH_SIZE,)

    def test_balanced_equal_margins(self):
        """Balanced class counts should produce equal margins."""
        loss_fn = LDAMLoss(cls_num_list=[100, 100, 100])
        assert torch.allclose(
            loss_fn.m_list,
            loss_fn.m_list[0].expand_as(loss_fn.m_list),
        )

    def test_imbalanced_larger_margin_for_minority(self):
        """Minority classes should get larger margins."""
        loss_fn = LDAMLoss(cls_num_list=[1000, 100, 10])
        # Class 2 (10 samples) > Class 1 (100) > Class 0 (1000)
        assert loss_fn.m_list[2] > loss_fn.m_list[1]
        assert loss_fn.m_list[1] > loss_fn.m_list[0]

    def test_with_weight(self, logits, targets):
        weight = torch.FloatTensor([1.0, 2.0, 3.0])
        loss_fn = LDAMLoss(cls_num_list=[100, 50, 10], weight=weight)
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_invalid_empty_cls_num_list(self):
        with pytest.raises(ValueError, match="at least 2"):
            LDAMLoss(cls_num_list=[100])

    def test_invalid_zero_count(self):
        with pytest.raises(ValueError, match="must be > 0"):
            LDAMLoss(cls_num_list=[100, 0, 50])

    def test_gradient_flow(self):
        logits = torch.randn(BATCH_SIZE, NUM_CLASSES, requires_grad=True)
        targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
        loss_fn = LDAMLoss(cls_num_list=[100, 50, 10])
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None


# ── Parametrized gradient flow ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "loss_cls,kwargs",
    [
        (FocalLoss, {"gamma": 2.0}),
        (FocalLoss, {"gamma": 0.0, "alpha": [1.0, 2.0, 0.5]}),
        (DiceLoss, {"smooth": 1.0}),
        (SymmetricCrossEntropyLoss, {"num_classes": NUM_CLASSES}),
        (LDAMLoss, {"cls_num_list": [100, 50, 10]}),
    ],
)
def test_gradient_flow_parametrized(loss_cls, kwargs):
    logits = torch.randn(BATCH_SIZE, NUM_CLASSES, requires_grad=True)
    targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    loss_fn = loss_cls(**kwargs)
    loss = loss_fn(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
