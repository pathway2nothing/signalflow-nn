"""Tests for XceptionTime, XCM, gMLP, OmniScaleCNN, and ConvTran encoders."""

import pytest
import torch

from signalflow.nn.encoder.convtran import ConvTranEncoder
from signalflow.nn.encoder.gmlp import gMLPEncoder
from signalflow.nn.encoder.omniscale import OmniScaleCNNEncoder
from signalflow.nn.encoder.xception import XceptionTimeEncoder
from signalflow.nn.encoder.xcm import XCMEncoder


class TestXceptionTimeEncoder:
    def test_forward_shape(self, encoder_input, num_features):
        encoder = XceptionTimeEncoder(input_size=num_features, num_filters=32, num_blocks=2)
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], encoder.output_size)

    def test_output_size_property(self, num_features):
        encoder = XceptionTimeEncoder(input_size=num_features, num_filters=32, num_blocks=2)
        # num_blocks=2: filters stay 32 for block0 (32*2^0), 64 for block1 (32*2^1)
        # Actually: block0 out=32*2^0=32, block1 out=32*2^1=64
        assert encoder.output_size == 64

    def test_single_block(self, encoder_input, num_features):
        encoder = XceptionTimeEncoder(input_size=num_features, num_filters=64, num_blocks=1)
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], 64)
        assert encoder.output_size == 64

    def test_custom_kernel(self, encoder_input, num_features):
        encoder = XceptionTimeEncoder(input_size=num_features, num_filters=32, kernel_size=15)
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], encoder.output_size)

    def test_default_params(self):
        params = XceptionTimeEncoder.default_params()
        assert "input_size" in params
        assert "num_filters" in params
        assert "kernel_size" in params

    def test_gradient_flow(self, encoder_input, num_features):
        encoder = XceptionTimeEncoder(input_size=num_features, num_filters=32, num_blocks=2)
        out = encoder(encoder_input)
        loss = out.sum()
        loss.backward()
        for p in encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_variable_seq_len(self, num_features):
        encoder = XceptionTimeEncoder(input_size=num_features, num_filters=32, num_blocks=2)
        for seq_len in [30, 60, 120]:
            x = torch.randn(4, seq_len, num_features)
            out = encoder(x)
            assert out.shape == (4, encoder.output_size)


class TestXCMEncoder:
    def test_forward_shape(self, encoder_input, num_features):
        seq_len = encoder_input.shape[1]
        encoder = XCMEncoder(input_size=num_features, seq_len=seq_len, d_model=64)
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], 64)

    def test_output_size_property(self, num_features):
        encoder = XCMEncoder(input_size=num_features, seq_len=60, d_model=128)
        assert encoder.output_size == 128

    def test_custom_filters(self, encoder_input, num_features):
        seq_len = encoder_input.shape[1]
        encoder = XCMEncoder(
            input_size=num_features,
            seq_len=seq_len,
            num_filters_time=32,
            num_filters_space=48,
            d_model=64,
        )
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], 64)

    def test_default_params(self):
        params = XCMEncoder.default_params()
        assert "input_size" in params
        assert "seq_len" in params
        assert "d_model" in params

    def test_gradient_flow(self, encoder_input, num_features):
        seq_len = encoder_input.shape[1]
        encoder = XCMEncoder(input_size=num_features, seq_len=seq_len, d_model=64)
        out = encoder(encoder_input)
        loss = out.sum()
        loss.backward()
        for p in encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_dual_branch_output(self, encoder_input, num_features):
        """Verify both temporal and spatial branches contribute."""
        seq_len = encoder_input.shape[1]
        encoder = XCMEncoder(
            input_size=num_features,
            seq_len=seq_len,
            num_filters_time=32,
            num_filters_space=32,
            d_model=64,
        )
        encoder.eval()
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], 64)
        # Ensure output is not all zeros
        assert out.abs().sum() > 0


class TestgMLPEncoder:
    def test_forward_shape(self, encoder_input, num_features):
        seq_len = encoder_input.shape[1]
        encoder = gMLPEncoder(input_size=num_features, seq_len=seq_len, d_model=64)
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], 64)

    def test_output_size_property(self, num_features):
        encoder = gMLPEncoder(input_size=num_features, seq_len=60, d_model=128)
        assert encoder.output_size == 128

    def test_multi_layer(self, encoder_input, num_features):
        seq_len = encoder_input.shape[1]
        encoder = gMLPEncoder(
            input_size=num_features,
            seq_len=seq_len,
            d_model=64,
            d_ffn=128,
            num_layers=6,
        )
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], 64)

    def test_default_params(self):
        params = gMLPEncoder.default_params()
        assert "input_size" in params
        assert "seq_len" in params
        assert "d_ffn" in params

    def test_gradient_flow(self, encoder_input, num_features):
        seq_len = encoder_input.shape[1]
        encoder = gMLPEncoder(input_size=num_features, seq_len=seq_len, d_model=64)
        out = encoder(encoder_input)
        loss = out.sum()
        loss.backward()
        for p in encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_spatial_gating(self, num_features):
        """Verify SGU gating mechanism works for different seq lengths."""
        for seq_len in [30, 60]:
            encoder = gMLPEncoder(input_size=num_features, seq_len=seq_len, d_model=64)
            x = torch.randn(4, seq_len, num_features)
            out = encoder(x)
            assert out.shape == (4, 64)


class TestOmniScaleCNNEncoder:
    def test_forward_shape(self, encoder_input, num_features):
        encoder = OmniScaleCNNEncoder(input_size=num_features, num_filters=32)
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], 32)

    def test_output_size_property(self, num_features):
        encoder = OmniScaleCNNEncoder(input_size=num_features, num_filters=64)
        assert encoder.output_size == 64

    def test_custom_receptive_fields(self, encoder_input, num_features):
        encoder = OmniScaleCNNEncoder(
            input_size=num_features,
            num_filters=32,
            receptive_field_sizes=[3, 7, 15],
        )
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], 32)

    def test_single_block(self, encoder_input, num_features):
        encoder = OmniScaleCNNEncoder(input_size=num_features, num_filters=64, num_blocks=1)
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], 64)

    def test_default_params(self):
        params = OmniScaleCNNEncoder.default_params()
        assert "input_size" in params
        assert "num_filters" in params
        assert "receptive_field_sizes" in params

    def test_gradient_flow(self, encoder_input, num_features):
        encoder = OmniScaleCNNEncoder(input_size=num_features, num_filters=32)
        out = encoder(encoder_input)
        loss = out.sum()
        loss.backward()
        for p in encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_variable_seq_len(self, num_features):
        encoder = OmniScaleCNNEncoder(input_size=num_features, num_filters=32)
        for seq_len in [30, 60, 120]:
            x = torch.randn(4, seq_len, num_features)
            out = encoder(x)
            assert out.shape == (4, 32)


class TestConvTranEncoder:
    def test_forward_shape(self, encoder_input, num_features):
        encoder = ConvTranEncoder(input_size=num_features, d_model=64)
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], 64)

    def test_output_size_property(self, num_features):
        encoder = ConvTranEncoder(input_size=num_features, d_model=128)
        assert encoder.output_size == 128

    def test_multi_layer(self, encoder_input, num_features):
        encoder = ConvTranEncoder(
            input_size=num_features,
            d_model=64,
            num_layers=4,
            nhead=4,
            dim_feedforward=128,
        )
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], 64)

    def test_custom_conv_kernel(self, encoder_input, num_features):
        encoder = ConvTranEncoder(
            input_size=num_features,
            d_model=64,
            conv_kernel_size=11,
        )
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], 64)

    def test_default_params(self):
        params = ConvTranEncoder.default_params()
        assert "input_size" in params
        assert "d_model" in params
        assert "conv_kernel_size" in params

    def test_gradient_flow(self, encoder_input, num_features):
        encoder = ConvTranEncoder(input_size=num_features, d_model=64)
        out = encoder(encoder_input)
        loss = out.sum()
        loss.backward()
        for p in encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_variable_seq_len(self, num_features):
        encoder = ConvTranEncoder(input_size=num_features, d_model=64)
        for seq_len in [30, 60, 120]:
            x = torch.randn(4, seq_len, num_features)
            out = encoder(x)
            assert out.shape == (4, 64)


class TestNewEncoderConsistency:
    """Test that all new encoders follow the unified contract."""

    @pytest.mark.parametrize(
        "encoder_cls,kwargs",
        [
            (XceptionTimeEncoder, {"num_filters": 32, "num_blocks": 2}),
            (XCMEncoder, {"seq_len": 60, "d_model": 64}),
            (gMLPEncoder, {"seq_len": 60, "d_model": 64}),
            (OmniScaleCNNEncoder, {"num_filters": 32}),
            (ConvTranEncoder, {"d_model": 64}),
        ],
    )
    def test_contract(self, encoder_input, num_features, encoder_cls, kwargs):
        encoder = encoder_cls(input_size=num_features, **kwargs)

        # output_size property exists
        assert isinstance(encoder.output_size, int)
        assert encoder.output_size > 0

        # forward returns correct shape
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], encoder.output_size)
        assert out.ndim == 2

        # default_params returns dict with input_size
        params = encoder_cls.default_params()
        assert isinstance(params, dict)
        assert "input_size" in params

    @pytest.mark.parametrize(
        "encoder_cls,kwargs",
        [
            (XceptionTimeEncoder, {"num_filters": 32, "num_blocks": 2}),
            (XCMEncoder, {"seq_len": 60, "d_model": 64}),
            (gMLPEncoder, {"seq_len": 60, "d_model": 64}),
            (OmniScaleCNNEncoder, {"num_filters": 32}),
            (ConvTranEncoder, {"d_model": 64}),
        ],
    )
    def test_backward_pass(self, encoder_input, num_features, encoder_cls, kwargs):
        encoder = encoder_cls(input_size=num_features, **kwargs)
        out = encoder(encoder_input)
        loss = out.sum()
        loss.backward()
        grad_count = sum(1 for p in encoder.parameters() if p.requires_grad and p.grad is not None)
        assert grad_count > 0
