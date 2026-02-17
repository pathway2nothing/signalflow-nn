"""Tests for LSTM and GRU encoders."""

from signalflow.nn.encoder.gru import GRUEncoder
from signalflow.nn.encoder.lstm import LSTMEncoder


class TestLSTMEncoder:
    def test_forward_shape(self, encoder_input, num_features):
        encoder = LSTMEncoder(input_size=num_features, hidden_size=32, num_layers=1)
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], 32)

    def test_output_size_property(self, num_features):
        encoder = LSTMEncoder(input_size=num_features, hidden_size=64)
        assert encoder.output_size == 64

    def test_bidirectional(self, encoder_input, num_features):
        encoder = LSTMEncoder(input_size=num_features, hidden_size=32, bidirectional=True)
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], 64)  # 32 * 2
        assert encoder.output_size == 64

    def test_multi_layer(self, encoder_input, num_features):
        encoder = LSTMEncoder(input_size=num_features, hidden_size=32, num_layers=3, dropout=0.2)
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], 32)

    def test_single_layer_no_dropout(self, encoder_input, num_features):
        # dropout should be 0 when num_layers=1
        encoder = LSTMEncoder(input_size=num_features, hidden_size=32, num_layers=1, dropout=0.5)
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], 32)

    def test_default_params(self):
        params = LSTMEncoder.default_params()
        assert "input_size" in params
        assert "hidden_size" in params
        assert "num_layers" in params

    def test_gradient_flow(self, encoder_input, num_features):
        encoder = LSTMEncoder(input_size=num_features, hidden_size=32)
        out = encoder(encoder_input)
        loss = out.sum()
        loss.backward()
        for p in encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestGRUEncoder:
    def test_forward_shape(self, encoder_input, num_features):
        encoder = GRUEncoder(input_size=num_features, hidden_size=32, num_layers=1)
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], 32)

    def test_output_size_property(self, num_features):
        encoder = GRUEncoder(input_size=num_features, hidden_size=64)
        assert encoder.output_size == 64

    def test_bidirectional(self, encoder_input, num_features):
        encoder = GRUEncoder(input_size=num_features, hidden_size=32, bidirectional=True)
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], 64)
        assert encoder.output_size == 64

    def test_multi_layer(self, encoder_input, num_features):
        encoder = GRUEncoder(input_size=num_features, hidden_size=32, num_layers=3, dropout=0.2)
        out = encoder(encoder_input)
        assert out.shape == (encoder_input.shape[0], 32)

    def test_default_params(self):
        params = GRUEncoder.default_params()
        assert "input_size" in params
        assert "hidden_size" in params

    def test_gradient_flow(self, encoder_input, num_features):
        encoder = GRUEncoder(input_size=num_features, hidden_size=32)
        out = encoder(encoder_input)
        loss = out.sum()
        loss.backward()
        for p in encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestEncoderConsistency:
    """Test that LSTM and GRU encoders are interchangeable."""

    def test_same_output_shape(self, encoder_input, num_features):
        lstm = LSTMEncoder(input_size=num_features, hidden_size=32)
        gru = GRUEncoder(input_size=num_features, hidden_size=32)
        assert lstm(encoder_input).shape == gru(encoder_input).shape

    def test_same_output_size_property(self, num_features):
        lstm = LSTMEncoder(input_size=num_features, hidden_size=64)
        gru = GRUEncoder(input_size=num_features, hidden_size=64)
        assert lstm.output_size == gru.output_size

    def test_bidirectional_output_size_match(self, num_features):
        lstm = LSTMEncoder(input_size=num_features, hidden_size=32, bidirectional=True)
        gru = GRUEncoder(input_size=num_features, hidden_size=32, bidirectional=True)
        assert lstm.output_size == gru.output_size
