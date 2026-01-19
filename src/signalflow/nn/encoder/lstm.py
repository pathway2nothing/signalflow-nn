"""LSTM encoder for temporal features."""
import torch
import torch.nn as nn
from signalflow import SfTorchModuleMixin, sf_component

@sf_component(name="encoder/lstm")
class LSTMEncoder(nn.Module, SfTorchModuleMixin):
    """LSTM encoder for sequence processing.
    
    Processes temporal sequences and outputs fixed-size embeddings.
    
    Args:
        input_size: Number of input features per timestep
        hidden_size: Size of hidden state
        num_layers: Number of LSTM layers
        dropout: Dropout rate between layers
        bidirectional: Whether to use bidirectional LSTM
    
    Example:
        >>> encoder = LSTMEncoder(input_size=10, hidden_size=64, num_layers=2)
        >>> x = torch.randn(32, 60, 10)  # (batch, seq_len, features)
        >>> out = encoder(x)  # (32, 64) or (32, 128) if bidirectional
    """
        
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        **kwargs,
    ):
        """Initialize LSTM encoder.
        
        Args:
            input_size: Number of input features per timestep
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers (default: 2)
            dropout: Dropout rate between layers (default: 0.1)
            bidirectional: Whether to use bidirectional LSTM (default: False)
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        self._output_size = hidden_size * (2 if bidirectional else 1)
    
    @property
    def output_size(self) -> int:
        """Output embedding size."""
        return self._output_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch, output_size)
        """
        # LSTM output: (batch, seq_len, hidden_size * num_directions)
        # hidden: tuple of (h_n, c_n) where h_n is (num_layers * num_directions, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last hidden state from all layers
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            # h_n shape: (num_layers * 2, batch, hidden_size)
            forward = h_n[-2, :, :]  # Last layer forward
            backward = h_n[-1, :, :]  # Last layer backward
            out = torch.cat([forward, backward], dim=1)
        else:
            # h_n shape: (num_layers, batch, hidden_size)
            out = h_n[-1, :, :]  # Last layer
        
        return out
    
    @classmethod
    def default_params(cls) -> dict:
        """Default parameters for LSTM encoder."""
        return {
            "input_size": 10,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "bidirectional": False,
        }
    
    @classmethod
    def tune(cls, trial, model_size: str = "small") -> dict:
        """Optuna hyperparameter search space.
        
        Args:
            trial: Optuna trial object
            model_size: Size variant ('small', 'medium', 'large')
            
        Returns:
            Dictionary of hyperparameters
        """
        size_config = {
            "small": {"hidden": (32, 64), "layers": (1, 2)},
            "medium": {"hidden": (64, 128), "layers": (2, 3)},
            "large": {"hidden": (128, 256), "layers": (3, 4)},
        }
        
        config = size_config[model_size]
        
        return {
            "input_size": 10,  # Fixed, depends on features
            "hidden_size": trial.suggest_int("hidden_size", *config["hidden"]),
            "num_layers": trial.suggest_int("num_layers", *config["layers"]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "bidirectional": trial.suggest_categorical("bidirectional", [False, True]),
        }