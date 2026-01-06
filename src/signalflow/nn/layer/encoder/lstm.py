# src/signalflow/nn/layer/lstm_encoder.py
import torch
import torch.nn as nn
from signalflow.nn.layer.encoder import TemporalEncoder
from signalflow.core import sf_component
import optuna

@sf_component(name="lstm_encoder")
class LSTMEncoder(TemporalEncoder):
    """LSTM-based temporal encoder"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self._output_size = hidden_size * (2 if bidirectional else 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_size]
            
        Returns:
            embedding: [batch, hidden_size * (2 if bidirectional else 1)]
        """
        # lstm_out: [batch, seq_len, hidden_size * num_directions]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last timestep
        # If bidirectional, concat forward and backward
        if self.bidirectional:
            # h_n: [num_layers * 2, batch, hidden_size]
            # Take last layer, both directions
            forward_hidden = h_n[-2, :, :]  # [batch, hidden_size]
            backward_hidden = h_n[-1, :, :]  # [batch, hidden_size]
            embedding = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            # h_n: [num_layers, batch, hidden_size]
            embedding = h_n[-1, :, :]  # [batch, hidden_size]
        
        return embedding
    
    @property
    def output_size(self) -> int:
        return self._output_size
    
    @classmethod
    def default_params(cls) -> dict:
        return {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "bidirectional": False,
        }
    
    @classmethod
    def tune(cls, trial: optuna.Trial, model_size: str = "small") -> dict:
        size_config = {
            "small": {
                "hidden": (32, 64),
                "layers": (1, 2)
            },
            "medium": {
                "hidden": (64, 128),
                "layers": (2, 3)
            },
            "large": {
                "hidden": (128, 256),
                "layers": (3, 4)
            }
        }
        
        config = size_config[model_size]
        
        return {
            "hidden_size": trial.suggest_int("hidden_size", *config["hidden"]),
            "num_layers": trial.suggest_int("num_layers", *config["layers"]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "bidirectional": trial.suggest_categorical("bidirectional", [False, True]),
        }