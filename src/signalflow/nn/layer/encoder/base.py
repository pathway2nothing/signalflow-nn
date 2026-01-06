# src/signalflow/nn/layer/encoder.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from signalflow.core import SfTorchModuleMixin
import optuna

class TemporalEncoder(nn.Module, SfTorchModuleMixin, ABC):
    """Base class for temporal encoders
    
    Encoder converts sequence [batch, seq_len, input_size] 
    to embedding [batch, hidden_size]
    """
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_size]
            
        Returns:
            embedding: [batch, hidden_size]
        """
        pass
    
    @property
    @abstractmethod
    def output_size(self) -> int:
        """Size of the output embedding"""
        pass