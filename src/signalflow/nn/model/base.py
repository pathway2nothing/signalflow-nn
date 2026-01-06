from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import lightning as L
from signalflow.core import SfTorchModuleMixin, sf_component
from dataclasses import dataclass

@dataclass
class BaseSignalFlowModel(L.LightningModule, SfTorchModuleMixin, ABC):
    """Base class for all SignalFlow neural models"""
    
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    @abstractmethod
    def forward(self, x):
        """Forward pass - must implement"""
        pass
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    @abstractmethod
    def compute_loss(self, y_hat, y):
        """Compute loss - must implement"""
        pass