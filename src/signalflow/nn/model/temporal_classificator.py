# src/signalflow/nn/model/temporal_classificator.py
import torch
import torch.nn as nn
import lightning as L
from signalflow.core import SfTorchModuleMixin, sf_component
from signalflow.nn.layer.encoder import TemporalEncoder
from signalflow.nn.head.classifier import MLPClassifierHead
import optuna

@sf_component(name="temporal_classifier")
class TemporalClassificator(L.LightningModule, SfTorchModuleMixin):
    """Temporal signal classifier = Encoder + Head
    
    Architecture:
        Input [batch, seq_len, features] 
        -> Encoder [batch, embedding_size]
        -> Head [batch, num_classes]
    """
    
    def __init__(
        self,
        encoder: TemporalEncoder,
        head: MLPClassifierHead | None = None,
        num_classes: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        class_weights: list[float] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'head'])
        
        self.encoder = encoder
        
        if head is None:
            head = MLPClassifierHead(
                input_size=encoder.output_size,
                num_classes=num_classes,
            )
        
        self.head = head
        
        # Loss function
        if class_weights is not None:
            weight = torch.FloatTensor(class_weights)
        else:
            weight = None
        
        self.loss_fn = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, features]
            
        Returns:
            logits: [batch, num_classes]
        """

        embedding = self.encoder(x)  # [batch, embedding_size]


        logits = self.head(embedding)  # [batch, num_classes]
        
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Metrics
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }
    
    @classmethod
    def default_params(cls) -> dict:
        return {
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "num_classes": 3,
        }
    
    @classmethod
    def tune(cls, trial: optuna.Trial, model_size: str = "small") -> dict:
        return {
            "learning_rate": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "num_classes": 3,
        }