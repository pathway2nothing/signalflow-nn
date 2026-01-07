# FILE: src/signalflow/nn/validator/temporal_validator.py
"""
Temporal Validator for SignalFlow integration.
Validates signals using temporal patterns learned from historical data.
"""
import polars as pl
import torch
import lightning as L
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path

from signalflow.core import sf_component, Signals
from signalflow.validator import SignalValidator
from signalflow.nn.model import TemporalClassificator
from signalflow.nn.data import SignalDataModule
from signalflow.nn.head import MLPClassifierHead


@dataclass
@sf_component(name="temporal_validator")
class TemporalValidator(SignalValidator):
    """Temporal signal validator using deep learning.
    
    Acts as a wrapper around TemporalClassificator (LightningModule).
    Does NOT store encoder/head directly; delegates everything to the model.
    """
    
    encoder_config: dict = field(default_factory=lambda: {"name": "encoder/lstm", "params": {}})
    head_config: Optional[dict] = None
    window_size: int = 60
    num_classes: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    class_weights: Optional[list[float]] = None
    feature_cols: Optional[list[str]] = None
    checkpoint_path: Optional[Path] = None
    
    # Training params
    batch_size: int = 32
    max_epochs: int = 50
    early_stopping_patience: int = 10
    
    # DataModule params
    train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15)
    split_strategy: Literal["temporal", "random", "pair"] = "temporal"
    num_workers: int = 4
    
    # Internal state
    model: Optional[TemporalClassificator] = field(default=None, init=False)
    trainer: Optional[L.Trainer] = field(default=None, init=False)
    
    def __post_init__(self):
        super().__post_init__()
        if self.checkpoint_path:
             self._setup_model()
             self.load_checkpoint(self.checkpoint_path)

    def _setup_model(self):
        """Initialize the LightningModule with components"""
        # 1. Create components locally
        encoder = self._create_encoder()
        head = self._create_head(input_size=encoder.output_size)
        
        # 2. Inject into Model
        self.model = TemporalClassificator(
            encoder=encoder,
            head=head,
            num_classes=self.num_classes,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            class_weights=self.class_weights,
        )

    def _create_encoder(self) -> nn.Module:
        from signalflow.core import get_component
        
        encoder_name = self.encoder_config.get("name")
        encoder_params = self.encoder_config.get("params", {})
        
        if not encoder_name:
            raise ValueError("encoder_config must contain 'name'")
            
        encoder_cls = get_component(encoder_name)
        return encoder_cls(**encoder_params)
    
    def _create_head(self, input_size: int) -> MLPClassifierHead:
        if self.head_config is None:
            return MLPClassifierHead(
                input_size=input_size,
                num_classes=self.num_classes,
            )
        
        head_params = self.head_config.copy()
        head_params["input_size"] = input_size
        head_params["num_classes"] = self.num_classes
        return MLPClassifierHead(**head_params)
    
    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: pl.DataFrame | pl.Series,
        X_val: Optional[pl.DataFrame] = None,
        y_val: Optional[pl.DataFrame | pl.Series] = None,
        log_dir: Optional[Path] = None,
        accelerator: str = "auto",
        devices: int | list[int] = 1,
    ) -> "TemporalValidator":
        """Train the validator."""
        if self.model is None:
            self._setup_model()

        if "pair" not in y_train.columns or "timestamp" not in y_train.columns:
            raise ValueError("y_train must contain 'pair' and 'timestamp' columns")

        datamodule = SignalDataModule(
            features_df=X_train,
            signals_df=y_train,
            window_size=self.window_size,
            train_val_test_split=self.train_val_test_split,
            split_strategy=self.split_strategy,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            feature_cols=self.feature_cols,
        )
        
        callbacks = [
            L.callbacks.EarlyStopping(monitor="val_loss", patience=self.early_stopping_patience, mode="min"),
            L.callbacks.ModelCheckpoint(
                monitor="val_loss",
                dirpath=log_dir / "checkpoints" if log_dir else None,
                filename="best-{epoch:02d}-{val_loss:.2f}",
                save_top_k=1,
                mode="min",
            ),
        ]
        
        self.trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            default_root_dir=log_dir,
            enable_progress_bar=True,
        )
        
        self.trainer.fit(self.model, datamodule)
        
        if callbacks[1].best_model_path:
            self.model = TemporalClassificator.load_from_checkpoint(callbacks[1].best_model_path)
            
        return self

    def validate_signals(
        self, 
        signals: Signals, 
        features: pl.DataFrame,
        prefix: str = "probability_",
    ) -> Signals:
        """Run inference."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        self.model.eval()
        
        from signalflow.nn.data.signal_datamodule import SignalWindowDataset
        from torch.utils.data import DataLoader
        import numpy as np

        signals_df = signals.value
        
        # Add dummy label for dataset compatibility
        if "label" not in signals_df.columns:
            signals_df = signals_df.with_columns(pl.lit(0).alias("label"))

        # Access input size via the model's encoder
        input_dim = self.feature_cols or self.model.encoder.input_size
        
        dataset = SignalWindowDataset(
            features_df=features,
            signals_df=signals_df,
            window_size=self.window_size,
            feature_cols=input_dim
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        
        all_probs = []
        
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.model.device)
                logits = self.model(batch_x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        if not all_probs:
            return signals

        all_probs = np.vstack(all_probs)
        
        prob_cols = {
            f"{prefix}class_{i}": all_probs[:, i].tolist()
            for i in range(self.num_classes)
        }
        
        result_df = signals_df.with_columns([
            pl.Series(name, values)
            for name, values in prob_cols.items()
        ])
        
        return Signals(result_df)

    def save_checkpoint(self, path: Path):
        if self.trainer:
            self.trainer.save_checkpoint(path)
            
    def load_checkpoint(self, path: Path):
        self.model = TemporalClassificator.load_from_checkpoint(path)