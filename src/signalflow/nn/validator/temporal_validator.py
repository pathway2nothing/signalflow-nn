# src/signalflow/nn/validator/temporal_validator.py
"""
Temporal Validator for SignalFlow integration.
Validates signals using temporal patterns learned from historical data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Any

import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import polars as pl

from signalflow.core import sf_component, Signals
from signalflow.validator import SignalValidator
from signalflow.nn.model.temporal_classificator import TemporalClassificator, TrainingConfig
from signalflow.nn.data import SignalDataModule


@dataclass
@sf_component(name="temporal_validator")
class TemporalValidator(SignalValidator):
    """Temporal signal validator using deep learning.
    
    Wrapper around TemporalClassificator (LightningModule) that integrates
    with SignalFlow's validation pipeline. Supports configurable encoder
    and head architectures via registry.
    
    Attributes:
        encoder_type: Registry name of encoder ('encoder/lstm', 'encoder/gru').
        encoder_params: Parameters for encoder constructor.
        head_type: Registry name of head or None for default linear.
        head_params: Parameters for head constructor.
        window_size: Number of timesteps in input window.
        num_classes: Number of output classes.
        class_weights: Optional class weights for imbalanced data.
        training_config: Training configuration dict.
        feature_cols: List of feature columns (auto-detected if None).
        checkpoint_path: Path to load pretrained model.
        
        batch_size: Batch size for training/inference.
        max_epochs: Maximum training epochs.
        early_stopping_patience: Early stopping patience.
        train_val_test_split: Data split ratios.
        split_strategy: How to split data ('temporal', 'random', 'pair').
        num_workers: DataLoader workers.
    
    Example:
        >>> validator = TemporalValidator(
        ...     encoder_type="encoder/lstm",
        ...     encoder_params={
        ...         "input_size": 20,
        ...         "hidden_size": 64,
        ...         "num_layers": 2,
        ...     },
        ...     head_type="head/cls/mlp",
        ...     head_params={"hidden_sizes": [128], "dropout": 0.2},
        ...     window_size=60,
        ...     training_config={"learning_rate": 1e-3},
        ... )
        >>> 
        >>> # Train
        >>> validator.fit(X_train, y_train)
        >>> 
        >>> # Validate signals
        >>> validated = validator.validate_signals(signals, features)
    """
    
    # Architecture config
    encoder_type: str = "encoder/lstm"
    encoder_params: dict[str, Any] = field(default_factory=lambda: {
        "input_size": 10,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
    })
    head_type: Optional[str] = "head/cls/mlp"
    head_params: Optional[dict[str, Any]] = field(default_factory=lambda: {
        "hidden_sizes": [128],
        "dropout": 0.2,
    })
    
    # Model config
    window_size: int = 60
    num_classes: int = 3
    class_weights: Optional[list[float]] = None
    training_config: dict[str, Any] = field(default_factory=dict)
    
    # Feature config
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
    model: Optional[TemporalClassificator] = field(default=None, init=False, repr=False)
    trainer: Optional[L.Trainer] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize validator and optionally load checkpoint."""
        if self.checkpoint_path:
            self._setup_model()
            self.load_checkpoint(self.checkpoint_path)
    
    def _setup_model(self):
        """Initialize the TemporalClassificator with config."""
        # Build TrainingConfig from dict
        tc = TrainingConfig.from_dict(self.training_config) if self.training_config else TrainingConfig()
        
        self.model = TemporalClassificator(
            encoder_type=self.encoder_type,
            encoder_params=self.encoder_params,
            head_type=self.head_type,
            head_params=self.head_params,
            num_classes=self.num_classes,
            class_weights=self.class_weights,
            training_config=tc,
        )
    
    def _infer_input_size(self, features_df: pl.DataFrame) -> int:
        """Infer input size from features DataFrame."""
        exclude_cols = {self.pair_col, self.ts_col, "label", "signal", "signal_type"}
        
        if self.feature_cols is not None:
            return len(self.feature_cols)
        
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        return len(feature_cols)
    
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
        """Train the validator on labeled signals.
        
        Args:
            X_train: Features DataFrame [pair, timestamp, feature_1, ...].
            y_train: Labels DataFrame [pair, timestamp, label].
            X_val: Validation features (optional, uses split from DataModule).
            y_val: Validation labels (optional).
            log_dir: Directory for logs and checkpoints.
            accelerator: Lightning accelerator ('auto', 'cpu', 'gpu').
            devices: Number of devices or device list.
            
        Returns:
            Self for method chaining.
            
        Raises:
            ValueError: If y_train missing required columns.
        """
        # Validate y_train has required columns
        if isinstance(y_train, pl.DataFrame):
            if "pair" not in y_train.columns or "timestamp" not in y_train.columns:
                raise ValueError("y_train must contain 'pair' and 'timestamp' columns")
        
        # Update encoder input_size if not set
        input_size = self._infer_input_size(X_train)
        if "input_size" not in self.encoder_params or self.encoder_params.get("input_size") != input_size:
            self.encoder_params = {**self.encoder_params, "input_size": input_size}
        
        # Setup model
        if self.model is None:
            self._setup_model()
        
        # Create DataModule
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
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.early_stopping_patience,
                mode="min",
            ),
        ]
        
        checkpoint_callback = None
        if log_dir:
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                dirpath=log_dir / "checkpoints",
                filename="best-{epoch:02d}-{val_loss:.4f}",
                save_top_k=1,
                mode="min",
            )
            callbacks.append(checkpoint_callback)
        
        # Trainer
        self.trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            default_root_dir=log_dir,
            enable_progress_bar=True,
        )
        
        # Train
        self.trainer.fit(self.model, datamodule)
        
        # Load best checkpoint if available
        if checkpoint_callback and checkpoint_callback.best_model_path:
            self.model = TemporalClassificator.load_from_checkpoint(
                checkpoint_callback.best_model_path
            )
        
        return self
    
    def validate_signals(
        self,
        signals: Signals,
        features: pl.DataFrame,
        prefix: str = "probability_",
    ) -> Signals:
        """Add validation probabilities to signals.
        
        Runs inference on each signal and adds probability columns.
        
        Args:
            signals: Input Signals container.
            features: Features DataFrame with [pair, timestamp, features...].
            prefix: Prefix for probability columns.
            
        Returns:
            Signals with added probability columns.
            
        Raises:
            ValueError: If model not trained.
        """
        import numpy as np
        from signalflow.nn.data.signal_datamodule import SignalWindowDataset
        from torch.utils.data import DataLoader
        
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        self.model.eval()
        signals_df = signals.value
        
        # Add dummy label for dataset compatibility
        if "label" not in signals_df.columns:
            signals_df = signals_df.with_columns(pl.lit(0).alias("label"))
        
        # Create dataset
        dataset = SignalWindowDataset(
            features_df=features,
            signals_df=signals_df,
            window_size=self.window_size,
            feature_cols=self.feature_cols,
        )
        
        if len(dataset) == 0:
            return signals
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        
        all_probs = []
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(device)
                logits = self.model(batch_x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        if not all_probs:
            return signals
        
        all_probs = np.vstack(all_probs)
        
        # Create probability columns
        # Map class indices to meaningful names if possible
        class_names = {
            0: "none",
            1: "rise", 
            2: "fall",
        }
        
        prob_cols = {}
        for i in range(self.num_classes):
            name = class_names.get(i, f"class_{i}")
            prob_cols[f"{prefix}{name}"] = all_probs[:, i].tolist()
        
        # Add columns to signals
        result_df = signals.value.with_columns([
            pl.Series(name, values)
            for name, values in prob_cols.items()
        ])
        
        return Signals(result_df)
    
    def predict(self, signals: Signals, X: pl.DataFrame) -> Signals:
        """Predict class labels for signals.
        
        Args:
            signals: Input signals.
            X: Features DataFrame.
            
        Returns:
            Signals with 'validation_pred' column.
        """
        validated = self.validate_signals(signals, X, prefix="probability_")
        
        # Get predicted class
        prob_cols = [c for c in validated.value.columns if c.startswith("probability_")]
        
        if not prob_cols:
            return validated
        
        # Find max probability column
        result_df = validated.value.with_columns(
            pl.concat_list(prob_cols)
            .list.arg_max()
            .alias("validation_pred")
        )
        
        return Signals(result_df)
    
    def predict_proba(self, signals: Signals, X: pl.DataFrame) -> Signals:
        """Predict class probabilities for signals.
        
        Alias for validate_signals().
        """
        return self.validate_signals(signals, X)
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint.
        """
        if self.trainer:
            self.trainer.save_checkpoint(path)
        elif self.model:
            torch.save(self.model.state_dict(), path)
    
    def load_checkpoint(self, path: Path):
        """Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file.
        """
        self.model = TemporalClassificator.load_from_checkpoint(path)
    
    def save(self, path: str | Path) -> None:
        """Save validator state.
        
        Args:
            path: Path to save file.
        """
        import pickle
        
        path = Path(path)
        
        state = {
            "encoder_type": self.encoder_type,
            "encoder_params": self.encoder_params,
            "head_type": self.head_type,
            "head_params": self.head_params,
            "window_size": self.window_size,
            "num_classes": self.num_classes,
            "class_weights": self.class_weights,
            "training_config": self.training_config,
            "feature_cols": self.feature_cols,
            "model_state_dict": self.model.state_dict() if self.model else None,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: str | Path) -> "TemporalValidator":
        """Load validator from file.
        
        Args:
            path: Path to saved validator.
            
        Returns:
            Loaded TemporalValidator.
        """
        import pickle
        
        path = Path(path)
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        validator = cls(
            encoder_type=state["encoder_type"],
            encoder_params=state["encoder_params"],
            head_type=state["head_type"],
            head_params=state["head_params"],
            window_size=state["window_size"],
            num_classes=state["num_classes"],
            class_weights=state["class_weights"],
            training_config=state["training_config"],
            feature_cols=state["feature_cols"],
        )
        
        if state["model_state_dict"]:
            validator._setup_model()
            validator.model.load_state_dict(state["model_state_dict"])
        
        return validator