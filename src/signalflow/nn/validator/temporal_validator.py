from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Any
import pickle

import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import polars as pl

from signalflow import sf_component, Signals
from signalflow.validator import SignalValidator
from signalflow.nn.model.temporal_classificator import TemporalClassificator, TrainingConfig
from signalflow.nn.data.signal_data_module import SignalDataModule
from signalflow.nn.data.signal_window_dataset import SignalWindowDataset
from signalflow.nn.data.ts_preprocessor import TimeSeriesPreprocessor


@dataclass
@sf_component(name="temporal_validator")
class TemporalValidator(SignalValidator):
    """Temporal signal validator using deep learning."""
    
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
    
    preprocessor: Optional[TimeSeriesPreprocessor] = None
    
    window_size: int = 60
    window_timeframe: int = 1
    num_classes: int = 3
    class_weights: Optional[list[float]] = None
    training_config: dict[str, Any] = field(default_factory=dict)
    
    feature_cols: Optional[list[str]] = None
    checkpoint_path: Optional[Path] = None
    
    batch_size: int = 32
    max_epochs: int = 50
    early_stopping_patience: int = 10
    
    train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15)
    split_strategy: Literal["temporal", "random", "pair"] = "temporal"
    num_workers: int = 4
    
    model: Optional[TemporalClassificator] = field(default=None, init=False, repr=False)
    trainer: Optional[L.Trainer] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        if self.checkpoint_path:
            self._setup_model()
            self.load_checkpoint(self.checkpoint_path)
    
    def _setup_model(self):
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
        if self.feature_cols is not None:
            return len(self.feature_cols)
            
        exclude_cols = {"pair", "timestamp", "label", "signal", "signal_type"}
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
        
        if isinstance(y_train, pl.DataFrame):
            if "pair" not in y_train.columns or "timestamp" not in y_train.columns:
                raise ValueError("y_train must contain 'pair' and 'timestamp' columns")
        
        input_size = self._infer_input_size(X_train)
        if "input_size" not in self.encoder_params or self.encoder_params.get("input_size") != input_size:
            self.encoder_params = {**self.encoder_params, "input_size": input_size}
        
        if self.model is None:
            self._setup_model()
        
        datamodule = SignalDataModule(
            features_df=X_train,
            signals_df=y_train,
            preprocessor=self.preprocessor,
            window_size=self.window_size,
            window_timeframe=self.window_timeframe,
            train_val_test_split=self.train_val_test_split,
            split_strategy=self.split_strategy,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            feature_cols=self.feature_cols,
        )
        
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
        
        self.trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            default_root_dir=log_dir,
            enable_progress_bar=True,
        )
        
        self.trainer.fit(self.model, datamodule)
        
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
        import numpy as np
        from torch.utils.data import DataLoader
        
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        self.model.eval()
        signals_df = signals.value
        n_signals = signals_df.height
        
        if self.preprocessor:
            processed_features = self.preprocessor.transform(features)
        else:
            processed_features = features
        
        if "label" not in signals_df.columns:
            signals_df_with_label = signals_df.with_columns(pl.lit(0).alias("label"))
        else:
            signals_df_with_label = signals_df
        
        dataset = SignalWindowDataset(
            features_df=processed_features,
            signals_df=signals_df_with_label,
            window_size=self.window_size,
            window_timeframe=self.window_timeframe,
            feature_cols=self.feature_cols,
        )
        
        all_probs = np.full((n_signals, self.num_classes), np.nan, dtype=np.float32)
        
        if len(dataset) == 0:
            return self._return_empty_predictions(signals, prefix, all_probs)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        
        valid_probs = []
        device = next(self.model.parameters()).device
        self.model.to(device)
        
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(device)
                logits = self.model(batch_x)
                probs = torch.softmax(logits, dim=1)
                valid_probs.append(probs.cpu().numpy())
        
        if valid_probs:
            valid_probs = np.vstack(valid_probs)
            valid_indices = dataset.valid_signal_indices
            for i, orig_idx in enumerate(valid_indices):
                all_probs[orig_idx] = valid_probs[i]
        
        return self._format_predictions(signals, prefix, all_probs)

    def _format_predictions(self, signals, prefix, all_probs):
        class_names = {0: "none", 1: "rise", 2: "fall"}
        prob_cols = {}
        for i in range(self.num_classes):
            name = class_names.get(i, f"class_{i}")
            prob_cols[f"{prefix}{name}"] = all_probs[:, i].tolist()
        
        result_df = signals.value.with_columns([
            pl.Series(name, values)
            for name, values in prob_cols.items()
        ])
        return Signals(result_df)

    def _return_empty_predictions(self, signals, prefix, all_probs):
        return self._format_predictions(signals, prefix, all_probs)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "version": "0.2.1",
            "encoder_type": self.encoder_type,
            "encoder_params": self.encoder_params,
            "head_type": self.head_type,
            "head_params": self.head_params,
            "window_size": self.window_size,
            "window_timeframe": self.window_timeframe,
            "num_classes": self.num_classes,
            "class_weights": self.class_weights,
            "training_config": self.training_config,
            "feature_cols": self.feature_cols,
            "preprocessor": self.preprocessor, 
            "model_state_dict": self.model.state_dict() if self.model else None,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: str | Path) -> "TemporalValidator":
        path = Path(path)
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        validator = cls(
            encoder_type=state["encoder_type"],
            encoder_params=state["encoder_params"],
            head_type=state["head_type"],
            head_params=state["head_params"],
            window_size=state["window_size"],
            window_timeframe=state.get("window_timeframe", 1),
            num_classes=state["num_classes"],
            class_weights=state["class_weights"],
            training_config=state["training_config"],
            feature_cols=state["feature_cols"],
            preprocessor=state.get("preprocessor"), 
        )
        
        if state["model_state_dict"]:
            validator._setup_model()
            validator.model.load_state_dict(state["model_state_dict"])
        
        return validator