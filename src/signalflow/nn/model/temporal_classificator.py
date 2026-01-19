# src/signalflow/nn/model/temporal_classificator.py
"""Temporal signal classifier with configurable encoder and head."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
import lightning as L
import optuna

from signalflow import SfTorchModuleMixin, sf_component, SfComponentType, default_registry


@dataclass
class TrainingConfig:
    """Training configuration for TemporalClassificator.
    
    Separates training hyperparameters from model architecture.
    
    Attributes:
        learning_rate: Initial learning rate for optimizer.
        weight_decay: L2 regularization coefficient.
        optimizer: Optimizer type ('adamw', 'adam', 'sgd').
        scheduler: LR scheduler type ('reduce_on_plateau', 'cosine', 'none').
        scheduler_patience: Patience for ReduceLROnPlateau.
        scheduler_factor: Factor for ReduceLROnPlateau.
        warmup_steps: Number of warmup steps for cosine scheduler.
        label_smoothing: Label smoothing factor for CrossEntropyLoss.
        gradient_clip_val: Gradient clipping value (None to disable).
    
    Example:
        >>> config = TrainingConfig(
        ...     learning_rate=1e-3,
        ...     weight_decay=1e-5,
        ...     scheduler='cosine',
        ...     warmup_steps=100
        ... )
    """
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    
    # Scheduler
    scheduler: Literal["reduce_on_plateau", "cosine", "none"] = "reduce_on_plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    warmup_steps: int = 0
    
    # Loss
    label_smoothing: float = 0.0
    
    # Regularization
    gradient_clip_val: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for save_hyperparameters."""
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "scheduler_patience": self.scheduler_patience,
            "scheduler_factor": self.scheduler_factor,
            "warmup_steps": self.warmup_steps,
            "label_smoothing": self.label_smoothing,
            "gradient_clip_val": self.gradient_clip_val,
        }
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@sf_component(name="temporal_classifier")
class TemporalClassificator(L.LightningModule, SfTorchModuleMixin):
    """Temporal signal classifier: Encoder + Classification Head.
    
    Configurable architecture that loads encoder and head from SignalFlow registry.
    Supports multiple encoder types (LSTM, GRU, Transformer) and head types.
    
    Architecture:
        Input [batch, seq_len, features] 
        -> Encoder [batch, embedding_size]
        -> Head [batch, num_classes]
    
    Args:
        encoder_type: Registry name of encoder (e.g., 'encoder/lstm', 'encoder/gru').
        encoder_params: Parameters for encoder constructor.
        head_type: Registry name of head (e.g., 'head/cls/mlp'). If None, uses default MLP.
        head_params: Parameters for head constructor.
        num_classes: Number of output classes.
        class_weights: Optional class weights for imbalanced data.
        training_config: Training configuration (optimizer, scheduler, etc.).
    
    Example:
        >>> # Using config dictionaries
        >>> model = TemporalClassificator(
        ...     encoder_type="encoder/lstm",
        ...     encoder_params={
        ...         "input_size": 20,
        ...         "hidden_size": 64,
        ...         "num_layers": 2,
        ...         "dropout": 0.1,
        ...     },
        ...     head_type="head/cls/mlp",
        ...     head_params={
        ...         "hidden_sizes": [128],
        ...         "dropout": 0.2,
        ...     },
        ...     num_classes=3,
        ...     training_config=TrainingConfig(learning_rate=1e-3),
        ... )
        >>> 
        >>> # Forward pass
        >>> x = torch.randn(32, 60, 20)  # batch, seq_len, features
        >>> logits = model(x)  # [32, 3]
    
    Note:
        Encoder must have `output_size` property.
        Head receives `input_size` automatically from encoder.output_size.
    """
    
    def __init__(
        self,
        encoder_type: str,
        encoder_params: dict[str, Any],
        head_type: str | None = None,
        head_params: dict[str, Any] | None = None,
        num_classes: int = 3,
        class_weights: list[float] | None = None,
        training_config: TrainingConfig | None = None,
    ):
        super().__init__()
        
        # Use default training config if not provided
        if training_config is None:
            training_config = TrainingConfig()
        
        # Store config for save_hyperparameters
        self._training_config = training_config
        self._encoder_type = encoder_type
        self._encoder_params = encoder_params
        self._head_type = head_type
        self._head_params = head_params or {}
        self._num_classes = num_classes
        self._class_weights = class_weights
        
        # Save hyperparameters (excludes nn.Module objects)
        self.save_hyperparameters({
            "encoder_type": encoder_type,
            "encoder_params": encoder_params,
            "head_type": head_type,
            "head_params": head_params,
            "num_classes": num_classes,
            "class_weights": class_weights,
            **training_config.to_dict(),
        })
        
        # Build encoder from registry
        self.encoder = self._build_encoder(encoder_type, encoder_params)
        
        # Build head from registry (or use default)
        self.head = self._build_head(
            head_type=head_type,
            head_params=head_params,
            input_size=self.encoder.output_size,
            num_classes=num_classes,
        )
        
        # Loss function
        weight = torch.FloatTensor(class_weights) if class_weights else None
        self.loss_fn = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=training_config.label_smoothing,
        )
        
        # Metrics tracking
        self._train_correct = 0
        self._train_total = 0
    
    def _build_encoder(self, encoder_type: str, encoder_params: dict[str, Any]) -> nn.Module:
        """Build encoder from registry.
        
        Args:
            encoder_type: Registry name (e.g., 'encoder/lstm').
            encoder_params: Constructor parameters.
            
        Returns:
            Encoder module with `output_size` property.
            
        Raises:
            KeyError: If encoder_type not found in registry.
            AttributeError: If encoder doesn't have output_size.
        """
        encoder_cls = default_registry.get(SfComponentType.TORCH_MODULE, encoder_type)
        encoder = encoder_cls(**encoder_params)
        
        # Validate encoder has output_size
        if not hasattr(encoder, "output_size"):
            raise AttributeError(
                f"Encoder '{encoder_type}' must have 'output_size' property. "
                f"Got: {type(encoder).__name__}"
            )
        
        return encoder
    
    def _build_head(
        self,
        head_type: str | None,
        head_params: dict[str, Any] | None,
        input_size: int,
        num_classes: int,
    ) -> nn.Module:
        """Build classification head from registry or use default.
        
        Args:
            head_type: Registry name or None for default MLP.
            head_params: Constructor parameters (without input_size/num_classes).
            input_size: Input size from encoder.
            num_classes: Number of output classes.
            
        Returns:
            Head module.
        """
        head_params = head_params or {}
        
        # Always inject input_size and num_classes
        head_params = {
            **head_params,
            "input_size": input_size,
            "num_classes": num_classes,
        }
        
        if head_type is None:
            # Default: simple linear head
            return nn.Linear(input_size, num_classes)
        
        head_cls = default_registry.get(SfComponentType.TORCH_MODULE, head_type)
        return head_cls(**head_params)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and head.
        
        Args:
            x: Input tensor [batch, seq_len, features].
            
        Returns:
            Logits tensor [batch, num_classes].
        """
        embedding = self.encoder(x)  # [batch, embedding_size]
        logits = self.head(embedding)  # [batch, num_classes]
        return logits
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Metrics
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        
        return loss
    
    def predict_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """Prediction step - returns probabilities."""
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        logits = self(x)
        return torch.softmax(logits, dim=1)
    
    def configure_optimizers(self) -> dict:
        """Configure optimizer and scheduler based on TrainingConfig."""
        config = self._training_config
        
        # Optimizer
        if config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
        
        # Scheduler
        if config.scheduler == "none":
            return {"optimizer": optimizer}
        
        if config.scheduler == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=config.scheduler_factor,
                patience=config.scheduler_patience,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                },
            }
        
        if config.scheduler == "cosine":
            # Estimate total steps (will be overridden by trainer if provided)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        
        raise ValueError(f"Unknown scheduler: {config.scheduler}")
    
    @classmethod
    def default_params(cls) -> dict[str, Any]:
        """Default parameters for quick instantiation.
        
        Returns:
            Dictionary with default configuration.
        """
        return {
            "encoder_type": "encoder/lstm",
            "encoder_params": {
                "input_size": 10,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.1,
                "bidirectional": False,
            },
            "head_type": "head/cls/mlp",
            "head_params": {
                "hidden_sizes": [128],
                "dropout": 0.2,
                "activation": "relu",
            },
            "num_classes": 3,
            "class_weights": None,
            "training_config": TrainingConfig(),
        }
    
    @classmethod
    def tune(
        cls,
        trial: optuna.Trial,
        model_size: Literal["small", "medium", "large"] = "small",
        encoder_type: str = "encoder/lstm",
        input_size: int = 10,
        num_classes: int = 3,
    ) -> dict[str, Any]:
        """Optuna hyperparameter search space.
        
        Creates a complete configuration for hyperparameter tuning.
        Delegates to encoder and head tune() methods.
        
        Args:
            trial: Optuna trial object.
            model_size: Size variant ('small', 'medium', 'large').
            encoder_type: Which encoder to tune.
            input_size: Number of input features.
            num_classes: Number of output classes.
            
        Returns:
            Dictionary with full model configuration.
            
        Example:
            >>> import optuna
            >>> 
            >>> def objective(trial):
            ...     params = TemporalClassificator.tune(
            ...         trial, 
            ...         model_size="medium",
            ...         input_size=20,
            ...     )
            ...     model = TemporalClassificator(**params)
            ...     # ... train and evaluate ...
            ...     return val_loss
            >>> 
            >>> study = optuna.create_study(direction="minimize")
            >>> study.optimize(objective, n_trials=50)
        """
        # Get encoder class and tune
        encoder_cls = default_registry.get(SfComponentType.TORCH_MODULE, encoder_type)
        encoder_params = encoder_cls.tune(trial, model_size=model_size)
        encoder_params["input_size"] = input_size
        
        # Get head type and tune
        head_type = trial.suggest_categorical(
            "head_type", 
            ["head/cls/mlp", None]
        )
        
        if head_type is not None:
            head_cls = default_registry.get(SfComponentType.TORCH_MODULE, head_type)
            head_params = head_cls.tune(trial, model_size=model_size)
        else:
            head_params = None
        
        # Training config
        training_config = TrainingConfig(
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            optimizer=trial.suggest_categorical("optimizer", ["adamw", "adam"]),
            scheduler=trial.suggest_categorical("scheduler", ["reduce_on_plateau", "cosine"]),
            label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.2),
        )
        
        return {
            "encoder_type": encoder_type,
            "encoder_params": encoder_params,
            "head_type": head_type,
            "head_params": head_params,
            "num_classes": num_classes,
            "training_config": training_config,
        }
    
    @classmethod
    def from_config(
        cls,
        encoder_type: str,
        encoder_params: dict[str, Any],
        head_type: str | None = None,
        head_params: dict[str, Any] | None = None,
        num_classes: int = 3,
        class_weights: list[float] | None = None,
        training_config: dict[str, Any] | None = None,
    ) -> "TemporalClassificator":
        """Create model from config dictionaries.
        
        Convenience factory method that accepts TrainingConfig as dict.
        
        Args:
            encoder_type: Registry name of encoder.
            encoder_params: Encoder parameters.
            head_type: Registry name of head (optional).
            head_params: Head parameters (optional).
            num_classes: Number of classes.
            class_weights: Class weights (optional).
            training_config: Training config as dict (optional).
            
        Returns:
            Configured TemporalClassificator instance.
            
        Example:
            >>> model = TemporalClassificator.from_config(
            ...     encoder_type="encoder/gru",
            ...     encoder_params={"input_size": 20, "hidden_size": 128},
            ...     training_config={"learning_rate": 5e-4},
            ... )
        """
        tc = None
        if training_config is not None:
            tc = TrainingConfig.from_dict(training_config)
        
        return cls(
            encoder_type=encoder_type,
            encoder_params=encoder_params,
            head_type=head_type,
            head_params=head_params,
            num_classes=num_classes,
            class_weights=class_weights,
            training_config=tc,
        )