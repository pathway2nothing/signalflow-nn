# signalflow.nn/heads/confidence_head.py
"""Classification head with confidence/uncertainty estimation."""

from typing import Literal

import optuna
import torch
import torch.nn as nn

from signalflow import sf_component, SfTorchModuleMixin


@sf_component(name="head/multi/cls_conf")
class ClassificationWithConfidenceHead(nn.Module, SfTorchModuleMixin):
    """Classification head with confidence estimation.
    
    Outputs both class logits and a confidence score.
    Useful for selective prediction (reject low-confidence signals).
    
    The confidence branch learns to predict when the model is uncertain.
    This allows filtering unreliable predictions at inference time.
    
    Args:
        input_size: Size of encoder output.
        num_classes: Number of output classes.
        hidden_sizes: Hidden layer dimensions for shared layers. Default: [128].
        dropout: Dropout probability. Default: 0.2.
        
    Example:
        >>> head = ClassificationWithConfidenceHead(
        ...     input_size=256, 
        ...     num_classes=3,
        ... )
        >>> x = torch.randn(32, 256)
        >>> logits, confidence = head(x)
        >>> # logits: [32, 3], confidence: [32, 1]
        >>> 
        >>> # Filter by confidence threshold
        >>> high_conf_mask = (confidence > 0.7).squeeze()
        >>> reliable_preds = logits[high_conf_mask].argmax(dim=1)
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128]
        
        self.input_size = input_size
        self.num_classes = num_classes

        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_size = hidden_size
        
        self.shared = nn.Sequential(*layers)
        self.shared_dim = current_size
        
        self.classifier = nn.Linear(current_size, num_classes)
        
        self.confidence_head = nn.Sequential(
            nn.Linear(current_size, current_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(current_size // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor [batch, input_size].
            
        Returns:
            Tuple of (logits [batch, num_classes], confidence [batch, 1]).
        """
        features = self.shared(x)
        logits = self.classifier(features)
        confidence = self.confidence_head(features)
        return logits, confidence
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get only logits (for compatibility with standard training)."""
        logits, _ = self.forward(x)
        return logits
    
    def predict_with_confidence(
        self, 
        x: torch.Tensor, 
        threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict classes with confidence filtering.
        
        Args:
            x: Input tensor [batch, input_size].
            threshold: Minimum confidence threshold.
            
        Returns:
            Tuple of (predictions, confidences, mask of reliable predictions).
        """
        logits, confidence = self.forward(x)
        predictions = logits.argmax(dim=1)
        reliable_mask = (confidence >= threshold).squeeze(-1)
        return predictions, confidence.squeeze(-1), reliable_mask
    
    @classmethod
    def default_params(cls) -> dict:
        """Default parameters."""
        return {
            "hidden_sizes": [128],
            "dropout": 0.2,
        }
    
    @classmethod
    def tune(cls, trial: optuna.Trial, model_size: Literal["small", "medium", "large"] = "small") -> dict:
        """Optuna hyperparameter search space."""
        size_config = {
            "small": {"hidden_range": (64, 128), "max_layers": 2},
            "medium": {"hidden_range": (128, 256), "max_layers": 2},
            "large": {"hidden_range": (256, 512), "max_layers": 3},
        }
        
        config = size_config[model_size]
        
        num_layers = trial.suggest_int("head_num_layers", 1, config["max_layers"])
        hidden_sizes = [
            trial.suggest_int(f"head_hidden_{i}", *config["hidden_range"])
            for i in range(num_layers)
        ]
        
        return {
            "hidden_sizes": hidden_sizes,
            "dropout": trial.suggest_float("head_dropout", 0.1, 0.4),
        }


class ConfidenceLoss(nn.Module):
    """Loss function for training classification with confidence.
    
    Combines classification loss with confidence calibration:
    - Classification loss (cross-entropy) on all samples
    - Confidence should be high when prediction is correct
    - Confidence should be low when prediction is wrong
    
    Args:
        alpha: Weight for confidence calibration loss. Default: 0.5.
        
    Example:
        >>> head = ClassificationWithConfidenceHead(input_size=256, num_classes=3)
        >>> loss_fn = ConfidenceLoss(alpha=0.5)
        >>> 
        >>> logits, confidence = head(x)
        >>> loss = loss_fn(logits, confidence, targets)
    """
    
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
    
    def forward(
        self, 
        logits: torch.Tensor, 
        confidence: torch.Tensor, 
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.
        
        Args:
            logits: Predicted logits [batch, num_classes].
            confidence: Predicted confidence [batch, 1].
            targets: Target class indices [batch].
            
        Returns:
            Combined loss value.
        """
        ce = self.ce_loss(logits, targets)
        
        preds = logits.argmax(dim=1)
        correct = (preds == targets).float()
        
        conf = confidence.squeeze(-1)
        conf_loss = nn.functional.binary_cross_entropy(conf, correct)
        
        total_loss = ce.mean() + self.alpha * conf_loss
        
        return total_loss