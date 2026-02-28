<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../logo-dark.svg" width="120">
  <source media="(prefers-color-scheme: light)" srcset="../logo.svg" width="120">
  <img alt="SignalFlow" src="../logo.png" width="120">
</picture>

# signalflow-nn

**Neural network extension for SignalFlow — 14 encoders, 7 heads, 4 loss functions**

<p>
<a href="https://pypi.org/project/signalflow-nn/"><img src="https://img.shields.io/badge/version-0.6.0-7c3aed" alt="Version"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-3b82f6?logo=python&logoColor=white" alt="Python 3.12+"></a>
<img src="https://img.shields.io/badge/pytorch-ef4444?logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/lightning-792ee5?logo=lightning&logoColor=white" alt="Lightning">
</p>

</div>

---

Part of the [SignalFlow](https://github.com/pathway2nothing/sf-project) ecosystem.

PyTorch/Lightning library for financial time series classification. Provides modular encoders, classification heads, and loss functions designed for trading signal validation and prediction.

## Installation

```bash
pip install signalflow-nn
```

**Requires:** Python ≥ 3.12, signalflow-trading ≥ 0.5.0, PyTorch ≥ 2.2, Lightning ≥ 2.5

## Quick Start

```python
from signalflow.nn.encoder import TransformerEncoder
from signalflow.nn.head import MLPClassifierHead
from signalflow.nn.model import TemporalClassificator
from signalflow.nn.data import SignalDataModule
import lightning as pl

# Create model
model = TemporalClassificator(
    encoder_type="encoder/transformer",
    encoder_params={"d_model": 64, "nhead": 4, "num_layers": 2},
    head_type="head/cls/mlp",
    head_params={"hidden_sizes": [32]},
    num_classes=3,  # fall, neutral, rise
)

# Create data module
dm = SignalDataModule(
    data=df,
    window_size=60,
    batch_size=32,
    split_strategy="temporal",
)

# Train
trainer = pl.Trainer(max_epochs=50, accelerator="auto")
trainer.fit(model, dm)
```

## Encoders (14)

| Encoder | Architecture | Best For |
|---------|-------------|----------|
| `LSTMEncoder` | Bidirectional LSTM | Sequential patterns |
| `GRUEncoder` | Gated Recurrent Unit | Faster training |
| `TCNEncoder` | Temporal Convolutional Network | Long-range dependencies |
| `TransformerEncoder` | Self-attention + positional encoding | Complex relationships |
| `PatchTSTEncoder` | Patch-based Transformer | Multivariate time series |
| `TSMixerEncoder` | All-MLP (Google 2023) | Efficient mixing |
| `InceptionTimeEncoder` | Multi-scale convolutions | Multi-resolution features |
| `ResNet1dEncoder` | 1D ResNet | Deep representations |
| `XceptionTimeEncoder` | Depthwise separable conv | Efficient computation |
| `Conv1dEncoder` | 1D CNN | Local patterns |
| `XCMEncoder` | Cross-Channel Mixing | Channel interactions |
| `gMLPEncoder` | Gating MLP | Spatial/channel gating |
| `OmniScaleCNNEncoder` | Multi-scale CNN | Scale-invariant features |
| `ConvTranEncoder` | Conv + Transformer hybrid | Combined strengths |

## Classification Heads (7)

| Head | Use Case |
|------|----------|
| `LinearClassifierHead` | Simple baseline |
| `MLPClassifierHead` | Non-linear classification |
| `ResidualClassifierHead` | Deep with skip connections |
| `AttentionClassifierHead` | Attention-weighted pooling |
| `OrdinalRegressionHead` | Ordered classes (fall < neutral < rise) |
| `DistributionHead` | Probability distributions |
| `ClassificationWithConfidenceHead` | Class + confidence score |

## Loss Functions (4)

| Loss | Purpose |
|------|---------|
| `FocalLoss` | Class imbalance — down-weights easy examples |
| `DiceLoss` | Imbalanced multi-class |
| `LDAMLoss` | Large margin for rare classes |
| `SymmetricCrossEntropyLoss` | Noisy labels |

## SignalFlow Integration

Use as a validator in the SignalFlow pipeline:

```python
import signalflow as sf

result = (
    sf.Backtest("nn_validated")
    .data(raw=raw)
    .detector("sma_cross", fast_period=20, slow_period=50)
    .validator("nn/transformer", d_model=64, nhead=4)
    .entry(size_pct=0.1)
    .exit(tp=0.03, sl=0.015)
    .run()
)
```

## Package Structure

| Module | Description |
|--------|-------------|
| `signalflow.nn.data` | Data loading, windowing, temporal splitting |
| `signalflow.nn.encoder` | 14 feature encoding architectures |
| `signalflow.nn.head` | 7 output head architectures |
| `signalflow.nn.layer` | Custom neural network layers |
| `signalflow.nn.loss` | 4 specialized loss functions |
| `signalflow.nn.model` | `TemporalClassificator` — complete model |
| `signalflow.nn.validator` | SignalFlow validator integration |

---

**License:** MIT &ensp;·&ensp; Part of [SignalFlow](https://github.com/pathway2nothing/sf-project)
