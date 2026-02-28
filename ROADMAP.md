# signalflow-nn Roadmap

## Current State (v0.2.6)

- 14 encoders: LSTM, GRU, Conv1d, TCN, InceptionTime, Transformer, PatchTST, ResNet1D, TSMixer, XceptionTime, XCM, gMLP, OmniScaleCNN, ConvTran
- 7 heads: Linear, MLP, Residual, Attention, Ordinal, Distribution, Confidence
- Data pipeline: TimeSeriesPreprocessor, SignalWindowDataset, SignalDataModule
- Training: TemporalClassificator (Lightning), TemporalValidator
- Custom losses: OrdinalCrossEntropyLoss, ConfidenceLoss

---

## High Priority

### New Architectures

- [ ] **MambaEncoder** — State Space Model (Gu & Dao, 2023). Linear O(T) complexity vs O(T^2) transformers. Ideal for long sequences. Most SSM work is on forecasting — gap in classification
- [ ] **iTransformerEncoder** — Inverted Transformer (ICLR 2024 Spotlight). Variates as tokens instead of timesteps. Captures cross-asset correlations in multivariate data
- [ ] **TimesNetEncoder** — 2D periodicity-aware CNN (ICLR 2023). Transforms 1D series to 2D by detected periods, applies 2D conv. Natural fit for intraday/weekly/monthly cycles

### Data Augmentation Module

- [ ] `TimeSeriesAugmentor` class with composable transforms
  - [ ] Jittering (Gaussian noise)
  - [ ] Scaling (random amplitude)
  - [ ] Magnitude warping (smooth random curves)
  - [ ] Time warping (temporal distortion)
  - [ ] Window slicing / cropping
  - [ ] Mixup / CutMix for time series
  - [ ] FrAug — frequency-domain augmentation (spectral perturbation)
- [ ] Integration with SignalDataModule (on-the-fly augmentation in DataLoader)

### Loss Functions

- [ ] Populate `loss/` module with reusable losses
  - [ ] FocalLoss — class imbalance (Lin et al. 2017)
  - [ ] LabelSmoothingCrossEntropy
  - [ ] MixupLoss — for augmentation pipeline
  - [ ] Move OrdinalCrossEntropyLoss and ConfidenceLoss to `loss/` module

---

## Medium Priority

### Self-Supervised Pre-training

- [ ] Contrastive learning framework for encoder pre-training
  - [ ] TS2Vec — hierarchical contrastive loss over augmented views
  - [ ] SoftCLT (ICLR 2024) — soft positive/negative assignments (plug-in improvement)
- [ ] Pre-train on unlabeled market data, fine-tune with TemporalClassificator
- [ ] Integration: `pretrain()` method or separate `ContrastiveTrainer`

### Training Pipeline Improvements

- [ ] Additional optimizers: LION, Sophia
- [ ] Additional schedulers: OneCycleLR, WarmupCosineAnnealing
- [ ] Ensemble wrapper — combine multiple encoder+head pairs, aggregate predictions
- [ ] Gradient accumulation config in TrainingConfig

### Interpretability

- [ ] Attention map extraction utility for Transformer/PatchTST/ConvTran encoders
- [ ] Grad-CAM for CNN encoders (Conv1d, InceptionTime, ResNet1D)
- [ ] KAN-based classification head (Kolmogorov-Arnold Networks) — interpretable spline activations

---

## Low Priority (Research)

### Foundation Model Integration

- [ ] MOMENT encoder wrapper (CMU, ICML 2024) — pre-trained on Time-series Pile, zero-shot/few-shot classification via HuggingFace
- [ ] Chronos-2 embedding extractor (Amazon, 2025) — `embed()` API for downstream classification

### Advanced Techniques

- [ ] Diffusion-based augmentation (Diffusion-TS, ICLR 2024) for minority class oversampling
- [ ] Concept Bottleneck integration — define financial concepts (trend, volatility, volume profile), train with interpretable constraints
- [ ] Mamba-2 / SSD — State Space Duality, adaptive recurrence/attention switching

### Infrastructure

- [ ] mypy type checking configuration
- [ ] Model compression: quantization, pruning for production
- [ ] Benchmarking utilities for UCR/UEA archive comparison
- [ ] MLflow / W&B logging integration in TemporalValidator

---

## Tutorials (ipynb)

- [ ] `notebooks/01_quickstart.ipynb` — basic pipeline: data -> model -> predict
- [ ] `notebooks/02_custom_architecture.ipynb` — encoder/head selection, registry pattern
- [ ] `notebooks/03_hyperparameter_tuning.ipynb` — Optuna integration
- [ ] `notebooks/04_advanced_heads.ipynb` — ordinal, confidence, distribution heads

---

## References

- Mamba: Gu & Dao (2023), arXiv:2312.00752
- iTransformer: Liu et al. (ICLR 2024), github.com/thuml/iTransformer
- TimesNet: Wu et al. (ICLR 2023), arXiv:2210.02186
- SoftCLT: Lee et al. (ICLR 2024), github.com/seunghan96/softclt
- TS2Vec: Yue et al. (AAAI 2022), github.com/yuezhihan/ts2vec
- MOMENT: Goswami et al. (ICML 2024), github.com/moment-timeseries-foundation-model/moment
- Chronos-2: Amazon Science (2025), amazon-science/chronos-forecasting
- FocalLoss: Lin et al. (2017), arXiv:1708.02002
- KAN: Liu et al. (2024), arXiv:2408.07314
- FrAug: frequency-domain augmentation for time series
- Diffusion-TS: ICLR 2024, github.com/Y-debug-sys/Diffusion-TS
