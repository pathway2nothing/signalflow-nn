# signalflow-nn

Neural network extension for [signalflow-trading](https://pypi.org/project/signalflow-trading/) built on PyTorch and Lightning.

## Installation

```bash
pip install signalflow-nn
```

## Requirements

- Python ≥ 3.12
- signalflow-trading ≥ 0.2.1
- PyTorch ≥ 2.2
- Lightning ≥ 2.5

## Modules

- **`data`** – Data loading and preprocessing utilities
- **`encoder`** – Feature encoding layers
- **`head`** – Output head architectures
- **`layer`** – Custom neural network layers
- **`loss`** – Loss functions
- **`model`** – Complete model implementations
- **`validator`** – Model validation utilities

## Usage

```python
import signalflow.nn as nn

# Access submodules
nn.data
nn.encoder
nn.head
nn.model
nn.loss
nn.validator
```

## License

See [signalflow-trading](https://pypi.org/project/signalflow-trading/) for license details.
