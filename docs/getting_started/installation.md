# Installation

## Requirements

- Python 3.10 or higher
- PyTorch 2.0 or higher

## Install from PyPI

```bash
pip install deep-inference
```

## Install from Source

```bash
git clone https://github.com/rawatpranjal/deep-inference
cd deep-inference
pip install -e .
```

## Optional Dependencies

Install optional dependencies for additional functionality:

```bash
# Development tools (testing, linting)
pip install deep-inference[dev]

# Documentation building
pip install deep-inference[docs]

# Plotting (matplotlib, seaborn)
pip install deep-inference[plotting]

# All optional dependencies
pip install deep-inference[all]
```

## Verify Installation

```python
from deep_inference import structural_dml
print("deep-inference installed successfully!")
```

## Dependencies

Core dependencies (installed automatically):

- `torch>=2.0` - Deep learning backend
- `numpy>=1.24` - Numerical computing
- `pandas>=2.0` - Data manipulation
- `scipy>=1.10` - Scientific computing
- `scikit-learn>=1.3` - Machine learning utilities
- `formulaic>=1.0` - Formula parsing
- `tabulate>=0.9` - Table formatting
- `tqdm>=4.65` - Progress bars
