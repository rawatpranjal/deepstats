# Installation

## Requirements

- Python 3.10 or higher
- PyTorch 2.0 or higher

## Install from PyPI

```bash
pip install deepstats
```

## Install from Source

```bash
git clone https://github.com/rawatpranjal/deepstats
cd deepstats
pip install -e .
```

## Optional Dependencies

Install optional dependencies for additional functionality:

```bash
# Development tools (testing, linting)
pip install deepstats[dev]

# Documentation building
pip install deepstats[docs]

# Plotting (matplotlib, seaborn)
pip install deepstats[plotting]

# All optional dependencies
pip install deepstats[all]
```

## Verify Installation

```python
import deepstats
print(deepstats.__version__)
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
