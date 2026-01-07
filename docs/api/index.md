# API Reference

Complete API documentation for `deepstats`.

```{toctree}
:maxdepth: 2

dgp
families
models
inference
metrics
```

## Quick Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `get_dgp(name)` | Get data generating process by name |
| `get_family(name)` | Get statistical family by name |
| `influence(X, T, Y, family, config)` | Run influence function inference |
| `naive(X, T, Y, family, config)` | Run naive inference |
| `bootstrap(X, T, Y, family, config)` | Run bootstrap inference |

### Available DGPs

```python
from deepstats import DGPS
print(DGPS)  # ['linear', 'gamma', 'gumbel', 'poisson', 'logit', 'tobit', 'negbin', 'weibull', ...]
```

### Available Families

```python
from deepstats import FAMILIES
print(FAMILIES)  # ['linear', 'gamma', 'gumbel', 'poisson', 'logit', 'tobit', 'negbin', 'weibull', ...]
```

## Module Overview

### dgp

Data generating processes for simulation studies. Each DGP produces synthetic data with known ground truth.

### families

Statistical families defining loss functions, residuals, Hessian weights, and influence scores.

### models

Neural network architectures: `StructuralNet` for parameter estimation, `NuisanceNet` for nuisance functions.

### inference

Inference methods: `naive`, `influence`, and `bootstrap`.

### metrics

Monte Carlo metrics computation and reporting.
