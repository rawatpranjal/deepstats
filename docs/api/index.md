# API Reference

Complete API documentation for `deep-inference`.

```{toctree}
:maxdepth: 2

inference
families
targets
lambda
models
metrics
```

## Quick Reference

### Two APIs

| API | Use Case |
|-----|----------|
| `structural_dml()` | Production, 8 families, fixed target E[β] |
| `inference()` | Flexible targets, regime detection, RCT support |

### Main Entry Points

```python
from deep_inference import structural_dml, inference

# Legacy API (production-ready)
result = structural_dml(
    Y=Y, T=T, X=X,
    family='linear',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50
)

# New API (flexible)
from deep_inference.lambda_.compute import Normal

result = inference(
    Y=Y, T=T, X=X,
    model='logit',
    target='ame',              # Flexible target
    is_randomized=True,        # Regime A
    treatment_dist=Normal(0, 1)
)
```

### Available Families

```python
from deep_inference import FAMILY_REGISTRY
print(list(FAMILY_REGISTRY.keys()))
# ['linear', 'logit', 'poisson', 'tobit', 'negbin', 'gamma', 'gumbel', 'weibull']
```

### Family Classes

```python
from deep_inference import (
    LinearFamily, LogitFamily, PoissonFamily, TobitFamily,
    NegBinFamily, GammaFamily, GumbelFamily, WeibullFamily
)
```

## Module Overview

### structural_dml

The main entry point. Trains a structural neural network with influence function-based inference.

```python
from deep_inference import structural_dml

result = structural_dml(
    Y,                      # Outcome variable (n,)
    T,                      # Treatment variable (n,)
    X,                      # Covariates (n, d)
    family='linear',        # Statistical family
    hidden_dims=[64, 32],   # Network architecture
    epochs=100,             # Training epochs
    n_folds=50,             # Cross-fitting folds
    lr=0.01,               # Learning rate
    batch_size=64,         # Mini-batch size
    weight_decay=1e-4,     # L2 regularization
    verbose=False          # Print progress
)
```

### DMLResult

The result object returned by `structural_dml`:

| Attribute | Description |
|-----------|-------------|
| `mu_hat` | Debiased point estimate of E[beta(X)] |
| `mu_naive` | Naive (biased) estimate |
| `se` | Standard error |
| `ci_lower` | Lower bound of 95% CI |
| `ci_upper` | Upper bound of 95% CI |
| `theta_hat` | Estimated parameters (n, theta_dim) |
| `psi` | Influence scores (n,) |
| `diagnostics` | Dict with training diagnostics |

### families

Statistical families defining loss functions, gradients, Hessians, and influence scores.

### targets

Target functionals for inference: `AverageParameter`, `AME`, `CustomTarget`.

### lambda

Lambda estimation strategies: `ComputeLambda` (Regime A), `AnalyticLambda` (B), `EstimateLambda` (C).

### models

Neural network architectures: `StructuralNet` for parameter estimation.

### metrics

Helper functions for computing coverage and SE ratios.
