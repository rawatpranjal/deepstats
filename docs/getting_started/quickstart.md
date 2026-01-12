# Quickstart

This guide shows you how to use `deep-inference` to estimate treatment effects with valid inference.

## Basic Workflow

### 1. Import and Prepare Data

```python
import numpy as np
from deep_inference import structural_dml

# Example: Generate synthetic data
np.random.seed(42)
n = 2000
X = np.random.randn(n, 10)  # Covariates
T = np.random.randn(n)       # Treatment
beta_true = np.cos(np.pi * X[:, 0]) * (X[:, 1] > 0) + 0.5 * X[:, 2]
Y = X[:, 0] + beta_true * T + np.random.randn(n)
```

### 2. Run Inference

```python
result = structural_dml(
    Y=Y,
    T=T,
    X=X,
    family='linear',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50,
    lr=0.01
)
```

### 3. Interpret Results

```python
print(f"Estimate:    {result.mu_hat:.4f}")
print(f"Std Error:   {result.se:.4f}")
print(f"95% CI:      [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dims` | `[64, 32]` | Network architecture |
| `epochs` | `100` | Training epochs |
| `n_folds` | `50` | Cross-fitting folds |
| `lr` | `0.01` | Learning rate |
| `batch_size` | `64` | Mini-batch size |
| `weight_decay` | `1e-4` | L2 regularization |

## Comparing Naive vs Debiased

```python
result = structural_dml(
    Y=Y, T=T, X=X,
    family='linear',
    epochs=100,
    n_folds=50
)

# The result contains both estimates
print(f"Naive estimate:    {result.mu_naive:.4f}")
print(f"Debiased estimate: {result.mu_hat:.4f}")
print(f"Difference:        {result.mu_hat - result.mu_naive:.4f}")

# Expected coverage:
# - Naive: ~10-30% (severely undercovered)
# - Debiased (Influence): ~95%
```

## Supported Families

```python
# Linear: continuous outcomes
result = structural_dml(Y, T, X, family='linear')

# Logit: binary outcomes (0/1)
result = structural_dml(Y, T, X, family='logit')

# Poisson: count outcomes
result = structural_dml(Y, T, X, family='poisson')

# Tobit: censored continuous outcomes
result = structural_dml(Y, T, X, family='tobit')

# And more: gamma, gumbel, negbin, weibull
```

## Next Steps

- See [Tutorials](../tutorials/index.md) for detailed examples with each model
- Read [Theory](../theory/index.md) for the mathematical background
- Check [API Reference](../api/index.md) for complete documentation
