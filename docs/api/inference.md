# Inference Module

Inference methods for structural estimation.

## Main Functions

### naive

```{eval-rst}
.. autofunction:: deepstats.naive
```

### influence

```{eval-rst}
.. autofunction:: deepstats.influence
```

### bootstrap

```{eval-rst}
.. autofunction:: deepstats.bootstrap
```

## Available Methods

```{eval-rst}
.. autodata:: deepstats.METHODS
```

## Usage Example

```python
from deepstats import get_dgp, get_family, naive, influence, bootstrap

# Generate data
dgp = get_dgp("linear", seed=42)
data = dgp.generate(n=2000)

# Get family
family = get_family("linear")

# Configuration
config = {
    "hidden_dims": [64, 32],
    "epochs": 100,
    "n_folds": 50,
    "lr": 0.01
}

# Run different inference methods
naive_result = naive(data.X, data.T, data.Y, family, config)
if_result = influence(data.X, data.T, data.Y, family, config)
boot_result = bootstrap(data.X, data.T, data.Y, family, config)

# Compare results
print(f"True:      {data.mu_true:.4f}")
print(f"Naive:     {naive_result.mu_hat:.4f} +/- {naive_result.se:.4f}")
print(f"Influence: {if_result.mu_hat:.4f} +/- {if_result.se:.4f}")
print(f"Bootstrap: {boot_result.mu_hat:.4f} +/- {boot_result.se:.4f}")
```

## Result Object

All inference functions return a result object with:

| Attribute | Description |
|-----------|-------------|
| `mu_hat` | Point estimate |
| `se` | Standard error |
| `ci_lower` | Lower bound of 95% CI |
| `ci_upper` | Upper bound of 95% CI |
| `psi` | Influence scores (influence method only) |

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dims` | `[64, 32]` | Network architecture |
| `epochs` | `100` | Training epochs |
| `n_folds` | `50` | Cross-fitting folds (influence only) |
| `lr` | `0.01` | Learning rate |
| `batch_size` | `64` | Mini-batch size |
| `weight_decay` | `1e-4` | L2 regularization |
| `n_bootstrap` | `100` | Bootstrap replicates (bootstrap only) |

## Method Comparison

| Method | Coverage | SE Calibration | Speed |
|--------|----------|----------------|-------|
| Naive | ~10-30% | Underestimates | Fast |
| Bootstrap | ~70-85% | Partial | Slow |
| **Influence** | **~95%** | **Correct** | Medium |

## Cross-Fitting Details

The `influence` function uses K-fold cross-fitting:

1. Split data into K folds
2. For each fold k:
   - Train on all other folds
   - Compute influence scores on fold k
3. Aggregate all influence scores
4. Compute mean and SE

This prevents overfitting bias in the influence function estimates.
