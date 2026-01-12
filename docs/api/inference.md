# Inference Module

The main inference function for structural estimation.

## Main Function

### structural_dml

```python
from deep_inference import structural_dml

result = structural_dml(
    Y,                      # Outcome variable (n,) array-like
    T,                      # Treatment variable (n,) array-like
    X,                      # Covariates (n, d) array-like
    family='linear',        # Family name or instance
    hidden_dims=[64, 32],   # Network architecture
    epochs=100,             # Training epochs
    n_folds=50,             # Cross-fitting folds
    lr=0.01,               # Learning rate
    batch_size=64,         # Mini-batch size
    weight_decay=1e-4,     # L2 regularization
    verbose=False          # Print progress
)
```

## Usage Example

```python
import numpy as np
from deep_inference import structural_dml

# Prepare data
np.random.seed(42)
n = 2000
X = np.random.randn(n, 10)
T = np.random.randn(n)
Y = X[:, 0] + 0.5 * T + np.random.randn(n)

# Run inference
result = structural_dml(
    Y=Y, T=T, X=X,
    family='linear',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50
)

# Access results
print(f"Estimate: {result.mu_hat:.4f}")
print(f"SE: {result.se:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
print(f"Naive: {result.mu_naive:.4f}")
```

## DMLResult Object

The result object contains:

| Attribute | Type | Description |
|-----------|------|-------------|
| `mu_hat` | float | Debiased point estimate of E[beta(X)] |
| `mu_naive` | float | Naive (biased) estimate |
| `se` | float | Standard error |
| `ci_lower` | float | Lower bound of 95% CI |
| `ci_upper` | float | Upper bound of 95% CI |
| `theta_hat` | ndarray | Estimated parameters (n, theta_dim) |
| `psi` | ndarray | Influence scores (n,) |
| `diagnostics` | dict | Training diagnostics |

### Diagnostics Dictionary

```python
diagnostics = result.diagnostics
print(diagnostics.get('min_lambda_eigenvalue'))  # Hessian stability
print(diagnostics.get('correction_ratio'))       # IF correction magnitude
print(diagnostics.get('pct_regularized'))        # % observations regularized
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dims` | `[64, 32]` | Hidden layer sizes |
| `epochs` | `100` | Training epochs |
| `n_folds` | `50` | Cross-fitting folds |
| `lr` | `0.01` | Learning rate |
| `batch_size` | `64` | Mini-batch size |
| `weight_decay` | `1e-4` | L2 regularization |
| `verbose` | `False` | Print progress |

### Architecture Guidelines

| Sample Size | Recommended Architecture |
|-------------|-------------------------|
| n < 1,000 | `[32, 16]` |
| 1,000 < n < 10,000 | `[64, 32]` |
| 10,000 < n < 100,000 | `[128, 64, 32]` |
| n > 100,000 | `[256, 128, 64]` |

### Fold Selection

| Use Case | Recommended K |
|----------|--------------|
| Quick exploration | 10-20 |
| Final results | 50 |
| Very large data | 20-50 |

## Cross-Fitting Algorithm

The `structural_dml` function uses K-fold cross-fitting:

1. Split data into K folds
2. For each fold k:
   - Train structural network on all other folds
   - Compute influence scores on fold k
3. Aggregate all influence scores
4. Compute: `mu_hat = mean(psi)`, `se = std(psi) / sqrt(n)`

This prevents overfitting bias in the influence function estimates.

## Comparing Naive vs Debiased

```python
result = structural_dml(Y, T, X, family='linear')

print(f"Naive estimate:    {result.mu_naive:.4f}")
print(f"Debiased estimate: {result.mu_hat:.4f}")
print(f"Bias correction:   {result.mu_hat - result.mu_naive:.4f}")

# The naive estimate underestimates uncertainty
# The debiased estimate has valid 95% coverage
```

## Expected Coverage

| Method | Coverage | SE Calibration |
|--------|----------|----------------|
| Naive | ~10-30% | Underestimates |
| **Influence** | **~95%** | **Correct** |
