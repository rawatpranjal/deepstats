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
print(result.summary())
```

Output:
```
==============================================================================
                            Structural DML Results
==============================================================================
Family:           Linear               Target:           E[beta]
No. Observations: 2000                 No. Folds:        50
==============================================================================
                  coef     std err         z     P>|z|      [0.025    0.975]
------------------------------------------------------------------------------
     E[beta]    0.0312      0.0234     1.333    0.183    -0.0147    0.0771
==============================================================================
Diagnostics:
  Min Lambda eigenvalue:    0.998765
  Mean condition number:    1.00
  Correction ratio:         0.1234
------------------------------------------------------------------------------
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

# View full results including naive estimate
print(result.summary())

# Access individual components if needed
print(f"\nNaive estimate:    {result.mu_naive:.4f}")
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

## New `inference()` API

The new API provides additional flexibility for targets and experimental designs.

### Flexible Targets

```python
from deep_inference import inference

# Average Marginal Effect (probability scale, not log-odds)
result = inference(
    Y, T, X,
    model='logit',
    target='ame',
    t_tilde=0.0  # Evaluate at T=0
)

print(f"AME: {result.mu_hat:.4f} ± {result.se:.4f}")
```

### Custom Target Functions

Define any target and get autodiff Jacobians for free:

```python
import torch

def avg_prediction(x, theta, t_tilde):
    """Average P(Y=1|T=t̃)"""
    alpha, beta = theta[0], theta[1]
    return torch.sigmoid(alpha + beta * t_tilde)

result = inference(
    Y, T, X,
    model='logit',
    target_fn=avg_prediction,
    t_tilde=0.0
)
```

### Randomized Experiments (Regime A)

For RCTs with known treatment distribution, Lambda can be computed instead of estimated:

```python
from deep_inference.lambda_.compute import Normal

result = inference(
    Y, T, X,
    model='logit',
    target='beta',
    is_randomized=True,
    treatment_dist=Normal(mean=0.0, std=1.0)
)

print(f"Regime: {result.diagnostics['regime']}")  # 'A'
```

Benefits:
- No neural network needed for Lambda
- 2-way cross-fitting (faster)
- More stable estimates

## Next Steps

- See [Tutorials](../tutorials/index.md) for detailed examples with each model
- Read [Theory](../theory/index.md) for the mathematical background
- Check [API Reference](../api/index.md) for complete documentation
