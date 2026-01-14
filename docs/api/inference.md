# Inference API

This module provides two main functions for structural estimation with valid inference.

## API Overview

| Function | Use Case | Target | Lambda |
|----------|----------|--------|--------|
| `structural_dml()` | Production, 8 families | E[β(X)] fixed | Estimated |
| `inference()` | Flexible targets, regimes | Custom h(θ) | Auto-selected |

---

## `structural_dml()` - Legacy API

The production-ready API supporting 8 GLM families.

### Signature

```python
from deep_inference import structural_dml

result = structural_dml(
    Y,                      # (n,) outcomes
    T,                      # (n,) treatments
    X,                      # (n, d) covariates
    family='linear',        # Family name: 'linear', 'logit', 'poisson', etc.
    target=None,            # Target variant (e.g., 'ame' for logit)
    hidden_dims=[64, 32],   # Network architecture
    epochs=100,             # Training epochs
    n_folds=50,             # Cross-fitting folds
    lr=0.01,                # Learning rate
    lambda_method='aggregate',  # Lambda estimation method
    verbose=False           # Print progress
)
```

### Supported Families

| Family | Model | θ_dim | Notes |
|--------|-------|-------|-------|
| `linear` | Y = α + βT + ε | 2 | OLS-equivalent |
| `logit` | P(Y=1) = σ(α + βT) | 2 | Binary outcomes |
| `poisson` | Y ~ Pois(exp(α + βT)) | 2 | Count data |
| `gamma` | Y ~ Gamma(k, exp(α + βT)) | 2 | Positive continuous |
| `gumbel` | Y ~ Gumbel(α + βT, σ) | 2 | Extreme values |
| `tobit` | Y = max(0, α + βT + σε) | 3 | Censored |
| `negbin` | Y ~ NegBin(exp(α + βT), r) | 2 | Overdispersed counts |
| `weibull` | Y ~ Weibull(k, exp(α + βT)) | 2 | Survival/duration |

### Example

```python
import numpy as np
from deep_inference import structural_dml

# Generate data
np.random.seed(42)
n = 2000
X = np.random.randn(n, 10)
T = np.random.randn(n)
Y = X[:, 0] + 0.5 * T + np.random.randn(n)

# Run inference
result = structural_dml(
    Y=Y, T=T, X=X,
    family='linear',
    n_folds=50,
    epochs=100
)

print(f"Estimate: {result.mu_hat:.4f} ± {result.se:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

### DMLResult Object

| Attribute | Type | Description |
|-----------|------|-------------|
| `mu_hat` | float | Debiased estimate of E[β(X)] |
| `mu_naive` | float | Naive (biased) estimate |
| `se` | float | Standard error |
| `ci_lower` | float | Lower 95% CI bound |
| `ci_upper` | float | Upper 95% CI bound |
| `theta_hat` | ndarray | Estimated θ(x) for all observations |
| `psi` | ndarray | Influence function values |
| `diagnostics` | dict | Training diagnostics |

---

## `inference()` - New Flexible API

The new API with flexible targets and automatic regime detection.

### Signature

```python
from deep_inference import inference

result = inference(
    Y,                      # (n,) outcomes
    T,                      # (n,) treatments
    X,                      # (n, d) covariates
    # Model specification (choose one):
    model='logit',          # Built-in: 'linear', 'logit'
    loss=None,              # OR custom loss function
    theta_dim=None,         # Required if custom loss
    # Target specification (choose one):
    target='beta',          # Built-in: 'beta', 'ame'
    target_fn=None,         # OR custom target function
    t_tilde=None,           # Evaluation point (default: mean(T))
    # Regime settings:
    is_randomized=False,    # True for RCTs
    treatment_dist=None,    # Known F_T (e.g., Normal(0, 1))
    lambda_method=None,     # Override auto-detection
    # Cross-fitting:
    n_folds=50,
    # Network:
    hidden_dims=[64, 32],
    epochs=100,
    lr=0.01,
    ridge=1e-4,
    verbose=False
)
```

### Built-in Targets

| Target | Formula | Use Case |
|--------|---------|----------|
| `'beta'` | E[β(X)] | Average treatment effect (log-odds for logit) |
| `'ame'` | E[p(1-p)β] | Average marginal effect (probability scale) |

### Custom Target Functions

Define any target h(x, θ, t̃) and the Jacobian is computed via autodiff:

```python
import torch

def my_target(x, theta, t_tilde):
    """Average prediction at treatment level t_tilde."""
    alpha, beta = theta[0], theta[1]
    return torch.sigmoid(alpha + beta * t_tilde)

result = inference(
    Y, T, X,
    model='logit',
    target_fn=my_target,
    t_tilde=0.0
)
```

### Three Regimes

| Regime | Condition | Lambda Method | Cross-Fitting |
|--------|-----------|---------------|---------------|
| **A** | RCT + known F_T | Compute (MC integration) | 2-way |
| **B** | Linear model | Analytic (closed-form) | 2-way |
| **C** | Observational + nonlinear | Estimate (neural net) | 3-way |

```python
from deep_inference.lambda_.compute import Normal

# Regime A: Randomized experiment
result = inference(
    Y, T, X,
    model='logit',
    target='beta',
    is_randomized=True,
    treatment_dist=Normal(mean=0.0, std=1.0)
)
print(f"Regime: {result.diagnostics['regime']}")  # 'A'
```

### InferenceResult Object

| Attribute | Type | Description |
|-----------|------|-------------|
| `mu_hat` | float | Point estimate |
| `se` | float | Standard error |
| `ci_lower` | float | Lower 95% CI |
| `ci_upper` | float | Upper 95% CI |
| `psi_values` | Tensor | Influence function values |
| `theta_hat` | Tensor | Estimated θ(x) |
| `diagnostics` | dict | Regime, lambda method, etc. |

---

## Configuration Guidelines

### Network Architecture

| Sample Size | Recommended |
|-------------|-------------|
| n < 1,000 | `[32, 16]` |
| 1,000 - 10,000 | `[64, 32]` |
| 10,000 - 100,000 | `[128, 64, 32]` |
| n > 100,000 | `[256, 128, 64]` |

### Cross-Fitting Folds

| Use Case | K |
|----------|---|
| Quick exploration | 10-20 |
| Production | 50 |
| Very large data | 20-50 |

### Lambda Method (for `structural_dml`)

| Method | When to Use |
|--------|-------------|
| `'aggregate'` | Default for nonlinear models |
| `'pointwise'` | When heterogeneity is smooth |

---

## Algorithm Overview

### Cross-Fitting (K-Fold)

```
For k = 1 to K:
    Train: Fit θ̂(x) on folds ≠ k
    [If 3-way: Fit Λ̂(x) on separate fold]
    Eval: Compute ψ on fold k

Aggregate: μ̂ = mean(ψ), SE = std(ψ)/√n
```

### Influence Function

The influence function corrects for regularization bias:

```
ψ(z) = H(θ̂) - H_θ · Λ(x)⁻¹ · ℓ_θ(z, θ̂)
```

Where:
- H(θ) = target functional (e.g., E[β])
- H_θ = Jacobian of target w.r.t. θ
- Λ(x) = E[ℓ_θθ | X=x] = conditional Hessian
- ℓ_θ = score (gradient of loss)

---

## Expected Performance

| Method | Coverage | SE Ratio |
|--------|----------|----------|
| Naive | ~10-30% | << 1 |
| **Influence** | **~95%** | **~1.0** |
