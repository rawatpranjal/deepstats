# Gamma Model Tutorial

The Gamma model is for continuous positive outcomes with right-skewed distributions.

## When to Use

Use the Gamma model when:
- Outcome is continuous and strictly positive
- Data is right-skewed (long right tail)
- Variance increases with the mean
- Examples: insurance claims, healthcare costs, time durations, income

## Mathematical Setup

### Data Generating Process

$$Y \sim \text{Gamma}(k, \mu/k)$$

where:
$$\mu = \exp(\alpha(X) + \beta(X) \cdot T)$$

And:
- $k$ is the shape parameter (controls variance)
- $\mu$ is the mean: $E[Y] = \mu$
- $\text{Var}[Y] = \mu^2 / k$

### Estimand

$$\mu^* = E[\beta(X)]$$

The average effect on the log-mean across the covariate distribution.

### Loss Function

$$L(Y, T, \theta) = Y/\mu + \log(\mu)$$

Gamma deviance loss (up to constants).

### Influence Score Components

| Component | Formula |
|-----------|---------|
| Residual | $r = 1 - Y/\mu$ |
| Hessian weight $W$ | $Y/\mu$ |
| Score $\nabla\ell$ | $r \cdot [1, T]$ |

Note: The Hessian depends on $\theta$ through $\mu = \exp(\alpha + \beta T)$.

## Complete Example

```python
import numpy as np
from deep_inference import structural_dml

# Generate synthetic data
np.random.seed(42)
n = 2000
X = np.random.randn(n, 10)
T = np.random.randn(n)

# True parameters
alpha_true = 2.0 + 0.3 * X[:, 0]
beta_true = 0.5 + 0.2 * X[:, 0]  # Heterogeneous effect
mu_true = beta_true.mean()

# Generate Gamma outcomes
mu = np.exp(alpha_true + beta_true * T)
shape = 2.0
Y = np.random.gamma(shape, mu / shape, size=n)

print(f"True mu* = {mu_true:.6f}")

# Run inference
result = structural_dml(
    Y=Y, T=T, X=X,
    family='gamma',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50,
    lr=0.01
)

print("\n--- Results ---")
print(f"Estimate: {result.mu_hat:.4f}")
print(f"SE:       {result.se:.4f}")
print(f"95% CI:   [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

## Expected Results

From Monte Carlo validation (M=30, N=10,000, K=50 folds):

| Method | Coverage | SE Ratio | Target |
|--------|----------|----------|--------|
| Naive | ~10-30% | ~0.3-0.5 | — |
| **Influence** | **~95-100%** | **~1.2** | 93-97% |

### Interpretation

- The Gamma family tends to produce slightly conservative SE estimates (ratio > 1.0)
- Valid coverage is maintained across different shape parameters

## Real-World Applications

### Healthcare Costs

Estimate the effect of a treatment on medical expenditure:

```python
# Y = medical costs ($)
# T = treatment indicator
# X = (age, comorbidities, insurance type, ...)
# Target: E[beta(X)] = average effect on log-cost

result = structural_dml(Y, T, X, family='gamma')
```

### Insurance Claims

Estimate how policy features affect claim amounts:

```python
# Y = claim amount
# T = deductible level
# X = (policyholder demographics, ...)
# Target: E[beta(X)] = average elasticity

result = structural_dml(Y, T, X, family='gamma')
```

## Key Takeaways

1. **Right-skewed positive data**: Gamma is ideal when outcomes are strictly positive and variance increases with mean
2. **Hessian depends on theta**: Requires three-way splitting (automatic in `structural_dml`)
3. **Log-link interpretation**: $\beta$ represents effect on log-mean, so $\exp(\beta)$ is multiplicative effect
4. **Shape parameter**: Higher shape = lower variance relative to mean
