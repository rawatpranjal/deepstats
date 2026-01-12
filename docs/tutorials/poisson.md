# Poisson Model Tutorial

The Poisson model handles count data with heterogeneous treatment effects.

## When to Use

Use the Poisson model when:
- Outcome is a non-negative integer (0, 1, 2, ...)
- Variance approximately equals the mean
- Examples: patent counts, doctor visits, accidents

## Mathematical Setup

### Data Generating Process

$$Y \sim \text{Poisson}(\lambda(X, T))$$

Where:
$$\lambda = \exp(\alpha(X) + \beta(X) \cdot T)$$

The log-link ensures $\lambda > 0$.

### Estimand

$$\mu^* = E[\beta(X)]$$

The average treatment effect on the log-rate.

### Loss Function

$$L(Y, T, \theta) = \lambda - Y \log \lambda$$

Poisson negative log-likelihood (up to constants).

### Influence Score Components

| Component | Formula |
|-----------|---------|
| Residual $r$ | $Y - \lambda$ |
| Hessian weight $W$ | $\lambda$ |
| Score $\nabla\ell$ | $-r \cdot [1, \tilde{T}]$ |

Note: Weight $W = \lambda$ means high-count observations get more weight.

## Complete Example

```python
import numpy as np
from deep_inference import structural_dml

# Generate count data
np.random.seed(42)
n = 2000
X = np.random.randn(n, 10)
T = np.random.randn(n)

# True structural functions
alpha_true = 1.0 + 0.2 * X[:, 0]
beta_true = 0.3 + 0.1 * X[:, 0]
lam = np.exp(alpha_true + beta_true * T)
Y = np.random.poisson(lam).astype(float)
mu_true = beta_true.mean()

print(f"True mu* = {mu_true:.6f}")
print(f"Mean count = {Y.mean():.2f}")
print(f"Max count = {Y.max()}")

# Run inference
result = structural_dml(
    Y=Y, T=T, X=X,
    family='poisson',
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

## Interpreting Coefficients

With the log-link, $\beta$ represents a **semi-elasticity**:

$$\frac{\partial \log E[Y]}{\partial T} = \beta(X)$$

A unit increase in $T$ changes $E[Y]$ by approximately $100 \cdot \beta$%.

### Example Interpretation

If $\hat{\mu} = 0.05$, then on average a 1-unit increase in treatment increases the expected count by 5%.

## Real-World Applications

### Patent Counts

```python
# Y = number of patents filed
# T = R&D spending (log)
# X = (firm size, industry, prior patents, ...)
# Target: E[beta(X)] = average R&D elasticity of patenting

result = structural_dml(Y, T, X, family='poisson')
```

### Doctor Visits

```python
# Y = number of doctor visits per year
# T = insurance generosity
# X = (age, health status, income, ...)
# Target: E[beta(X)] = average effect of insurance on utilization

result = structural_dml(Y, T, X, family='poisson')
```

### Traffic Accidents

```python
# Y = number of accidents at intersection
# T = speed limit
# X = (traffic volume, weather, road design, ...)
# Target: E[beta(X)] = average effect of speed on accidents

result = structural_dml(Y, T, X, family='poisson')
```

## Poisson vs Negative Binomial

If your count data shows **overdispersion** (variance > mean), consider the Negative Binomial model instead:

```python
# Check for overdispersion
print(f"Mean: {Y.mean():.2f}")
print(f"Variance: {Y.var():.2f}")

if Y.var() > 1.5 * Y.mean():
    print("Consider using NegBin model")
    result = structural_dml(Y, T, X, family='negbin')
```

## Key Takeaways

1. **Log-link interpretation**: Coefficients are semi-elasticities
2. **Weight = lambda**: High counts get more influence
3. **Check for overdispersion**: Use NegBin if variance >> mean
4. **Count data is common**: Many economic outcomes are counts
