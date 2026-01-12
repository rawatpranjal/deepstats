# Negative Binomial Model Tutorial

The Negative Binomial model handles overdispersed count data.

## When to Use

Use the Negative Binomial model when:
- Outcome is a non-negative integer (0, 1, 2, ...)
- Variance exceeds the mean (overdispersion)
- Poisson model underfits the variance
- Examples: doctor visits with heavy users, insurance claims

## Mathematical Setup

### Data Generating Process

$$Y \sim \text{NegBin}(\mu, r)$$

Where:
$$\mu = \exp(\alpha(X) + \beta(X) \cdot T)$$

The variance is $\text{Var}(Y) = \mu + \alpha \mu^2$ where $\alpha$ is the overdispersion parameter.

### Estimand

$$\mu^* = E[\beta(X)]$$

The average treatment effect on the log-rate (same as Poisson).

### Loss Function

$$L(Y, T, \theta) = \mu - Y \log \mu$$

Modified Poisson-like loss accounting for overdispersion.

### Influence Score Components

| Component | Formula |
|-----------|---------|
| Residual $r$ | $(Y - \mu) / (1 + \alpha\mu)$ |
| Hessian weight $W$ | $\mu / (1 + \alpha\mu)$ |
| Score $\nabla\ell$ | $-r \cdot [1, \tilde{T}]$ |

The overdispersion $\alpha$ downweights high-mean observations.

## Complete Example

```python
import numpy as np
from deep_inference import structural_dml

# Generate overdispersed count data
np.random.seed(42)
n = 2000
X = np.random.randn(n, 10)
T = np.random.randn(n)

# True structural functions
alpha_true = 1.0 + 0.2 * X[:, 0]
beta_true = 0.3 + 0.1 * X[:, 0]
mu = np.exp(alpha_true + beta_true * T)

# Add overdispersion via gamma-Poisson mixture
r = 2.0  # dispersion parameter
p = r / (r + mu)
Y = np.random.negative_binomial(r, p).astype(float)
mu_true = beta_true.mean()

print(f"True mu* = {mu_true:.6f}")
print(f"Mean count = {Y.mean():.2f}")
print(f"Variance = {Y.var():.2f}")
print(f"Variance/Mean ratio = {Y.var()/Y.mean():.2f}")

# Run inference
result = structural_dml(
    Y=Y, T=T, X=X,
    family='negbin',
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

## Poisson vs Negative Binomial

### When to Use Each

| Condition | Model |
|-----------|-------|
| Var(Y) $\approx$ Mean(Y) | Poisson |
| Var(Y) > Mean(Y) | Negative Binomial |
| Var(Y) < Mean(Y) | Underdispersion (rare) |

### Diagnostic Check

```python
# Simple overdispersion test
mean_y = Y.mean()
var_y = Y.var()
dispersion_ratio = var_y / mean_y

print(f"Mean: {mean_y:.2f}")
print(f"Variance: {var_y:.2f}")
print(f"Dispersion ratio: {dispersion_ratio:.2f}")

if dispersion_ratio > 1.5:
    print("Overdispersion detected -> use NegBin")
    result = structural_dml(Y, T, X, family='negbin')
elif dispersion_ratio < 0.8:
    print("Underdispersion detected -> consider alternatives")
else:
    print("Approximately equidispersed -> Poisson OK")
    result = structural_dml(Y, T, X, family='poisson')
```

## Real-World Applications

### Healthcare Utilization

```python
# Y = number of doctor visits
# T = insurance status
# X = (age, chronic conditions, income, ...)
# Target: E[beta(X)] = average insurance effect on utilization

# Why NegBin: Some patients are heavy users (many visits),
# creating overdispersion in visit counts
result = structural_dml(Y, T, X, family='negbin')
```

### Insurance Claims

```python
# Y = number of claims filed
# T = deductible amount
# X = (policy type, customer age, history, ...)
# Target: E[beta(X)] = average deductible effect on claim frequency

# Why NegBin: Claim counts often show clustering
# (some customers file many claims, most file few)
result = structural_dml(Y, T, X, family='negbin')
```

### Species Counts

```python
# Y = number of species observed
# T = habitat protection level
# X = (area size, climate, elevation, ...)
# Target: E[beta(X)] = average protection effect on biodiversity

# Why NegBin: Ecological counts are typically overdispersed
result = structural_dml(Y, T, X, family='negbin')
```

## Overdispersion Parameter

The overdispersion parameter $\alpha$ controls how much extra variance exists:

- $\alpha = 0$: Reduces to Poisson
- $\alpha = 0.5$: Moderate overdispersion
- $\alpha = 1.0$: Strong overdispersion

### Effect on Inference

Higher overdispersion means:
- Less information per observation
- Wider confidence intervals
- Weight function $W = \mu/(1 + \alpha\mu)$ approaches $1/\alpha$ for large $\mu$

## Key Takeaways

1. **Check dispersion first**: Plot variance vs mean before choosing model
2. **Overdispersion is common**: Real count data usually shows Var > Mean
3. **Same interpretation as Poisson**: Log-link means semi-elasticity
4. **Weight downweighting**: High-count observations get relatively less weight than in Poisson
5. **Robust to misspecification**: NegBin is safer default than Poisson for count data
