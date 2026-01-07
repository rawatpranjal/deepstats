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

$$\mu^* = E[0.3 \cdot \beta(X)]$$

Same as Poisson, with 0.3 scaling.

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
from deepstats import get_dgp, get_family, naive, influence
import numpy as np

# Generate overdispersed count data
dgp = get_dgp("negbin", d=10, overdispersion=0.5, seed=42)
data = dgp.generate(n=2000)

print(f"True mu* = {data.mu_true:.6f}")
print(f"Mean count = {data.Y.mean():.2f}")
print(f"Variance = {data.Y.var():.2f}")
print(f"Variance/Mean ratio = {data.Y.var()/data.Y.mean():.2f}")

# Get family
family = get_family("negbin")

# Configuration
config = {
    "hidden_dims": [64, 32],
    "epochs": 100,
    "n_folds": 50,
    "lr": 0.01
}

# Run inference
if_result = influence(data.X, data.T, data.Y, family, config)

print("\n--- Influence Function ---")
print(f"Estimate: {if_result.mu_hat:.4f}")
print(f"SE:       {if_result.se:.4f}")
print(f"95% CI:   [{if_result.ci_lower:.4f}, {if_result.ci_upper:.4f}]")
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
mean_y = data.Y.mean()
var_y = data.Y.var()
dispersion_ratio = var_y / mean_y

print(f"Mean: {mean_y:.2f}")
print(f"Variance: {var_y:.2f}")
print(f"Dispersion ratio: {dispersion_ratio:.2f}")

if dispersion_ratio > 1.5:
    print("Overdispersion detected -> use NegBin")
elif dispersion_ratio < 0.8:
    print("Underdispersion detected -> consider alternatives")
else:
    print("Approximately equidispersed -> Poisson OK")
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
```

### Insurance Claims

```python
# Y = number of claims filed
# T = deductible amount
# X = (policy type, customer age, history, ...)
# Target: E[beta(X)] = average deductible effect on claim frequency

# Why NegBin: Claim counts often show clustering
# (some customers file many claims, most file few)
```

### Species Counts

```python
# Y = number of species observed
# T = habitat protection level
# X = (area size, climate, elevation, ...)
# Target: E[beta(X)] = average protection effect on biodiversity

# Why NegBin: Ecological counts are typically overdispersed
```

## Overdispersion Parameter

The overdispersion parameter $\alpha$ controls how much extra variance exists:

- $\alpha = 0$: Reduces to Poisson
- $\alpha = 0.5$: Moderate overdispersion (default in DGP)
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
