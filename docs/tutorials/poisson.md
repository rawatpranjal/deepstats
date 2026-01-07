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

$$\mu^* = E[0.3 \cdot \beta(X)]$$

Note: The DGP scales $\beta$ by 0.3 to keep counts in a reasonable range.

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
from deepstats import get_dgp, get_family, naive, influence

# Generate count data
dgp = get_dgp("poisson", d=10, seed=42)
data = dgp.generate(n=2000)

print(f"True mu* = {data.mu_true:.6f}")
print(f"Mean count = {data.Y.mean():.2f}")
print(f"Max count = {data.Y.max()}")

# Get family
family = get_family("poisson")

# Configuration
config = {
    "hidden_dims": [64, 32],
    "epochs": 100,
    "n_folds": 50,
    "lr": 0.01
}

# Run inference
naive_result = naive(data.X, data.T, data.Y, family, config)
if_result = influence(data.X, data.T, data.Y, family, config)

print("\n--- Naive Method ---")
print(f"Estimate: {naive_result.mu_hat:.4f}")
print(f"SE:       {naive_result.se:.4f}")

print("\n--- Influence Function ---")
print(f"Estimate: {if_result.mu_hat:.4f}")
print(f"SE:       {if_result.se:.4f}")
print(f"95% CI:   [{if_result.ci_lower:.4f}, {if_result.ci_upper:.4f}]")
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
```

### Doctor Visits

```python
# Y = number of doctor visits per year
# T = insurance generosity
# X = (age, health status, income, ...)
# Target: E[beta(X)] = average effect of insurance on utilization
```

### Traffic Accidents

```python
# Y = number of accidents at intersection
# T = speed limit
# X = (traffic volume, weather, road design, ...)
# Target: E[beta(X)] = average effect of speed on accidents
```

## Poisson vs Negative Binomial

If your count data shows **overdispersion** (variance > mean), consider the Negative Binomial model instead:

```python
# Check for overdispersion
print(f"Mean: {data.Y.mean():.2f}")
print(f"Variance: {data.Y.var():.2f}")

if data.Y.var() > 1.5 * data.Y.mean():
    print("Consider using NegBin model")
    family = get_family("negbin")
```

## Key Takeaways

1. **Log-link interpretation**: Coefficients are semi-elasticities
2. **Weight = lambda**: High counts get more influence
3. **Check for overdispersion**: Use NegBin if variance >> mean
4. **Scale factor**: DGP uses 0.3 scaling for reasonable count ranges
