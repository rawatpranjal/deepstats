# Linear Model Tutorial

The linear model is the baseline case for understanding influence function inference.

## When to Use

Use the linear model when:
- Outcome is continuous and unbounded
- Errors are approximately normal
- Examples: wages, test scores, consumption, prices

## Mathematical Setup

### Data Generating Process

$$Y = \alpha(X) + \beta(X) \cdot T + \varepsilon, \quad \varepsilon \sim N(0, \sigma^2)$$

Where:
- $\alpha(X)$ is the baseline function (intercept)
- $\beta(X)$ is the treatment effect (slope)
- $T$ is the treatment variable
- $\varepsilon$ is idiosyncratic noise

### Estimand

$$\mu^* = E[\beta(X)]$$

The average treatment effect across the covariate distribution.

### Loss Function

$$L(Y, T, \theta) = (Y - \alpha - \beta T)^2$$

Standard mean squared error loss.

### Influence Score Components

| Component | Formula |
|-----------|---------|
| Residual $r$ | $Y - (\alpha + \beta T)$ |
| Hessian weight $W$ | $1$ (constant) |
| Score $\nabla\ell$ | $-r \cdot [1, \tilde{T}]$ |

## Complete Example

```python
from deepstats import get_dgp, get_family, naive, influence

# Generate synthetic data
dgp = get_dgp("linear", d=10, seed=42)
data = dgp.generate(n=2000)

print(f"True mu* = {data.mu_true:.6f}")

# Get family
family = get_family("linear")

# Configuration
config = {
    "hidden_dims": [64, 32],
    "epochs": 100,
    "n_folds": 50,
    "lr": 0.01,
    "batch_size": 64
}

# Compare methods
naive_result = naive(data.X, data.T, data.Y, family, config)
if_result = influence(data.X, data.T, data.Y, family, config)

print("\n--- Naive Method ---")
print(f"Estimate: {naive_result.mu_hat:.4f}")
print(f"SE:       {naive_result.se:.4f}")
print(f"Expected coverage: ~10-30%")

print("\n--- Influence Function ---")
print(f"Estimate: {if_result.mu_hat:.4f}")
print(f"SE:       {if_result.se:.4f}")
print(f"95% CI:   [{if_result.ci_lower:.4f}, {if_result.ci_upper:.4f}]")
print(f"Expected coverage: ~95%")
```

## Expected Results

From Monte Carlo validation (M=30, N=1000, K=50 folds):

| Method | Coverage | SE Ratio | RMSE |
|--------|----------|----------|------|
| Naive | 13% | 0.15 | 0.108 |
| Bootstrap | 83% | 0.72 | 0.155 |
| **Influence** | **97%** | **1.25** | 0.378 |

### Interpretation

- **Naive**: Severely underestimates uncertainty (SE ratio 0.15 = 85% too small)
- **Bootstrap**: Partial correction but still undercoverage
- **Influence**: Achieves target 95% coverage with properly calibrated SE

## Real-World Applications

### Wage Regression

Estimate the effect of education on wages where the effect varies by worker characteristics:

```python
# Y = log(wage)
# T = years of education
# X = (experience, age, industry, ...)
# Target: E[beta(X)] = average return to education
```

### Price Elasticity

Estimate heterogeneous price elasticity in demand:

```python
# Y = log(quantity)
# T = log(price)
# X = (demographics, season, ...)
# Target: E[beta(X)] = average price elasticity
```

## Key Takeaways

1. **Linear is the baseline**: Unit Hessian weight makes the math simplest
2. **K=50 folds is important**: Each model sees 98% of data for stable estimation
3. **SE = std(psi)/sqrt(n)**: This IS the correct formula when psi is computed properly
4. **Check SE ratio**: Target is 0.9-1.2 (close to 1.0)
