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
import numpy as np
from deep_inference import structural_dml

# Generate synthetic data
np.random.seed(42)
n = 2000
X = np.random.randn(n, 10)
T = np.random.randn(n)
beta_true = 0.5 + 0.3 * X[:, 0]  # Heterogeneous effect
Y = X[:, 0] + beta_true * T + np.random.randn(n)
mu_true = beta_true.mean()

print(f"True mu* = {mu_true:.6f}")

# Run inference
result = structural_dml(
    Y=Y, T=T, X=X,
    family='linear',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50,
    lr=0.01
)

print("\n--- Results ---")
print(f"Estimate: {result.mu_hat:.4f}")
print(f"SE:       {result.se:.4f}")
print(f"95% CI:   [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")

# Compare to naive
print(f"\nNaive estimate: {result.mu_naive:.4f}")
print(f"Bias correction: {result.mu_hat - result.mu_naive:.4f}")
```

## Expected Results

From [Eval 01: Parameter Recovery](../validation/eval_01.md):

| Family | Corr(α) | Corr(β) | Status |
|--------|---------|---------|--------|
| linear | 0.994 | 0.998 | PASS |

The influence function correction produces valid confidence intervals. See [Validation](../validation/index.md) for full results.

## Real-World Applications

### Wage Regression

Estimate the effect of education on wages where the effect varies by worker characteristics:

```python
# Y = log(wage)
# T = years of education
# X = (experience, age, industry, ...)
# Target: E[beta(X)] = average return to education

result = structural_dml(Y, T, X, family='linear')
```

### Price Elasticity

Estimate heterogeneous price elasticity in demand:

```python
# Y = log(quantity)
# T = log(price)
# X = (demographics, season, ...)
# Target: E[beta(X)] = average price elasticity

result = structural_dml(Y, T, X, family='linear')
```

## Key Takeaways

1. **Linear is the baseline**: Unit Hessian weight makes the math simplest
2. **K=50 folds is important**: Each model sees 98% of data for stable estimation
3. **SE = std(psi)/sqrt(n)**: This IS the correct formula when psi is computed properly
4. **Check SE ratio**: Target is 0.9-1.2 (close to 1.0)
