# Gumbel Model Tutorial

The Gumbel model is for extreme value analysis and modeling maxima/minima.

## When to Use

Use the Gumbel model when:
- Outcome represents an extreme value (maximum or minimum)
- Data follows a Type I extreme value distribution
- Examples: maximum flood levels, peak stress loads, extreme temperatures, maximum auction bids

## Mathematical Setup

### Data Generating Process

$$Y \sim \text{Gumbel}(\mu, \sigma)$$

where:
$$\mu = \alpha(X) + \beta(X) \cdot T$$

And:
- $\mu$ is the location parameter
- $\sigma$ is the scale parameter (fixed)
- $E[Y] = \mu + \sigma \gamma$ (where $\gamma \approx 0.5772$ is Euler's constant)
- $\text{Var}[Y] = \pi^2 \sigma^2 / 6$

### Estimand

$$\mu^* = E[\beta(X)]$$

The average effect on the location parameter across the covariate distribution.

### Loss Function

$$L(Y, T, \theta) = z + \exp(-z)$$

where $z = (Y - \mu) / \sigma$ is the standardized residual.

### Influence Score Components

| Component | Formula |
|-----------|---------|
| Standardized $z$ | $(Y - \mu) / \sigma$ |
| Hessian weight $W$ | $\exp(-z) / \sigma^2$ |
| Score $\nabla\ell$ | $-\frac{1}{\sigma}(1 - e^{-z}) \cdot [1, T]$ |

Note: The Hessian depends on $\theta$ through $z = (Y - \mu) / \sigma$.

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
alpha_true = 5.0 + 0.5 * X[:, 0]
beta_true = 1.0 + 0.3 * X[:, 0]  # Heterogeneous effect
mu_true = beta_true.mean()
scale = 1.0

# Generate Gumbel outcomes
mu = alpha_true + beta_true * T
Y = np.random.gumbel(mu, scale, size=n)

print(f"True mu* = {mu_true:.6f}")

# Run inference
result = structural_dml(
    Y=Y, T=T, X=X,
    family='gumbel',
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

From [Eval 01: Parameter Recovery](../validation/eval_01.md):

| Family | Corr(α) | Corr(β) | Status |
|--------|---------|---------|--------|
| gumbel | 0.967 | 0.991 | PASS |

The influence function correction produces valid confidence intervals. See [Validation](../validation/index.md) for full results.

## Real-World Applications

### Flood Risk Analysis

Estimate how infrastructure affects maximum flood levels:

```python
# Y = annual maximum flood height
# T = dam capacity
# X = (upstream area, rainfall, soil type, ...)
# Target: E[beta(X)] = average effect on flood maxima

result = structural_dml(Y, T, X, family='gumbel')
```

### Structural Engineering

Estimate effect of material properties on maximum stress:

```python
# Y = maximum observed stress
# T = material thickness
# X = (load conditions, temperature, ...)
# Target: E[beta(X)] = average effect on max stress

result = structural_dml(Y, T, X, family='gumbel')
```

### Auction Prices

Estimate how auction features affect maximum bids:

```python
# Y = winning bid price
# T = auction duration
# X = (item characteristics, bidder count, ...)
# Target: E[beta(X)] = average effect on max bid

result = structural_dml(Y, T, X, family='gumbel')
```

## Key Takeaways

1. **Extreme value data**: Gumbel is for modeling maxima (or minima with sign flip)
2. **Location-scale family**: Linear effect on location parameter
3. **Hessian depends on theta**: Requires three-way splitting
4. **Scale parameter**: Fixed at initialization; controls spread of distribution
