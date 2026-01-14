# Weibull Model Tutorial

The Weibull model is for survival analysis and time-to-event data.

## When to Use

Use the Weibull model when:
- Outcome is a positive duration or time-to-event
- Hazard rate changes over time (increasing or decreasing)
- Examples: equipment failure times, customer churn, patient survival, subscription duration

## Mathematical Setup

### Data Generating Process

$$Y \sim \text{Weibull}(k, \lambda)$$

where:
$$\lambda = \exp(\alpha(X) + \beta(X) \cdot T)$$

And:
- $k$ is the shape parameter (controls hazard shape)
- $\lambda$ is the scale parameter
- $E[Y] = \lambda \Gamma(1 + 1/k)$
- $k < 1$: decreasing hazard (early failures)
- $k = 1$: exponential (constant hazard)
- $k > 1$: increasing hazard (wear-out)

### Estimand

$$\mu^* = E[\beta(X)]$$

The average effect on log-scale parameter across the covariate distribution.

### Loss Function

$$L(Y, T, \theta) = k \log(\lambda) - (k-1)\log(Y) + (Y/\lambda)^k$$

Weibull negative log-likelihood (up to constants).

### Influence Score Components

| Component | Formula |
|-----------|---------|
| Scale $\lambda$ | $\exp(\alpha + \beta T)$ |
| $z$ | $(Y/\lambda)^k$ |
| Hessian weight $W$ | $k^2 \cdot z$ |
| Score $\nabla\ell$ | $k(1 - z) \cdot [1, T]$ |

Note: The Hessian depends on $\theta$ through $\lambda = \exp(\alpha + \beta T)$.

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
alpha_true = 3.0 + 0.3 * X[:, 0]
beta_true = 0.5 + 0.2 * X[:, 0]  # Heterogeneous effect
mu_true = beta_true.mean()
shape = 2.0  # Increasing hazard

# Generate Weibull outcomes
scale = np.exp(alpha_true + beta_true * T)
Y = np.random.weibull(shape, size=n) * scale

print(f"True mu* = {mu_true:.6f}")

# Run inference
result = structural_dml(
    Y=Y, T=T, X=X,
    family='weibull',
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
| weibull | 0.993 | 0.986 | PASS |

The influence function correction produces valid confidence intervals. See [Validation](../validation/index.md) for full results.

## Real-World Applications

### Equipment Reliability

Estimate how maintenance affects failure times:

```python
# Y = time to failure (hours)
# T = maintenance frequency
# X = (equipment age, usage intensity, ...)
# Target: E[beta(X)] = average effect on log-lifetime

result = structural_dml(Y, T, X, family='weibull')
```

### Customer Churn

Estimate effect of engagement on subscription duration:

```python
# Y = subscription duration (months)
# T = engagement score
# X = (demographics, plan type, ...)
# Target: E[beta(X)] = average effect on log-duration

result = structural_dml(Y, T, X, family='weibull')
```

### Clinical Trials

Estimate treatment effect on survival time:

```python
# Y = survival time (days)
# T = treatment indicator
# X = (age, disease stage, biomarkers, ...)
# Target: E[beta(X)] = average effect on log-survival

result = structural_dml(Y, T, X, family='weibull')
```

## Key Takeaways

1. **Time-to-event data**: Weibull is the standard for survival analysis with parametric hazard
2. **Shape parameter matters**:
   - $k < 1$: infant mortality (decreasing hazard)
   - $k = 1$: constant hazard (exponential)
   - $k > 1$: wear-out (increasing hazard)
3. **Log-link interpretation**: $\beta$ affects log-scale, so $\exp(\beta)$ is the hazard ratio
4. **Hessian depends on theta**: Requires three-way splitting
