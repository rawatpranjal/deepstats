# Logit Model Tutorial

The logit model handles binary outcomes with heterogeneous treatment effects.

## When to Use

Use the logit model when:
- Outcome is binary (0 or 1)
- You're modeling probability of an event
- Examples: purchase decisions, market entry, default

## Mathematical Setup

### Data Generating Process

$$P(Y=1 | X, T) = \sigma(\alpha(X) + \beta(X) \cdot T)$$

Where $\sigma(z) = 1/(1 + e^{-z})$ is the sigmoid function.

### Estimand

$$\mu^* = E[\beta(X)]$$

The average log-odds ratio effect of treatment.

### Loss Function

$$L(Y, T, \theta) = -[Y \log p + (1-Y) \log(1-p)]$$

Binary cross-entropy loss where $p = \sigma(\alpha + \beta T)$.

### Influence Score Components

| Component | Formula |
|-----------|---------|
| Residual $r$ | $Y - p$ |
| Hessian weight $W$ | $p(1-p)$ |
| Score $\nabla\ell$ | $-r \cdot [1, \tilde{T}]$ |

Note: The weight $W = p(1-p)$ varies across observations based on predicted probability.

## Complete Example

```python
from deepstats import get_dgp, get_family, naive, influence

# Generate binary outcome data
dgp = get_dgp("logit", d=10, seed=42)
data = dgp.generate(n=2000)

print(f"True mu* = {data.mu_true:.6f}")
print(f"Outcome mean = {data.Y.mean():.3f}")

# Get family
family = get_family("logit")

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

## Expected Results

From Monte Carlo validation:

| Method | Coverage | SE Ratio | RMSE |
|--------|----------|----------|------|
| Naive | 3% | 0.03 | 0.108 |
| **Influence** | **90%** | **0.93** | 0.054 |

### Interpretation

- **Naive**: Nearly zero coverage - severely underestimates SE
- **Influence**: Near-target coverage with well-calibrated SE

## Alternative Targets

The logit family supports multiple target estimands:

### Log-Odds Ratio (Default)

```python
family = get_family("logit", target="beta")
# mu* = E[beta(X)] = average log-odds ratio
```

### Average Marginal Effect (AME)

```python
family = get_family("logit", target="ame")
# mu* = E[p(1-p) * beta(X)] = effect on probability scale
```

The AME tells you the average effect on the probability of the outcome, accounting for the nonlinear link function.

## Real-World Applications

### Credit Default

```python
# Y = 1 if default, 0 otherwise
# T = loan amount
# X = (income, credit score, employment, ...)
# Target: E[beta(X)] = average effect of loan size on default risk
```

### Market Entry

```python
# Y = 1 if firm enters market, 0 otherwise
# T = competitor count
# X = (market size, firm characteristics, ...)
# Target: E[beta(X)] = average effect of competition on entry
```

### Treatment Uptake

```python
# Y = 1 if patient takes treatment, 0 otherwise
# T = out-of-pocket cost
# X = (age, insurance status, condition severity, ...)
# Target: E[beta(X)] = average price sensitivity
```

## Numerical Considerations

### Hessian Stability

The weight $W = p(1-p)$ can be small when predictions are extreme ($p$ near 0 or 1). The package includes Tikhonov regularization for stable Hessian inversion:

$$\Lambda^{-1} = (\Lambda + \lambda I)^{-1}$$

### Checking Diagnostics

```python
# After running inference, check:
# - min(Lambda eigenvalue) > 1e-4 for stability
# - R_corr in [0.1, 1.0] for reasonable correction magnitude
```

## Key Takeaways

1. **Weight varies by p**: Observations with $p \approx 0.5$ get more weight
2. **Log-odds vs AME**: Choose target based on research question
3. **Hessian stability**: Check condition number in diagnostics
4. **Coverage slightly below 95%**: Logit is harder than linear, ~90% is good
