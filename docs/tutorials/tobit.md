# Tobit Model Tutorial

The Tobit model handles censored data where outcomes pile up at a boundary.

## When to Use

Use the Tobit model when:
- Outcome is censored at a boundary (typically 0)
- You observe the censored value, not the latent value
- Examples: labor supply (hours >= 0), expenditure, donations

## Mathematical Setup

### Data Generating Process

$$Y^* = \alpha(X) + \beta(X) \cdot T + \varepsilon, \quad \varepsilon \sim N(0, \sigma^2)$$

$$Y = \max(0, Y^*)$$

Where $Y^*$ is the latent (unobserved) variable and $Y$ is the observed (censored) outcome.

### Estimand

$$\mu^* = E[\beta(X)]$$

The average effect on the **latent** outcome.

### Loss Function

The Tobit likelihood has two parts:

1. **Censored observations** ($Y = 0$): Probability of $Y^* \leq 0$
2. **Uncensored observations** ($Y > 0$): Normal density

$$L = -\sum_{Y_i=0} \log \Phi\left(-\frac{\mu_i}{\sigma}\right) - \sum_{Y_i>0} \left[\log \phi\left(\frac{Y_i - \mu_i}{\sigma}\right) - \log \sigma\right]$$

### Influence Score Components

| Component | Formula |
|-----------|---------|
| Residual $r$ | Mills ratio (censored) or $(Y-\mu)/\sigma$ (uncensored) |
| Hessian weight $W$ | $1 - \Phi(-\mu/\sigma)$ |
| Score $\nabla\ell$ | Varies by censoring status |

The **Mills ratio** is $\phi(z)/\Phi(z)$ where $z = -\mu/\sigma$.

## Complete Example

```python
import numpy as np
from deep_inference import structural_dml

# Generate censored data
np.random.seed(42)
n = 2000
X = np.random.randn(n, 10)
T = np.random.randn(n)

# True structural functions
alpha_true = 0.5 + 0.3 * X[:, 0]
beta_true = 0.3 + 0.2 * X[:, 0]
sigma = 1.0
Y_star = alpha_true + beta_true * T + sigma * np.random.randn(n)
Y = np.maximum(0, Y_star)  # Censoring at 0
mu_true = beta_true.mean()

# Check censoring rate
censored_pct = (Y == 0).mean() * 100
print(f"True mu* = {mu_true:.6f}")
print(f"Censored at 0: {censored_pct:.1f}%")

# Run inference
result = structural_dml(
    Y=Y, T=T, X=X,
    family='tobit',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50,
    lr=0.01
)

print(result.summary())
```

## Alternative Targets

### Latent Effect (Default)

```python
# Default target is effect on latent Y*
result = structural_dml(Y, T, X, family='tobit')
# mu* = E[beta(X)] = effect on latent Y*
```

### Observed Effect

For observed effect, use the TobitFamily class directly:

```python
from deep_inference import TobitFamily, structural_dml

# Create family with observed target
family = TobitFamily(target='observed')
result = structural_dml(Y, T, X, family=family)
# mu* = E[beta(X) * Phi(mu/sigma)] = effect on observed E[Y]
```

The observed effect accounts for the probability of being uncensored.

## Parameter Structure

The Tobit model estimates **three** parameters per observation:

$$\theta(X) = [\alpha(X), \beta(X), \gamma(X)]$$

Where $\sigma(X) = \exp(\gamma(X))$ is the conditional variance.

## Real-World Applications

### Labor Supply

```python
# Y = hours worked (>= 0)
# T = wage rate
# X = (education, family size, non-labor income, ...)
# Target: E[beta(X)] = average labor supply elasticity

result = structural_dml(Y, T, X, family='tobit')
```

### Charitable Donations

```python
# Y = donation amount (>= 0)
# T = match rate offered
# X = (income, past giving, solicitation type, ...)
# Target: E[beta(X)] = average matching effect

result = structural_dml(Y, T, X, family='tobit')
```

### Durable Goods Expenditure

```python
# Y = spending on cars (many zeros)
# T = income change
# X = (current car age, household size, ...)
# Target: E[beta(X)] = average income effect on car spending

result = structural_dml(Y, T, X, family='tobit')
```

## Handling Different Censoring

### Left-censoring at 0 (Default)

```python
result = structural_dml(Y, T, X, family='tobit')  # Assumes Y >= 0
```

### Right-censoring

```python
# For data censored from above (e.g., top-coded income)
# Transform: Y_new = upper_bound - Y
# Then use standard Tobit
```

### Two-sided censoring

```python
# For data censored at both ends
# Requires custom implementation
```

## Key Takeaways

1. **Latent vs observed**: Choose target based on research question
2. **Mills ratio**: Key ingredient for censored observations
3. **Joint sigma estimation**: Model estimates conditional variance
4. **Check censoring rate**: Very high (>50%) or low (<10%) censoring can cause issues
5. **Three parameters**: alpha, beta, and gamma (log-sigma)
