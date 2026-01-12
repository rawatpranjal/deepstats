# Families Module

Statistical families for structural estimation.

## Factory Function

```python
from deep_inference import get_family

# Get a family by name
family = get_family("linear")
family = get_family("logit")
```

## Available Families

```python
from deep_inference import FAMILY_REGISTRY
print(list(FAMILY_REGISTRY.keys()))
# ['linear', 'logit', 'poisson', 'tobit', 'negbin', 'gamma', 'gumbel', 'weibull']
```

## Base Class

All families inherit from `BaseFamily` and implement:

| Method | Description |
|--------|-------------|
| `loss(y, t, theta)` | Per-observation loss |
| `gradient(y, t, theta)` | Gradient w.r.t. theta |
| `hessian(y, t, theta)` | Hessian w.r.t. theta |
| `hessian_depends_on_theta()` | Whether Hessian depends on theta |
| `per_obs_target(theta, t)` | Per-observation target h(theta) |
| `per_obs_target_gradient(theta, t)` | Gradient of target |

## Family Classes

### LinearFamily

```python
from deep_inference import LinearFamily

family = LinearFamily()
# Model: Y = alpha(X) + beta(X) * T + epsilon
# Loss: MSE
# Hessian: constant (2-way splitting)
```

### LogitFamily

```python
from deep_inference import LogitFamily

# Default: log-odds target
family = LogitFamily(target='beta')

# Alternative: average marginal effect
family = LogitFamily(target='ame')

# Model: P(Y=1) = sigmoid(alpha(X) + beta(X) * T)
# Loss: Binary cross-entropy
# Hessian: depends on theta (3-way splitting)
```

### PoissonFamily

```python
from deep_inference import PoissonFamily

family = PoissonFamily()
# Model: Y ~ Poisson(exp(alpha(X) + beta(X) * T))
# Loss: Poisson deviance
# Hessian: depends on theta
```

### TobitFamily

```python
from deep_inference import TobitFamily

# Default: effect on latent Y*
family = TobitFamily(target='latent')

# Alternative: effect on observed E[Y]
family = TobitFamily(target='observed')

# Model: Y = max(0, alpha(X) + beta(X) * T + sigma(X) * epsilon)
# theta_dim: 3 (alpha, beta, gamma=log(sigma))
```

### NegBinFamily

```python
from deep_inference import NegBinFamily

family = NegBinFamily(overdispersion=0.5)
# Model: Y ~ NegBin(mu, r) where mu = exp(alpha + beta * T)
# Use for overdispersed count data
```

### GammaFamily

```python
from deep_inference import GammaFamily

family = GammaFamily(shape=1.0)
# Model: Y ~ Gamma(shape, exp(alpha + beta * T))
# Use for positive, skewed outcomes
```

### GumbelFamily

```python
from deep_inference import GumbelFamily

family = GumbelFamily(scale=1.0)
# Model: Y ~ Gumbel(alpha + beta * T, scale)
# Use for extreme value analysis
```

### WeibullFamily

```python
from deep_inference import WeibullFamily

family = WeibullFamily(shape=1.0)
# Model: Y ~ Weibull(shape, exp(alpha + beta * T))
# Use for duration/survival analysis
```

## Usage Example

```python
from deep_inference import structural_dml, LogitFamily

# Using family name (string)
result = structural_dml(Y, T, X, family='logit')

# Using family instance (for custom options)
family = LogitFamily(target='ame')
result = structural_dml(Y, T, X, family=family)
```

## Family Methods

Each family provides closed-form implementations for efficiency:

```python
import torch
from deep_inference import LinearFamily

family = LinearFamily()
n = 100

# Create tensors
y = torch.randn(n)
t = torch.randn(n)
theta = torch.randn(n, 2)  # [alpha, beta]

# Compute quantities
loss = family.loss(y, t, theta)           # (n,)
grad = family.gradient(y, t, theta)       # (n, 2)
hess = family.hessian(y, t, theta)        # (n, 2, 2)
h = family.per_obs_target(theta, t)       # (n,)
dh = family.per_obs_target_gradient(theta, t)  # (n, 2)
```
