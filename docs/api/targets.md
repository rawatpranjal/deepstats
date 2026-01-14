# Targets Module

The `targets` module defines what quantity you want to estimate. A **target** is a functional H(θ) that maps the structural parameters to a scalar.

## Overview

| Target | Formula | Use Case |
|--------|---------|----------|
| `AverageParameter` | E[θ_j] | Average treatment effect |
| `AME` | E[p(1-p)·β] | Marginal effect on probability |
| `CustomTarget` | User-defined h(x,θ,t̃) | Any custom functional |

---

## Built-in Targets

### AverageParameter

The default target: average of a specific parameter across the population.

```python
from deep_inference.targets import AverageParameter

# Target: E[β(X)] where β is the second parameter (index=1)
target = AverageParameter(param_index=1, theta_dim=2)
```

**Formula:**
```
H(θ) = (1/n) Σ θ_j(x_i)
```

**Jacobian (closed-form):**
```
∂H/∂θ = [0, ..., 1/n, ..., 0]  (1 at position j)
```

### AME (Average Marginal Effect)

For logit models, the marginal effect on probability (not log-odds).

```python
from deep_inference.targets import AME

# Target: E[p(1-p)·β] evaluated at t_tilde
target = AME(param_index=1, model_type='logit')
```

**Formula (logit):**
```
H(θ, t̃) = (1/n) Σ σ'(α_i + β_i·t̃) · β_i
        = (1/n) Σ p_i(1-p_i) · β_i
```

Where p_i = σ(α_i + β_i·t̃).

**Jacobian (closed-form for logit):**
```
∂H/∂α = p(1-p)(1-2p)·β / n
∂H/∂β = p(1-p)[1 + (1-2p)·β·t̃] / n
```

---

## Custom Targets

Define any target function and the Jacobian is computed via autodiff.

### CustomTarget

```python
from deep_inference.targets import CustomTarget
import torch

def my_target(x, theta, t_tilde):
    """
    Custom target function.

    Args:
        x: Covariates (unused for average targets)
        theta: (theta_dim,) parameter vector
        t_tilde: Evaluation point for treatment

    Returns:
        Scalar value
    """
    alpha, beta = theta[0], theta[1]
    return torch.sigmoid(alpha + beta * t_tilde)

target = CustomTarget(h_fn=my_target)
```

### Example: Average Prediction

```python
import torch
from deep_inference import inference

def avg_prediction(x, theta, t_tilde):
    """E[P(Y=1|T=t̃)] = E[σ(α + β·t̃)]"""
    return torch.sigmoid(theta[0] + theta[1] * t_tilde)

result = inference(
    Y, T, X,
    model='logit',
    target_fn=avg_prediction,
    t_tilde=0.0  # Prediction at T=0
)
```

### Example: Counterfactual Comparison

```python
def treatment_effect(x, theta, t_tilde):
    """E[P(Y=1|T=1) - P(Y=1|T=0)]"""
    alpha, beta = theta[0], theta[1]
    p1 = torch.sigmoid(alpha + beta * 1.0)
    p0 = torch.sigmoid(alpha + beta * 0.0)
    return p1 - p0

result = inference(
    Y, T, X,
    model='logit',
    target_fn=treatment_effect
)
```

---

## Target Protocol

All targets implement this interface:

```python
class Target(Protocol):
    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """Compute target value for a single observation."""
        ...

    def jacobian(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """Compute ∂h/∂θ. Falls back to autodiff if not implemented."""
        ...
```

### Implementing Custom Targets

```python
from deep_inference.targets import BaseTarget
import torch

class MyTarget(BaseTarget):
    def h(self, x, theta, t_tilde):
        # Your target computation
        return theta[0] ** 2 + theta[1] * t_tilde

    def jacobian(self, x, theta, t_tilde):
        # Optional: closed-form Jacobian (faster than autodiff)
        return torch.tensor([2 * theta[0], t_tilde])
```

---

## Using Targets with inference()

### Built-in Target Strings

```python
# Average beta (log-odds for logit)
result = inference(Y, T, X, model='logit', target='beta')

# Average marginal effect
result = inference(Y, T, X, model='logit', target='ame', t_tilde=0.0)
```

### Custom Target Functions

```python
def my_target(x, theta, t_tilde):
    return theta[0] + theta[1] * t_tilde

result = inference(Y, T, X, model='logit', target_fn=my_target, t_tilde=1.0)
```

---

## Autodiff Jacobian

When you provide a custom target function, the package automatically computes the Jacobian via PyTorch autodiff:

```python
# Internally, this happens:
theta.requires_grad_(True)
h_value = target_fn(x, theta, t_tilde)
jacobian = torch.autograd.grad(h_value, theta)[0]
```

This enables arbitrary differentiable targets without manual derivatives.
