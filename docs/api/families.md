# Families Module

Statistical families for structural estimation.

## Factory Function

```{eval-rst}
.. autofunction:: deepstats.get_family
```

## Available Families

```{eval-rst}
.. autodata:: deepstats.FAMILIES
```

## Base Class

```{eval-rst}
.. autoclass:: deepstats.families.BaseFamily
   :members:
   :undoc-members:
   :show-inheritance:
```

## Family Classes

### LinearFamily

```{eval-rst}
.. autoclass:: deepstats.families.LinearFamily
   :members:
   :show-inheritance:
```

### LogitFamily

```{eval-rst}
.. autoclass:: deepstats.families.LogitFamily
   :members:
   :show-inheritance:
```

### PoissonFamily

```{eval-rst}
.. autoclass:: deepstats.families.PoissonFamily
   :members:
   :show-inheritance:
```

### TobitFamily

```{eval-rst}
.. autoclass:: deepstats.families.TobitFamily
   :members:
   :show-inheritance:
```

### NegBinFamily

```{eval-rst}
.. autoclass:: deepstats.families.NegBinFamily
   :members:
   :show-inheritance:
```

### GammaFamily

```{eval-rst}
.. autoclass:: deepstats.families.GammaFamily
   :members:
   :show-inheritance:
```

### GumbelFamily

```{eval-rst}
.. autoclass:: deepstats.families.GumbelFamily
   :members:
   :show-inheritance:
```

### WeibullFamily

```{eval-rst}
.. autoclass:: deepstats.families.WeibullFamily
   :members:
   :show-inheritance:
```

## Usage Example

```python
from deepstats import get_family

# Get a family
family = get_family("linear")

# Family provides:
# - family.loss(Y, T, theta)    : Loss function
# - family.residual(Y, T, theta): Score residuals
# - family.weight(Y, T, theta)  : Hessian weights
# - family.influence_score(...) : Full influence score
```

## Family Methods

Each family implements:

| Method | Description |
|--------|-------------|
| `loss(Y, T, theta)` | Negative log-likelihood |
| `residual(Y, T, theta)` | Score residuals $r_i$ |
| `weight(Y, T, theta)` | Hessian weights $W_i$ |
| `influence_score(...)` | Complete influence function $\psi_i$ |
| `n_params` | Number of parameters per observation |

## Family-Specific Options

### Logit

```python
# Log-odds ratio (default)
family = get_family("logit", target="beta")

# Average marginal effect
family = get_family("logit", target="ame")
```

### Tobit

```python
# Effect on latent Y* (default)
family = get_family("tobit", target="latent")

# Effect on observed E[Y]
family = get_family("tobit", target="observed")
```
