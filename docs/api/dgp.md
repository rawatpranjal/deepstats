# DGP Module

Data Generating Processes for Monte Carlo simulation.

## Factory Function

```{eval-rst}
.. autofunction:: deepstats.get_dgp
```

## Available DGPs

```{eval-rst}
.. autodata:: deepstats.DGPS
```

## Base Class

```{eval-rst}
.. autoclass:: deepstats.dgp.BaseDGP
   :members:
   :undoc-members:
   :show-inheritance:
```

## DGP Classes

### LinearDGP

```{eval-rst}
.. autoclass:: deepstats.dgp.LinearDGP
   :members:
   :show-inheritance:
```

### LogitDGP

```{eval-rst}
.. autoclass:: deepstats.dgp.LogitDGP
   :members:
   :show-inheritance:
```

### PoissonDGP

```{eval-rst}
.. autoclass:: deepstats.dgp.PoissonDGP
   :members:
   :show-inheritance:
```

### TobitDGP

```{eval-rst}
.. autoclass:: deepstats.dgp.TobitDGP
   :members:
   :show-inheritance:
```

### NegBinDGP

```{eval-rst}
.. autoclass:: deepstats.dgp.NegBinDGP
   :members:
   :show-inheritance:
```

### GammaDGP

```{eval-rst}
.. autoclass:: deepstats.dgp.GammaDGP
   :members:
   :show-inheritance:
```

### GumbelDGP

```{eval-rst}
.. autoclass:: deepstats.dgp.GumbelDGP
   :members:
   :show-inheritance:
```

### WeibullDGP

```{eval-rst}
.. autoclass:: deepstats.dgp.WeibullDGP
   :members:
   :show-inheritance:
```

## Usage Example

```python
from deepstats import get_dgp

# Create a DGP
dgp = get_dgp("linear", d=10, seed=42)

# Generate data
data = dgp.generate(n=1000)

# Access generated data
X = data.X      # Covariates (N x d)
T = data.T      # Treatment (N,)
Y = data.Y      # Outcome (N,)
mu_true = data.mu_true  # True average effect

# Verify ground truth
dgp.verify_ground_truth(n_mc=100000)
```

## Common Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `d` | int | Number of signal covariates (default: 10) |
| `n_noise` | int | Number of noise covariates (default: 10) |
| `seed` | int | Random seed for reproducibility |
| `sigma` | float | Error standard deviation (model-specific) |
