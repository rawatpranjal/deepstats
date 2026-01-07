# Models Module

Neural network architectures for structural estimation.

## Network Classes

### StructuralNet

```{eval-rst}
.. autoclass:: deepstats.StructuralNet
   :members:
   :undoc-members:
   :show-inheritance:
```

### NuisanceNet

```{eval-rst}
.. autoclass:: deepstats.NuisanceNet
   :members:
   :undoc-members:
   :show-inheritance:
```

## Training Functions

```{eval-rst}
.. autofunction:: deepstats.train_structural
```

```{eval-rst}
.. autofunction:: deepstats.train_nuisance
```

## Usage Example

```python
import torch
from deepstats import StructuralNet, NuisanceNet

# Structural network: X -> [alpha(X), beta(X)]
structural_net = StructuralNet(
    input_dim=10,
    hidden_dims=[64, 32],
    n_params=2,  # alpha and beta
    dropout=0.1
)

# Nuisance network: X -> (E[T|X], Var[T|X])
nuisance_net = NuisanceNet(
    input_dim=10,
    hidden_dims=[32, 16]
)

# Forward pass
X = torch.randn(100, 10)
theta = structural_net(X)  # (100, 2)
t_mean, t_var = nuisance_net(X)  # (100,), (100,)
```

## Network Architecture

### StructuralNet

```
Input (d features)
    |
Linear(d, hidden_dims[0])
    |
ReLU + Dropout
    |
Linear(hidden_dims[0], hidden_dims[1])
    |
ReLU + Dropout
    |
...
    |
Linear(hidden_dims[-1], n_params)
    |
Output (n_params per observation)
```

### NuisanceNet

```
Input (d features)
    |
Linear(d, hidden_dims[0])
    |
ReLU
    |
...
    |
Linear(hidden_dims[-1], 2)
    |
    +-- mean (no activation)
    |
    +-- log_var -> softplus -> var
```

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dims` | `[64, 32]` | Hidden layer sizes |
| `epochs` | `100` | Training epochs |
| `lr` | `0.01` | Learning rate |
| `batch_size` | `64` | Mini-batch size |
| `weight_decay` | `1e-4` | L2 regularization |
| `dropout` | `0.1` | Dropout rate |
| `patience` | `10` | Early stopping patience |
