# Models Module

Neural network architectures for structural estimation.

## Network Classes

### StructuralNet

The main neural network architecture for structural parameter estimation.

```python
from deep_inference.models import StructuralNet

# Create network
net = StructuralNet(
    input_dim=10,           # Number of covariates
    hidden_dims=[64, 32],   # Hidden layer sizes
    theta_dim=2,            # Number of parameters (alpha, beta)
    dropout=0.1             # Dropout rate
)

# Forward pass
import torch
X = torch.randn(100, 10)
theta = net(X)  # (100, 2)
```

## Network Architecture

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
Linear(hidden_dims[-1], theta_dim)
    |
Output (theta_dim parameters per observation)
```

## Usage with structural_dml

The `structural_dml` function creates and trains the network internally:

```python
from deep_inference import structural_dml

result = structural_dml(
    Y=Y, T=T, X=X,
    family='linear',
    hidden_dims=[64, 32],  # Network architecture
    epochs=100,            # Training epochs
    lr=0.01               # Learning rate
)

# Access estimated parameters
theta_hat = result.theta_hat  # (n, theta_dim) numpy array
alpha_hat = theta_hat[:, 0]
beta_hat = theta_hat[:, 1]
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

## Architecture Guidelines

| Sample Size | Recommended Architecture |
|-------------|-------------------------|
| n < 1,000 | `[32, 16]` |
| 1,000 < n < 10,000 | `[64, 32]` |
| 10,000 < n < 100,000 | `[128, 64, 32]` |
| n > 100,000 | `[256, 128, 64]` |

## Custom Network Usage

For advanced users who want to use the network directly:

```python
import torch
import torch.nn as nn
from deep_inference.models import StructuralNet
from deep_inference import LinearFamily

# Create network and family
net = StructuralNet(input_dim=10, hidden_dims=[64, 32], theta_dim=2)
family = LinearFamily()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    theta = net(X_tensor)
    loss = family.loss(Y_tensor, T_tensor, theta).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
