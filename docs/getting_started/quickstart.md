# Quickstart

This guide shows you how to use `deepstats` to estimate treatment effects with valid inference.

## Basic Workflow

### 1. Generate Data

```python
from deepstats import get_dgp

# Create a data generating process
dgp = get_dgp("linear", seed=42)

# Generate N observations
data = dgp.generate(n=2000)

# data contains:
# - data.X: Covariates (N x d)
# - data.T: Treatment (N,)
# - data.Y: Outcome (N,)
# - data.mu_true: True average treatment effect
```

### 2. Select a Family

```python
from deepstats import get_family

# Linear family for continuous outcomes
family = get_family("linear")
```

### 3. Run Inference

```python
from deepstats import influence

result = influence(
    X=data.X,
    T=data.T,
    Y=data.Y,
    family=family,
    config={
        "hidden_dims": [64, 32],
        "epochs": 50,
        "n_folds": 50,
        "lr": 0.01
    }
)
```

### 4. Interpret Results

```python
print(f"True effect:  {data.mu_true:.4f}")
print(f"Estimate:     {result.mu_hat:.4f}")
print(f"Std Error:    {result.se:.4f}")
print(f"95% CI:       [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dims` | `[64, 32]` | Network architecture |
| `epochs` | `100` | Training epochs |
| `n_folds` | `50` | Cross-fitting folds |
| `lr` | `0.01` | Learning rate |
| `batch_size` | `64` | Mini-batch size |
| `weight_decay` | `1e-4` | L2 regularization |

## Comparing Methods

```python
from deepstats import naive, influence, bootstrap

# Naive (undercovers)
naive_result = naive(data.X, data.T, data.Y, family, config)

# Influence function (valid coverage)
if_result = influence(data.X, data.T, data.Y, family, config)

# Bootstrap (partial correction)
boot_result = bootstrap(data.X, data.T, data.Y, family, config)

print(f"Naive coverage:     ~10-30%")
print(f"Bootstrap coverage: ~70-85%")
print(f"Influence coverage: ~95%")
```

## Next Steps

- See [Tutorials](../tutorials/index.md) for detailed examples with each model
- Read [Theory](../theory/index.md) for the mathematical background
- Check [API Reference](../api/index.md) for complete documentation
