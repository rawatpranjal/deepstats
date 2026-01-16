# Flowchart: Choosing Your Model

Use this decision tree to select the right model family, target functional, and inference regime for your data.

## Interactive Decision Flowchart

```{mermaid}
flowchart TD
    subgraph START [" "]
        A([I have data Y, T, X])
    end

    A --> Q1{What type of<br/>outcome Y?}

    %% Outcome type branching
    Q1 -->|Continuous<br/>unbounded| LINEAR[family='linear']
    Q1 -->|Binary 0/1| LOGIT[family='logit']
    Q1 -->|Count 0,1,2...| COUNT{Overdispersed?}
    Q1 -->|Positive<br/>continuous| POS{Distribution shape?}
    Q1 -->|Censored| TOBIT[family='tobit']

    COUNT -->|No| POISSON[family='poisson']
    COUNT -->|Yes| NEGBIN[family='negbin']

    POS -->|Multiplicative errors| GAMMA[family='gamma']
    POS -->|Extreme values| GUMBEL[family='gumbel']
    POS -->|Time-to-event| WEIBULL[family='weibull']

    %% Target selection
    LINEAR --> T1{Target functional?}
    LOGIT --> T2{Target functional?}
    POISSON --> T3[target='beta']
    NEGBIN --> T3
    GAMMA --> T3
    GUMBEL --> T3
    WEIBULL --> T3
    TOBIT --> T4{Target functional?}

    T1 -->|Average parameter| BETA_LIN[target='beta']
    T1 -->|Custom function| CUSTOM[target_fn=...]

    T2 -->|Average parameter| BETA_LOG[target='beta']
    T2 -->|Average marginal effect| AME[target='ame']
    T2 -->|Custom function| CUSTOM

    T4 -->|Latent Y*| LATENT[target='latent']
    T4 -->|Observed Y| OBS[target='observed']

    %% Regime detection
    BETA_LIN --> R_B[Regime B<br/>Analytic Lambda]
    BETA_LOG --> R1{Treatment<br/>randomized?}
    AME --> R1
    T3 --> R1
    LATENT --> R_C
    OBS --> R_C
    CUSTOM --> R_C[Regime C<br/>Estimate Lambda]

    R1 -->|Yes + known distribution| R_A[Regime A<br/>Compute Lambda]
    R1 -->|No / observational| R_C

    %% Final API calls
    R_A --> API_A["inference(...,<br/>is_randomized=True,<br/>treatment_dist=Normal())"]
    R_B --> API_B["structural_dml(...,<br/>family='linear')"]
    R_C --> API_C["structural_dml(...,<br/>family='logit')"]

    %% Styling
    classDef question fill:#fff3cd,stroke:#856404
    classDef model fill:#d4edda,stroke:#155724
    classDef target fill:#cce5ff,stroke:#004085
    classDef regime fill:#f8d7da,stroke:#721c24
    classDef api fill:#e2e3e5,stroke:#383d41

    class Q1,COUNT,POS,T1,T2,T4,R1 question
    class LINEAR,LOGIT,POISSON,NEGBIN,GAMMA,GUMBEL,WEIBULL,TOBIT model
    class BETA_LIN,BETA_LOG,AME,LATENT,OBS,T3,CUSTOM target
    class R_A,R_B,R_C regime
    class API_A,API_B,API_C api
```

---

## Decision 1: Model Family

Choose based on your outcome variable type:

| Outcome Type | Family | Link Function | Example Use Cases |
|--------------|--------|---------------|-------------------|
| Continuous (unbounded) | `linear` | Identity | Wages, prices, test scores |
| Binary (0/1) | `logit` | Logit | Purchase decisions, clicks, conversions |
| Count (0, 1, 2, ...) | `poisson` | Log | Website visits, order counts |
| Overdispersed count | `negbin` | Log | Insurance claims, rare events |
| Positive continuous | `gamma` | Log | Expenditures, durations |
| Extreme values | `gumbel` | Identity | Maximum temperatures, flood levels |
| Time-to-event | `weibull` | Log | Survival times, equipment failure |
| Censored | `tobit` | Identity | Demand with stockouts |

### How to Check for Overdispersion

For count data, compare the variance to the mean:

```python
variance = np.var(Y)
mean = np.mean(Y)
overdispersion_ratio = variance / mean

if overdispersion_ratio > 1.5:
    print("Use family='negbin'")
else:
    print("Use family='poisson'")
```

---

## Decision 2: Target Functional

The target functional determines what quantity you're estimating:

### `target='beta'` (Default)

Estimates the average structural parameter: $\mu^* = \mathbb{E}[\beta(X)]$

```python
result = structural_dml(Y, T, X, family='logit', target='beta')
# result.mu_hat estimates E[beta(X)]
```

### `target='ame'` (Average Marginal Effect)

For binary outcomes, estimates the effect on probability scale:

$$\text{AME} = \mathbb{E}\left[\frac{\partial P(Y=1|X,T)}{\partial T}\Big|_{T=\tilde{t}}\right]$$

```python
result = inference(Y, T, X, model='logit', target='ame', t_tilde=0.0)
# result.mu_hat estimates the AME at T=0
```

### `target_fn=...` (Custom Target)

Define any target functional with automatic Jacobian computation:

```python
import torch

def my_target(x, theta, t_tilde):
    """Custom target: probability at treatment level t_tilde."""
    alpha, beta = theta[..., 0], theta[..., 1]
    eta = alpha + beta * t_tilde
    return torch.sigmoid(eta)

result = inference(Y, T, X, model='logit', target_fn=my_target, t_tilde=0.5)
```

---

## Decision 3: Regime & Lambda Strategy

The regime determines how $\Lambda(x) = \mathbb{E}[\ell_{\theta\theta}|X=x]$ is obtained:

| Regime | Condition | Lambda Strategy | Cross-Fitting |
|--------|-----------|-----------------|---------------|
| **A** | RCT with known $F_T$ | Compute via MC integration | 2-way |
| **B** | Linear model | Analytic closed-form | 2-way |
| **C** | Observational, nonlinear | Estimate with ridge regression | 3-way |

### Regime A: Randomized Experiments

When treatment is randomized and you know the distribution:

```python
from deep_inference import inference
from deep_inference.lambda_.compute import Normal

result = inference(
    Y, T, X,
    model='logit',
    target='beta',
    is_randomized=True,
    treatment_dist=Normal(mean=0, std=1)
)
```

### Regime B: Linear Models

For linear models, Lambda has a closed-form solution:

```python
result = structural_dml(Y, T, X, family='linear')
# Automatically uses analytic Lambda
```

### Regime C: Observational Studies

For nonlinear models with observational data (default):

```python
result = structural_dml(Y, T, X, family='logit')
# Uses ridge regression to estimate Lambda (default)
```

---

## Common Paths

### Path 1: Logistic Demand (Observational)

Binary purchase decisions with heterogeneous price sensitivity:

```python
import numpy as np
from deep_inference import structural_dml

# Data: Y = purchase (0/1), T = price, X = customer features
result = structural_dml(
    Y=Y, T=T, X=X,
    family='logit',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50
)

print(result.summary())
# mu_hat = average price sensitivity E[beta(X)]
```

**Path through flowchart**: Binary outcome → `logit` → `target='beta'` → Observational → Regime C → `structural_dml()`

### Path 2: RCT with Binary Outcome

A/B test where treatment assignment is known:

```python
from deep_inference import inference
from deep_inference.lambda_.compute import Bernoulli

result = inference(
    Y=Y, T=T, X=X,
    model='logit',
    target='ame',
    t_tilde=0.0,
    is_randomized=True,
    treatment_dist=Bernoulli(p=0.5)
)

print(result.summary())
# mu_hat = average marginal effect on P(Y=1)
```

**Path through flowchart**: Binary outcome → `logit` → `target='ame'` → Randomized → Regime A → `inference()`

### Path 3: Linear Model

Continuous outcome with linear structure:

```python
result = structural_dml(
    Y=Y, T=T, X=X,
    family='linear',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50
)

print(result.summary())
# mu_hat = E[beta(X)], the average treatment effect
```

**Path through flowchart**: Continuous outcome → `linear` → `target='beta'` → Regime B → `structural_dml()`

### Path 4: Custom Loss + Custom Target

For non-standard models, define your own loss and target:

```python
import torch
from deep_inference import inference

def custom_loss(y, t, theta):
    """Custom negative log-likelihood."""
    alpha, beta = theta[..., 0], theta[..., 1]
    mu = alpha + beta * t
    return (y - mu) ** 2  # Example: squared error

def custom_target(x, theta, t_tilde):
    """Custom target functional."""
    return theta[..., 1]  # Just return beta

result = inference(
    Y=Y, T=T, X=X,
    loss_fn=custom_loss,
    target_fn=custom_target,
    theta_dim=2
)
```

**Path through flowchart**: Custom → `target_fn=...` → Regime C → `inference()`

---

## Quick Reference

### Parameter Cheat Sheet

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `family` | str | `'linear'` | Model family: linear, logit, poisson, etc. |
| `target` | str | `'beta'` | Target functional: beta, ame, latent, observed |
| `target_fn` | callable | None | Custom target function |
| `hidden_dims` | list | `[64, 32]` | Neural network hidden layer sizes |
| `epochs` | int | 100 | Training epochs |
| `n_folds` | int | 50 | Number of cross-fitting folds |
| `lambda_method` | str | `'ridge'` | Lambda estimation: ridge, lgbm, aggregate |
| `is_randomized` | bool | False | Whether treatment is randomized |
| `treatment_dist` | object | None | Treatment distribution for Regime A |
| `t_tilde` | float | 0.0 | Evaluation point for AME |

### API Summary

```python
# Legacy API (recommended for most use cases)
from deep_inference import structural_dml
result = structural_dml(Y, T, X, family='logit', ...)

# New API (for custom targets and regimes)
from deep_inference import inference
result = inference(Y, T, X, model='logit', target='ame', ...)
```

---

## See Also

- [Getting Started](getting_started/index.md) - Installation and first steps
- [Tutorials](tutorials/index.md) - Detailed worked examples
- [Theory](theory/index.md) - Mathematical foundations
- [Algorithm](algorithm/index.md) - Implementation details
- [API Reference](api/index.md) - Full API documentation
