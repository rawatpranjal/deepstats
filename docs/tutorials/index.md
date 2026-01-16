# Tutorials

Step-by-step tutorials for each supported model family.

```{toctree}
:maxdepth: 2

gallery
linear
logit
poisson
tobit
negbin
gamma
gumbel
weibull
multimodal
```

## Overview

Each tutorial covers:

1. **When to use** - What types of data and research questions
2. **Mathematical setup** - The DGP, loss function, and estimand
3. **Code example** - Complete working code
4. **Results interpretation** - Understanding coverage and SE ratios
5. **Real-world applications** - Practical use cases

## Model Selection Guide

| Your Outcome | Distribution | Model | Tutorial |
|--------------|--------------|-------|----------|
| Continuous (unbounded) | Normal | Linear | [Linear](linear.md) |
| Binary (0/1) | Bernoulli | Logit | [Logit](logit.md) |
| Count (0, 1, 2, ...) | Poisson | Poisson | [Poisson](poisson.md) |
| Censored at boundary | Tobit | Tobit | [Tobit](tobit.md) |
| Overdispersed counts | Negative Binomial | NegBin | [NegBin](negbin.md) |
| Positive right-skewed | Gamma | Gamma | [Gamma](gamma.md) |
| Extreme values (maxima) | Gumbel | Gumbel | [Gumbel](gumbel.md) |
| Time-to-event | Weibull | Weibull | [Weibull](weibull.md) |
| **High-dim embeddings** | Any | Any | [Multimodal](multimodal.md) |

## Common Workflow

All tutorials follow the same basic pattern:

```python
from deep_inference import structural_dml
import numpy as np

# 1. Prepare your data
# Y: outcome variable
# T: treatment variable
# X: covariates (must be 2D array)

# 2. Run inference
result = structural_dml(
    Y=Y, T=T, X=X,
    family='linear',  # or 'logit', 'poisson', etc.
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50
)

# 3. Check results
print(f"Estimate: {result.mu_hat:.4f}")
print(f"SE: {result.se:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")

# Coverage should be ~95%, SE ratio should be ~1.0
```
