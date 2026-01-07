# Tutorials

Step-by-step tutorials for each supported model family.

```{toctree}
:maxdepth: 2

linear
logit
poisson
tobit
negbin
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

## Common Workflow

All tutorials follow the same basic pattern:

```python
from deepstats import get_dgp, get_family, influence

# 1. Get data (synthetic or real)
dgp = get_dgp("MODEL_NAME")
data = dgp.generate(n=2000)

# 2. Get family
family = get_family("MODEL_NAME")

# 3. Run inference
result = influence(data.X, data.T, data.Y, family, config)

# 4. Check results
print(f"Coverage should be ~95%")
print(f"SE ratio should be ~1.0")
```
