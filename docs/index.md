# deep-inference

```{raw} html
<p class="hero-tagline">
Deep Learning for Individual Heterogeneity with Valid Inference
</p>
```

`deep-inference` enriches structural economic models with deep learning while maintaining valid statistical inference. It implements the Farrell, Liang, and Misra (2021, 2025) framework.

```{raw} html
<div class="feature-grid">
  <div class="feature-card">
    <h3>Valid Inference</h3>
    <p>95% confidence intervals that actually cover 95% of the time</p>
  </div>
  <div class="feature-card">
    <h3>8 Model Families</h3>
    <p>Linear, Logit, Poisson, Tobit, Gamma, NegBin, Weibull, Gumbel</p>
  </div>
  <div class="feature-card">
    <h3>Flexible Targets</h3>
    <p>AME, custom targets with autodiff Jacobians</p>
  </div>
  <div class="feature-card">
    <h3>Regime Detection</h3>
    <p>Auto-selects optimal Lambda strategy for RCTs vs observational data</p>
  </div>
  <div class="feature-card">
    <h3>PyTorch Backend</h3>
    <p>Automatic differentiation for exact gradients and Hessians</p>
  </div>
</div>
```

## Quick Start

```python
import numpy as np
import torch
from deep_inference import structural_dml

# Heterogeneous logistic demand (binary outcomes)
np.random.seed(42)
torch.manual_seed(42)
n = 2000
X = np.random.randn(n, 5)
T = np.random.randn(n)

# Heterogeneous treatment effect: β(X) = 0.5 + 0.3*X₁
alpha = 0.2 * X[:, 0]
beta = 0.5 + 0.3 * X[:, 1]
prob = 1 / (1 + np.exp(-(alpha + beta * T)))
Y = np.random.binomial(1, prob).astype(float)

# Run influence function inference
result = structural_dml(
    Y=Y, T=T, X=X,
    family='logit',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50
)

print(result.summary())
```

**Output** (Date/Time will vary):
```
==============================================================================
                            Structural DML Results
==============================================================================
Family:           Logit                Target:           E[beta]
No. Observations: 2000                 No. Folds:        50
Date:             Fri, 16 Jan 2026     Time:             13:54:12
==============================================================================
                  coef     std err         z     P>|z|      [0.025    0.975]
------------------------------------------------------------------------------
     E[beta]    0.4704      0.0496     9.492  0.000      0.3733    0.5675
==============================================================================
Diagnostics:
  Min Lambda eigenvalue:    0.134946
  Mean condition number:    1.17
  Correction ratio:         45.2493
  Pct regularized:          0.0%
------------------------------------------------------------------------------
```

### Predictions & Visualization

```python
# Predict treatment effects for new observations
X_new = np.random.randn(5, 5)
beta_new = result.predict_beta(X_new)
print(f"Predicted β(X) for new data: {beta_new}")

# Predict probabilities at treatment level T=1
proba = result.predict_proba(X_new, t_value=1.0)
print(f"P(Y=1|X,T=1): {proba}")

# Visualize heterogeneity distributions
result.plot_distributions()
result.plot_heterogeneity(feature_idx=1)  # β(X) vs X₁
```

### New `inference()` API

The new API supports flexible targets and randomization mode:

```python
from deep_inference import inference
from deep_inference.lambda_.compute import Normal

# Average Marginal Effect (probability scale)
result = inference(Y, T, X, model='logit', target='ame', t_tilde=0.0)

# Randomized experiment (compute Lambda instead of estimating)
result = inference(Y, T, X, model='logit', target='beta',
                   is_randomized=True, treatment_dist=Normal(0, 1))
```

## Why deep-inference?

**The Problem**: Neural networks are great at prediction but naive inference produces invalid confidence intervals with coverage far below 95%.

**The Solution**: Influence function-based debiasing corrects for regularization bias, providing valid confidence intervals for economic targets like average treatment effects.

| Method | Coverage | SE Ratio |
|--------|----------|----------|
| Naive | 8% | 0.27 |
| **Influence** | **95%** | **1.08** |

## Documentation

```{toctree}
:maxdepth: 2
:caption: Contents

getting_started/index
flowchart
tutorials/index
theory/index
algorithm/index
validation/index
api/index
```

## References

- Farrell, Liang, Misra (2021): "Deep Neural Networks for Estimation and Inference" *Econometrica*
- Farrell, Liang, Misra (2025): "Deep Learning for Individual Heterogeneity" *Working Paper*

## License

MIT License
