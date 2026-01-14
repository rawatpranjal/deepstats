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
from deep_inference import structural_dml

# Generate data with heterogeneous treatment effects
np.random.seed(42)
n = 2000
X = np.random.randn(n, 10)
T = np.random.randn(n)
Y = X[:, 0] + 0.5 * T + np.random.randn(n)

# Run influence function inference
result = structural_dml(
    Y=Y, T=T, X=X,
    family='linear',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50
)

print(f"Estimate: {result.mu_hat:.4f} +/- {result.se:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
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
