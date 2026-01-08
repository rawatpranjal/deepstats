# deepstats

```{raw} html
<p class="hero-tagline">
Deep Learning for Individual Heterogeneity with Valid Inference
</p>
```

`deepstats` enriches structural economic models with deep learning while maintaining valid statistical inference. It implements the Farrell, Liang, and Misra (2021, 2025) framework.

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
    <h3>PyTorch Backend</h3>
    <p>Automatic differentiation for exact gradients and Hessians</p>
  </div>
</div>
```

## Quick Start

```python
from deepstats import get_dgp, get_family, influence

# Generate data with heterogeneous treatment effects
dgp = get_dgp("linear", seed=42)
data = dgp.generate(n=2000)

# Run influence function inference
family = get_family("linear")
result = influence(data.X, data.T, data.Y, family, {"n_folds": 50})

print(f"Estimate: {result.mu_hat:.4f} +/- {result.se:.4f}")
```

## Why deepstats?

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
