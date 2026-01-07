# deepstats

**Deep Learning for Individual Heterogeneity**

`deepstats` is a Python package for **enriching structural economic models** with deep learning. It implements the framework developed by Farrell, Liang, and Misra (2021, 2025) to recover rich, non-linear parameter heterogeneity ($\theta(X)$) while maintaining the interpretability and validity of structural economics.

## The Problem

Standard deep learning minimizes prediction error, which leads to biased parameter estimates. This is "The Inference Trap" - neural networks are great at prediction but naive inference produces invalid confidence intervals with coverage far below the nominal 95%.

## The Solution

`deepstats` implements **Influence Function-based Debiasing** to provide valid confidence intervals and p-values for economic targets. The key insight: we can correct for regularization bias using the influence function from semiparametric statistics.

## Key Features

- **Valid Inference**: 95% confidence intervals that actually cover 95% of the time
- **Multiple Families**: Linear, Logit, Poisson, Tobit, Gamma, NegBin, Weibull
- **Cross-Fitting**: K-fold cross-fitting for bias correction
- **PyTorch Backend**: Automatic differentiation for exact gradients and Hessians

## Quick Example

```python
from deepstats import get_dgp, get_family, influence

# Generate synthetic data
dgp = get_dgp("linear", seed=42)
data = dgp.generate(n=2000)

# Get the statistical family
family = get_family("linear")

# Run influence function inference
result = influence(
    X=data.X, T=data.T, Y=data.Y,
    family=family,
    config={"epochs": 50, "n_folds": 50}
)

print(f"Estimate: {result.mu_hat:.4f} +/- {result.se:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

## Documentation

```{toctree}
:maxdepth: 2
:caption: Contents

getting_started/index
tutorials/index
theory/index
api/index
```

## References

- Farrell, Liang, Misra (2021): "Deep Neural Networks for Estimation and Inference" *Econometrica*
- Farrell, Liang, Misra (2025): "Deep Learning for Individual Heterogeneity" *Working Paper*

## License

MIT License
