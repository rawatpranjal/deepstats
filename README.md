# `deepstats`

**Structural Deep Learning Models for Economics**

`deepstats` is a Python package for estimating structural economic models with heterogeneous parameters using Deep Learning. It implements the **one-step Newton correction** framework developed by Farrell, Liang, and Misra (2021, 2025) to provide statistically valid inference (confidence intervals and p-values) for complex economic targets.

This package allows you to move beyond rigid parametric assumptions and flexibly model how parameters like treatment effects, price elasticities, or risk preferences vary across individuals, while retaining the interpretability of a structural model.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Key Features

- **Flexible Heterogeneity:** Use PyTorch Neural Networks to model structural parameters (e.g., $\alpha(X), \beta(X)$) as arbitrary functions of covariates.
- **Valid Inference:** Automatically computes bias-corrected estimates and standard errors using a built-in influence function corrector.
- **K-Fold Cross-Fitting:** Uses 50-fold cross-fitting (98% train, 2% test per fold) following the FLM protocol.
- **Extensible:** Supports 8 standard econometric models and can be extended with custom targets.

## Installation

```bash
pip install -e .
```
*Requires Python 3.10+, PyTorch, NumPy, scikit-learn.*

## Quickstart

Estimate a linear model where the treatment effect $\beta$ varies with covariates $X$.

```python
from deepstats import get_dgp, get_family

# 1. Generate synthetic data
dgp = get_dgp("linear", seed=42)
data = dgp.generate(n=1000)

# 2. Get the family (defines loss, gradient, Hessian)
family = get_family("linear")

# 3. Run influence function inference
from deepstats import influence
from dataclasses import dataclass

@dataclass
class Config:
    epochs: int = 100
    lr: float = 0.01
    batch_size: int = 64
    hidden_dims: list = None
    dropout: float = 0.1
    weight_decay: float = 1e-4
    n_folds: int = 50
    seed: int = 42

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]

config = Config()
mu_hat, se = influence(data.X, data.T, data.Y, family, config)
print(f"ATE Estimate: {mu_hat:.4f} (SE: {se:.4f})")
print(f"True ATE: {data.mu_true:.4f}")
```

## How It Works: The Math

The goal is to estimate a parameter $\mu^* = \mathbb{E}[H(X, \theta^*(X))]$, where $\theta^*(X)$ are the true, heterogeneous structural parameters.

A naive plug-in estimator, $\hat{\mu}_{naive} = \frac{1}{N}\sum H(X_i, \hat{\theta}(X_i))$, is biased due to the regularization in the neural network estimator $\hat{\theta}$.

This package computes a **one-step bias correction** using the influence function. For each observation $i$, we compute a score $\psi_i$:

$$\psi_i = \underbrace{H(X_i, \hat{\theta}_i)}_{\text{Plug-in}} + \underbrace{\nabla_{\theta} H^\top \cdot \Lambda^{-1} \cdot \nabla_{\theta} \ell}_{\text{Correction}}$$

Where:
- $\hat{\theta}_i$: Parameters predicted by the neural network for observation $i$
- $\nabla_{\theta} \ell$: **Gradient** of the model's loss function (the "score")
- $\Lambda^{-1}$: Inverse **Hessian** of the loss (scales by curvature)
- $\nabla_{\theta} H$: **Jacobian** of the target w.r.t. parameters

The final estimate is:
$$\hat{\mu} = \frac{1}{N}\sum_{i=1}^{N} \psi_i, \quad SE = \frac{\text{std}(\psi)}{\sqrt{N}}$$

All nuisance functions are estimated on separate folds using K-fold cross-fitting.

## Supported Models (`families`)

The `family` parameter defines the structural loss, gradient, and Hessian.

| Family | Model Equation | Loss Function $\ell$ | Use Case |
| :--- | :--- | :--- | :--- |
| **Linear** | $Y = \alpha + \beta T + \varepsilon$ | $(Y - \mu)^2$ | Continuous outcomes |
| **Logit** | $P(Y=1) = \sigma(\alpha + \beta T)$ | Binary Cross-Entropy | Binary choice |
| **Poisson** | $Y \sim \text{Pois}(\exp(\alpha + \beta T))$ | $\lambda - Y \ln \lambda$ | Count data |
| **Gamma** | $Y \sim \text{Gamma}(k, \mu/k)$ | $Y/\mu + \ln \mu$ | Positive skewed data |
| **Gumbel** | $Y \sim \text{Gumbel}(\alpha + \beta T, s)$ | $(Y-\mu)/s + e^{-(Y-\mu)/s}$ | Maxima, durations |
| **Tobit** | $Y = \max(0, \alpha + \beta T + \varepsilon)$ | Censored NLL | Censored data |
| **NegBin** | $Y \sim \text{NB}(\mu, r)$ | Negative Binomial NLL | Overdispersed counts |
| **Weibull** | $Y \sim \text{Weibull}(k, \lambda)$ | Weibull NLL | Survival analysis |

## Monte Carlo Validation

Run simulations to validate coverage:

```bash
python -m src.deepstats.run_mc --M 50 --N 1000 --epochs 100 \
    --models linear poisson logit \
    --methods naive influence
```

Expected results:

| Method | SE Ratio | Coverage |
| :--- | :--- | :--- |
| Naive | ~0.2-0.3 | 30-50% |
| Influence | ~0.9-1.1 | 85-98% |

## Repository Structure

```
src/deepstats/     # Main package (6 files)
references/        # Academic papers
paper/             # Our paper (LaTeX)
prototypes/        # Experiments
archive/           # Old implementation (v1)
```

## Citation

If you use `deepstats` in your research, please cite:

```bibtex
@article{farrell2021deep,
  title={Deep neural networks for estimation and inference},
  author={Farrell, Max H and Liang, Tengyuan and Misra, Sanjog},
  journal={Econometrica},
  volume={89},
  number={1},
  pages={181--213},
  year={2021}
}

@article{farrell2025deep,
  title={Deep Learning for Individual Heterogeneity},
  author={Farrell, Max H and Liang, Tengyuan and Misra, Sanjog},
  journal={arXiv preprint arXiv:2010.14694},
  year={2025}
}
```

## License

MIT
