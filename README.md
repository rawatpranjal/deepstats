# `deep-inference`

**Deep Learning for Individual Heterogeneity**

[![GitHub](https://img.shields.io/badge/GitHub-rawatpranjal%2Fdeep--inference-blue?logo=github)](https://github.com/rawatpranjal/deep-inference)
[![PyPI](https://img.shields.io/pypi/v/deep-inference?color=blue)](https://pypi.org/project/deep-inference/)
[![Documentation](https://img.shields.io/badge/docs-rawatpranjal.github.io-blue)](https://rawatpranjal.github.io/deep-inference/)

**References:**
- Farrell, Liang, Misra (2021) ["Deep Neural Networks for Estimation and Inference"](https://doi.org/10.3982/ECTA16901) *Econometrica*
- Farrell, Liang, Misra (2025) "Deep Learning for Individual Heterogeneity"

---

`deep-inference` is a Python package for **enriching structural economic models** with deep learning. It implements the framework developed by Farrell, Liang, and Misra (2021, 2025) to recover rich, non-linear parameter heterogeneity ($\theta(X)$) while maintaining the interpretability and validity of structural economics.

Standard deep learning minimizes prediction error, which leads to biased parameter estimates ("The Inference Trap"). This package implements **Influence Function-based Debiasing** (a form of Double Machine Learning) to provide valid confidence intervals and p-values for economic targets.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why `deep-inference`?

Economic structure and Machine Learning are **complements**, not substitutes.
*   **Deep Learning** provides the capacity to learn complex, high-dimensional heterogeneity.
*   **Structural Models** provide the constraints necessary for causal interpretation and counterfactuals.

`deep-inference` enforces the structural loss (e.g., Tobit likelihood) on the output of a neural network, ensuring that the estimated parameters $\alpha(X), \beta(X)$ respect the economic theory.

## Installation

```bash
pip install deep-inference
```

## Quickstart

Estimate a linear demand model where Price Elasticity $\beta$ varies non-linearly with customer characteristics $X$.

```python
import numpy as np
from deep_inference import structural_dml

# 1. Generate synthetic data (Linear Demand with Heterogeneity)
np.random.seed(42)
n = 2000
X = np.random.randn(n, 10)  # Customer characteristics
T = np.random.randn(n)       # Treatment (e.g., price)
beta_true = np.cos(np.pi * X[:, 0]) * (X[:, 1] > 0) + 0.5 * X[:, 2]
Y = X[:, 0] + beta_true * T + np.random.randn(n)
mu_true = beta_true.mean()

# 2. Run Inference
# The package trains the structural network, computes the influence function,
# and aggregates results via cross-fitting.
result = structural_dml(
    Y=Y, T=T, X=X,
    family='linear',
    hidden_dims=[128, 64, 32],
    epochs=50,
    lr=0.01,
    n_folds=50,
)

print(f"Avg Elasticity (Truth): {mu_true:.4f}")
print(f"Avg Elasticity (Est):   {result.mu_hat:.4f}")
print(f"Standard Error:         {result.se:.4f}")
print(f"95% CI:                 [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

## New `inference()` API

The new `inference()` API provides additional flexibility:

- **Flexible targets**: Average Marginal Effect (AME), custom target functions
- **Randomization mode**: For RCTs, compute Λ directly instead of estimating
- **Regime auto-detection**: Automatically chooses optimal Lambda strategy

```python
from deep_inference import inference
from deep_inference.lambda_.compute import Normal

# Average Marginal Effect (probability scale, not log-odds)
result = inference(Y, T, X, model='logit', target='ame', t_tilde=0.0)

# Custom target with autodiff Jacobian
import torch
def my_target(x, theta, t_tilde):
    return torch.sigmoid(theta[0] + theta[1] * t_tilde)

result = inference(Y, T, X, model='logit', target_fn=my_target)

# Randomized experiment (Regime A)
result = inference(Y, T, X, model='logit', target='beta',
                   is_randomized=True, treatment_dist=Normal(0, 1))
```

### Three Estimation Regimes

| Regime | Condition | Lambda Method | Cross-Fitting |
|--------|-----------|---------------|---------------|
| **A** | RCT with known treatment distribution | Compute via MC | 2-way |
| **B** | Linear structural model | Closed-form analytic | 2-way |
| **C** | Observational, nonlinear | Neural network estimation | 3-way |

## Supported Structural Families

The package abstracts the math of influence functions. You simply select the family that matches your outcome variable.

| Family | Model Structure | Use Case |
|--------|-----------------|----------|
| **Linear** | $Y = \alpha(X) + \beta(X)T + \varepsilon$ | Wages, Test Scores, Consumption |
| **Logit** | $P(Y=1) = \sigma(\alpha(X) + \beta(X)T)$ | Binary Choice, Market Entry |
| **Poisson** | $Y \sim \text{Pois}(\exp(\alpha(X) + \beta(X)T))$ | Patent Counts, Doctor Visits |
| **Gamma** | $Y \sim \text{Gamma}(k, \exp(\alpha(X) + \beta(X)T))$ | Healthcare Costs, Insurance Claims |
| **Gumbel** | $Y \sim \text{Gumbel}(\alpha(X) + \beta(X)T, s)$ | Extreme Value Analysis |
| **Tobit** | $Y = \max(0, \alpha(X) + \beta(X)T + \sigma\varepsilon)$ | Labor Supply, Censored Demand |
| **NegBin** | $Y \sim \text{NegBin}(\exp(\alpha(X) + \beta(X)T), r)$ | Count Data with Overdispersion |
| **Weibull** | $Y \sim \text{Weibull}(k, \exp(\alpha(X) + \beta(X)T))$ | Duration Analysis, Survival |

*Note: For complex models like Tobit, `deep-inference` automatically handles the joint estimation of structural variance $\sigma(X)$ required for consistent inference.*

## Methodological Details

### 1. The Enriched Model
We replace fixed parameters $\theta$ with neural networks $\theta(X)$:

```math
\hat{\theta}(\cdot) = \arg \min_{\theta \in \mathcal{F}_{DNN}} \sum \ell(y_i, t_i, \theta(x_i))
```

### 2. The Influence Function Correction
Naive averaging of $\hat{\theta}(X)$ yields biased inference. We construct a Neyman-Orthogonal score $\psi$ using the **Influence Function**:

```math
\psi(z) = H(\hat{\theta}) + \nabla_\theta H \cdot \Lambda(x)^{-1} \cdot \nabla_\theta \ell(z, \hat{\theta})
```

Where $\Lambda(x) = \mathbb{E}[\nabla^2 \ell \mid X=x]$ is the conditional Hessian.
*   **Automatic Differentiation:** `deep-inference` uses PyTorch Autograd to compute exact Jacobians and Hessians for any model family.
*   **Stability:** Includes Tikhonov regularization for inverting Hessians in non-linear models (e.g., Logit/Tobit).

## Validation (Eval Suite Results)

The package includes a comprehensive eval suite (`evals/`) validating every mathematical component.

### Parameter Recovery (Eval 01)

Neural networks recover heterogeneous parameters θ(x) = [α(x), β(x)] across 12 families:

| Family | Corr(α) | Corr(β) | Status |
|--------|---------|---------|--------|
| linear | 0.991 | 0.995 | PASS |
| logit | 0.978 | 0.996 | PASS |
| poisson | 0.980 | 0.967 | PASS |
| negbin | 0.990 | 0.983 | PASS |
| gamma | 0.993 | 0.990 | PASS |
| weibull | 0.993 | 0.986 | PASS |
| gumbel | 0.967 | 0.991 | PASS |
| tobit | 0.987 | 0.988 | PASS |
| gaussian | 0.984 | 0.995 | PASS |
| probit | 0.983 | 0.985 | PASS |
| beta | 0.989 | 0.975 | PASS |
| zip | 0.969 | 0.944 | PASS |

**Result: 12/12 families PASS** (n=2000, epochs=100, 3 seeds)

### Lambda Estimation (Eval 03)

Conditional Hessian Λ(x) = E[ℓ_θθ|X=x] estimation methods:

| Method | Correlation | Frob Error | Time |
|--------|-------------|------------|------|
| aggregate | 0.000 | 0.121 | 0.02s |
| **mlp** | **0.997** | **0.018** | 12.5s |
| lgbm | 0.978 | 0.033 | 1.2s |
| rf | 0.904 | 0.060 | 0.3s |
| ridge | 0.508 | 0.087 | 0.08s |

**Result: MLP best accuracy, LGBM best speed/accuracy tradeoff**

### Coverage (Eval 05)

Monte Carlo validation (M=50 simulations, n=1000):

| Metric | Value | Target |
|--------|-------|--------|
| Coverage | 88% (44/50) | 85-97% |
| SE Ratio | 0.873 | 0.8-1.2 |
| Mean Bias | 0.002 | < 0.1 |

### Summary

| Eval | Tests | Result |
|------|-------|--------|
| 01: Parameter Recovery | 12 families × 3 seeds | 12/12 PASS |
| 02: Autodiff Accuracy | 31 checks | 31/31 PASS |
| 03: Lambda Estimation | 9 tests | 9/9 PASS |
| 04: Target Jacobian | 92 checks | 92/92 PASS |
| 05: Influence Functions | 4 rounds | 4/4 PASS |

## Citation

```bibtex
@article{farrell2021deep,
  title={Deep Neural Networks for Estimation and Inference},
  author={Farrell, Max H. and Liang, Tengyuan and Misra, Sanjog},
  journal={Econometrica},
  volume={89},
  number={1},
  pages={181--213},
  year={2021}
}

@article{farrell2025heterogeneity,
  title={Deep Learning for Individual Heterogeneity},
  author={Farrell, Max H. and Liang, Tengyuan and Misra, Sanjog},
  journal={Working Paper},
  year={2025}
}
```

## License

MIT
