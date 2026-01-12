# DGP Module (Archived)

Data Generating Processes for Monte Carlo simulation.

**Note:** The DGP module has been archived to `archive/deep_inference_v1/`. For Monte Carlo validation, see the Jupyter notebooks in `tutorials/`.

## Creating Your Own DGP

For simulation studies, create your own DGP in Python:

```python
import numpy as np
from scipy.special import expit  # sigmoid

def generate_linear_dgp(n, seed=None):
    """Generate data from a linear DGP."""
    if seed is not None:
        np.random.seed(seed)

    # Covariates
    X = np.random.randn(n, 10)
    T = np.random.randn(n)

    # True structural functions
    alpha_true = 1.0 + 0.5 * X[:, 0]
    beta_true = 0.5 + 0.3 * X[:, 0]

    # Outcome
    Y = alpha_true + beta_true * T + np.random.randn(n)

    return {
        'Y': Y,
        'T': T,
        'X': X,
        'mu_true': beta_true.mean()
    }

def generate_logit_dgp(n, seed=None):
    """Generate data from a logit DGP."""
    if seed is not None:
        np.random.seed(seed)

    # Covariates
    X = np.random.randn(n, 10)
    T = np.random.randn(n)

    # True structural functions
    alpha_true = 0.5 + 0.3 * X[:, 0]
    beta_true = -0.5 + 0.5 * X[:, 0]

    # Probability and outcome
    p = expit(alpha_true + beta_true * T)
    Y = np.random.binomial(1, p).astype(float)

    return {
        'Y': Y,
        'T': T,
        'X': X,
        'mu_true': beta_true.mean()
    }
```

## Usage

```python
from deep_inference import structural_dml

# Generate data
data = generate_linear_dgp(n=2000, seed=42)

# Run inference
result = structural_dml(
    Y=data['Y'],
    T=data['T'],
    X=data['X'],
    family='linear'
)

# Compare to truth
print(f"True: {data['mu_true']:.4f}")
print(f"Est:  {result.mu_hat:.4f}")
```

## Example DGPs in Tutorials

See the Jupyter notebooks for complete examples:

- `tutorials/01_linear_oracle.ipynb` - Linear DGP with OLS oracle
- `tutorials/02_logit_oracle.ipynb` - Logit DGP with logistic regression oracle
