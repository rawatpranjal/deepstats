"""Data generating processes for testing and benchmarking.

This module provides DGP functions that generate synthetic data with
known ground truth parameters for testing estimator accuracy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._typing import Float64Array


def make_linear_highdim(
    n: int = 1000,
    p: int = 50,
    sparsity: float = 0.2,
    sigma: float = 1.0,
    intercept: float = 2.0,
    seed: int | None = None,
) -> tuple[Float64Array, Float64Array, dict[str, Any]]:
    """Generate high-dimensional linear regression data.

    Y = intercept + X @ beta + epsilon
    where epsilon ~ N(0, sigma^2)

    Parameters
    ----------
    n : int, default=1000
        Number of observations.
    p : int, default=50
        Number of features.
    sparsity : float, default=0.2
        Fraction of non-zero coefficients.
    sigma : float, default=1.0
        Standard deviation of noise.
    intercept : float, default=2.0
        Intercept term.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray
        Feature matrix (n, p).
    y : ndarray
        Target vector (n,).
    metadata : dict
        Contains 'beta_true', 'sigma', 'intercept', 'n', 'p'.

    Examples
    --------
    >>> from deepstats.datasets import make_linear_highdim
    >>> X, y, meta = make_linear_highdim(n=1000, p=50, seed=42)
    >>> print(f"True non-zero coefficients: {sum(meta['beta_true'] != 0)}")
    """
    rng = np.random.default_rng(seed)

    # Generate features with some correlation
    X = rng.standard_normal((n, p))

    # Add some correlation structure
    for j in range(1, p):
        X[:, j] = 0.5 * X[:, j - 1] + np.sqrt(1 - 0.5**2) * X[:, j]

    # Sparse coefficients
    n_nonzero = max(1, int(p * sparsity))
    beta = np.zeros(p)
    nonzero_idx = rng.choice(p, size=n_nonzero, replace=False)
    beta[nonzero_idx] = rng.uniform(-1, 1, size=n_nonzero)

    # Generate response
    epsilon = rng.standard_normal(n) * sigma
    y = intercept + X @ beta + epsilon

    metadata = {
        "beta_true": beta,
        "sigma": sigma,
        "intercept": intercept,
        "n": n,
        "p": p,
        "sparsity": sparsity,
        "n_nonzero": n_nonzero,
        "nonzero_idx": nonzero_idx,
    }

    return X, y, metadata


def make_poisson_highdim(
    n: int = 1000,
    p: int = 50,
    sparsity: float = 0.2,
    intercept: float = 1.0,
    seed: int | None = None,
) -> tuple[Float64Array, Float64Array, dict[str, Any]]:
    """Generate high-dimensional Poisson regression data.

    Y ~ Poisson(exp(intercept + X @ beta))

    Parameters
    ----------
    n : int, default=1000
        Number of observations.
    p : int, default=50
        Number of features.
    sparsity : float, default=0.2
        Fraction of non-zero coefficients.
    intercept : float, default=1.0
        Intercept (log scale).
    seed : int, optional
        Random seed.

    Returns
    -------
    X : ndarray
        Feature matrix (n, p).
    y : ndarray
        Count vector (n,).
    metadata : dict
        Contains 'beta_true', 'intercept', etc.

    Examples
    --------
    >>> from deepstats.datasets import make_poisson_highdim
    >>> X, y, meta = make_poisson_highdim(n=1000, p=50, seed=42)
    >>> print(f"Mean count: {y.mean():.2f}")
    """
    rng = np.random.default_rng(seed)

    # Generate features
    X = rng.standard_normal((n, p))

    # Add correlation
    for j in range(1, p):
        X[:, j] = 0.5 * X[:, j - 1] + np.sqrt(1 - 0.5**2) * X[:, j]

    # Sparse coefficients (smaller magnitude for Poisson)
    n_nonzero = max(1, int(p * sparsity))
    beta = np.zeros(p)
    nonzero_idx = rng.choice(p, size=n_nonzero, replace=False)
    beta[nonzero_idx] = rng.uniform(-0.3, 0.3, size=n_nonzero)

    # Generate response
    log_lambda = intercept + X @ beta
    # Clamp to prevent extreme counts
    log_lambda = np.clip(log_lambda, -5, 5)
    y = rng.poisson(np.exp(log_lambda))

    metadata = {
        "beta_true": beta,
        "intercept": intercept,
        "n": n,
        "p": p,
        "sparsity": sparsity,
        "n_nonzero": n_nonzero,
        "nonzero_idx": nonzero_idx,
        "mean_count": y.mean(),
    }

    return X, y.astype(np.float64), metadata


def make_binary_highdim(
    n: int = 1000,
    p: int = 50,
    sparsity: float = 0.2,
    intercept: float = 0.0,
    seed: int | None = None,
) -> tuple[Float64Array, Float64Array, dict[str, Any]]:
    """Generate high-dimensional binary classification data.

    Y ~ Bernoulli(sigmoid(intercept + X @ beta))

    Parameters
    ----------
    n : int, default=1000
        Number of observations.
    p : int, default=50
        Number of features.
    sparsity : float, default=0.2
        Fraction of non-zero coefficients.
    intercept : float, default=0.0
        Intercept term.
    seed : int, optional
        Random seed.

    Returns
    -------
    X : ndarray
        Feature matrix (n, p).
    y : ndarray
        Binary vector (n,).
    metadata : dict
        Contains 'beta_true', 'intercept', etc.

    Examples
    --------
    >>> from deepstats.datasets import make_binary_highdim
    >>> X, y, meta = make_binary_highdim(n=1000, p=50, seed=42)
    >>> print(f"Class balance: {y.mean():.2f}")
    """
    rng = np.random.default_rng(seed)

    # Generate features
    X = rng.standard_normal((n, p))

    # Add correlation
    for j in range(1, p):
        X[:, j] = 0.5 * X[:, j - 1] + np.sqrt(1 - 0.5**2) * X[:, j]

    # Sparse coefficients
    n_nonzero = max(1, int(p * sparsity))
    beta = np.zeros(p)
    nonzero_idx = rng.choice(p, size=n_nonzero, replace=False)
    beta[nonzero_idx] = rng.uniform(-0.8, 0.8, size=n_nonzero)

    # Generate response
    logit = intercept + X @ beta
    prob = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, prob)

    metadata = {
        "beta_true": beta,
        "intercept": intercept,
        "n": n,
        "p": p,
        "sparsity": sparsity,
        "n_nonzero": n_nonzero,
        "nonzero_idx": nonzero_idx,
        "class_balance": y.mean(),
    }

    return X, y.astype(np.float64), metadata


def make_nonlinear_highdim(
    n: int = 1000,
    p: int = 50,
    complexity: str = "medium",
    sigma: float = 0.5,
    seed: int | None = None,
) -> tuple[Float64Array, Float64Array, dict[str, Any]]:
    """Generate non-linear high-dimensional data.

    Y = g(X) + epsilon

    where g includes interactions, non-linearities, and threshold effects.
    This DGP is designed to show when neural networks outperform linear models.

    Parameters
    ----------
    n : int, default=1000
        Number of observations.
    p : int, default=50
        Number of features.
    complexity : str, default="medium"
        Complexity level: "low", "medium", or "high".
    sigma : float, default=0.5
        Noise standard deviation.
    seed : int, optional
        Random seed.

    Returns
    -------
    X : ndarray
        Feature matrix (n, p).
    y : ndarray
        Target vector (n,).
    metadata : dict
        Contains 'complexity', 'sigma', description of g.

    Examples
    --------
    >>> from deepstats.datasets import make_nonlinear_highdim
    >>> X, y, meta = make_nonlinear_highdim(n=1000, p=50, complexity="medium")
    >>> print(meta['g_description'])
    """
    rng = np.random.default_rng(seed)

    # Generate features
    X = rng.standard_normal((n, p))

    # Add correlation
    for j in range(1, p):
        X[:, j] = 0.3 * X[:, j - 1] + np.sqrt(1 - 0.3**2) * X[:, j]

    # Generate non-linear function based on complexity
    if complexity == "low":
        # Polynomial terms only
        g = (
            0.5 * X[:, 0]
            - 0.3 * X[:, 1]
            + 0.2 * X[:, 0] ** 2
            - 0.1 * X[:, 1] ** 2
        )
        g_desc = "g(X) = 0.5*X0 - 0.3*X1 + 0.2*X0^2 - 0.1*X1^2"

    elif complexity == "medium":
        # Polynomial + interactions
        g = (
            0.5 * X[:, 0]
            - 0.3 * X[:, 1]
            + 0.4 * X[:, 2]
            + 0.3 * X[:, 0] * X[:, 1]  # interaction
            - 0.2 * X[:, 0] ** 2
            + 0.15 * np.sin(2 * X[:, 3])  # periodic
            + 0.2 * np.maximum(X[:, 4], 0)  # ReLU-like
        )
        g_desc = "g(X) = 0.5*X0 - 0.3*X1 + 0.4*X2 + 0.3*X0*X1 - 0.2*X0^2 + 0.15*sin(2*X3) + 0.2*max(X4, 0)"

    else:  # high
        # Complex non-linear function
        g = (
            0.5 * X[:, 0]
            - 0.3 * X[:, 1]
            + 0.4 * X[:, 2]
            + 0.3 * X[:, 0] * X[:, 1]
            + 0.2 * X[:, 1] * X[:, 2]
            + 0.15 * X[:, 0] * X[:, 2]
            - 0.2 * X[:, 0] ** 2
            + 0.1 * X[:, 1] ** 2
            + 0.15 * np.sin(2 * X[:, 3])
            + 0.1 * np.cos(3 * X[:, 4])
            + 0.2 * np.maximum(X[:, 5], 0)
            + 0.15 * np.tanh(X[:, 6])
            + 0.1 * (X[:, 7] > 0).astype(float) * X[:, 7]  # threshold
        )
        g_desc = "g(X) = complex function with interactions, polynomials, periodic, ReLU, tanh, threshold effects"

    # Add noise
    epsilon = rng.standard_normal(n) * sigma
    y = g + epsilon

    # Compute marginal effects (average derivatives) for first few variables
    # These are approximations for comparison
    eps = 1e-4
    marginal_effects = {}
    for j in range(min(5, p)):
        X_plus = X.copy()
        X_minus = X.copy()
        X_plus[:, j] += eps
        X_minus[:, j] -= eps

        # Recompute g for perturbed X (simplified)
        if complexity == "low":
            g_plus = 0.5 * X_plus[:, 0] - 0.3 * X_plus[:, 1] + 0.2 * X_plus[:, 0]**2 - 0.1 * X_plus[:, 1]**2
            g_minus = 0.5 * X_minus[:, 0] - 0.3 * X_minus[:, 1] + 0.2 * X_minus[:, 0]**2 - 0.1 * X_minus[:, 1]**2
        elif complexity == "medium":
            g_plus = (0.5 * X_plus[:, 0] - 0.3 * X_plus[:, 1] + 0.4 * X_plus[:, 2]
                     + 0.3 * X_plus[:, 0] * X_plus[:, 1] - 0.2 * X_plus[:, 0]**2
                     + 0.15 * np.sin(2 * X_plus[:, 3]) + 0.2 * np.maximum(X_plus[:, 4], 0))
            g_minus = (0.5 * X_minus[:, 0] - 0.3 * X_minus[:, 1] + 0.4 * X_minus[:, 2]
                      + 0.3 * X_minus[:, 0] * X_minus[:, 1] - 0.2 * X_minus[:, 0]**2
                      + 0.15 * np.sin(2 * X_minus[:, 3]) + 0.2 * np.maximum(X_minus[:, 4], 0))
        else:
            # For high complexity, just use numerical diff on original g
            g_plus = g_minus = g  # Placeholder

        me = np.mean((g_plus - g_minus) / (2 * eps))
        marginal_effects[f"x{j}"] = me

    metadata = {
        "complexity": complexity,
        "sigma": sigma,
        "n": n,
        "p": p,
        "g_description": g_desc,
        "marginal_effects_approx": marginal_effects,
    }

    return X, y, metadata
