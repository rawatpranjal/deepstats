"""Pytest configuration and fixtures for deepstats tests."""

import numpy as np
import pytest


@pytest.fixture
def seed():
    """Random seed for reproducibility."""
    return 42


@pytest.fixture
def linear_dgp(seed):
    """Generate linear DGP data: Y = X @ beta + epsilon.

    True coefficients: [0.5, -0.3, 0.8]
    """
    np.random.seed(seed)
    n = 1000
    p = 3
    X = np.random.randn(n, p)
    beta_true = np.array([0.5, -0.3, 0.8])
    epsilon = np.random.randn(n) * 0.5
    y = X @ beta_true + epsilon

    return {
        "X": X,
        "y": y,
        "beta_true": beta_true,
        "n": n,
        "p": p,
        "sigma": 0.5,
    }


@pytest.fixture
def causal_dgp(seed):
    """Generate causal DGP with known ATE.

    Y = tau * T + g(X) + epsilon
    T = Bernoulli(pi(X))

    True ATE = 2.0
    """
    np.random.seed(seed)
    n = 1000
    p = 5
    X = np.random.randn(n, p)

    # Propensity score
    logit = 0.3 * X[:, 0] - 0.2 * X[:, 1]
    prob = 1 / (1 + np.exp(-logit))
    T = np.random.binomial(1, prob)

    # Outcome
    tau_true = 2.0
    g_X = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.4 * X[:, 2]
    epsilon = np.random.randn(n) * 0.5
    Y = tau_true * T + g_X + epsilon

    return {
        "Y": Y,
        "T": T,
        "X": X,
        "tau_true": tau_true,
        "n": n,
        "p": p,
    }


@pytest.fixture
def poisson_dgp(seed):
    """Generate Poisson DGP: Y ~ Poisson(exp(g(X)))."""
    np.random.seed(seed)
    n = 1000
    p = 2
    X = np.random.randn(n, p)

    log_lambda = 1 + 0.3 * X[:, 0] - 0.2 * X[:, 1]
    y = np.random.poisson(np.exp(log_lambda))

    return {
        "X": X,
        "y": y.astype(float),
        "coeffs_true": np.array([0.3, -0.2]),
        "n": n,
        "p": p,
    }
