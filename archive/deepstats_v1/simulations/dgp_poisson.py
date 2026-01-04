"""Data generating processes for Poisson regression simulations.

This module provides DGPs for testing DeepPoisson inference on:
- E[lambda(X)]: Average rate parameter
- Var[lambda(X)]: Heterogeneity in rate
- Quantiles of lambda(X): Distribution of rates

Scenarios:
- Low-dimensional: p=5, simple true functions
- High-dimensional: p=50, sparse structure (only 5 covariates matter)
- Nonlinear: Complex nonlinear lambda function
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PoissonSimulationData:
    """Container for Poisson simulation data.

    Attributes
    ----------
    X : np.ndarray
        Covariates (n, p).
    y : np.ndarray
        Count outcomes (n,).
    true_lambda : np.ndarray
        True rate parameters lambda(X_i).
    true_mean_lambda : float
        True E[lambda(X)].
    true_var_lambda : float
        True Var[lambda(X)].
    true_quantiles : dict[float, float]
        True quantiles of lambda(X).
    scenario : str
        Name of scenario.
    """

    X: np.ndarray
    y: np.ndarray
    true_lambda: np.ndarray
    true_mean_lambda: float
    true_var_lambda: float
    true_quantiles: dict[float, float]
    scenario: str


def make_poisson_dgp_lowdim(
    seed: int,
    n: int = 2000,
    p: int = 5,
) -> PoissonSimulationData:
    """Low-dimensional Poisson DGP.

    True model: log(lambda(X)) = 0.5 + 0.3*X1 - 0.2*X2 + 0.1*X1*X2

    This is a simple setting where the neural network should easily
    learn the true function.

    Parameters
    ----------
    seed : int
        Random seed.
    n : int, default=2000
        Sample size.
    p : int, default=5
        Number of covariates.

    Returns
    -------
    PoissonSimulationData
        Container with data and true values.

    Examples
    --------
    >>> data = make_poisson_dgp_lowdim(seed=42, n=2000, p=5)
    >>> print(f"True E[lambda]: {data.true_mean_lambda:.3f}")
    >>> print(f"True Var[lambda]: {data.true_var_lambda:.3f}")
    """
    rng = np.random.default_rng(seed)

    # Generate covariates (standard normal)
    X = rng.standard_normal((n, p))

    # True log-rate function (depends on first 2 covariates)
    def log_lambda_func(X: np.ndarray) -> np.ndarray:
        return 0.5 + 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 0] * X[:, 1]

    log_lambda = log_lambda_func(X)
    true_lambda = np.exp(log_lambda)

    # Generate counts
    y = rng.poisson(true_lambda)

    # True targets (population quantities for this sample)
    true_mean_lambda = float(np.mean(true_lambda))
    true_var_lambda = float(np.var(true_lambda, ddof=0))

    quantiles_to_compute = (0.1, 0.25, 0.5, 0.75, 0.9)
    true_quantiles = {q: float(np.quantile(true_lambda, q)) for q in quantiles_to_compute}

    return PoissonSimulationData(
        X=X,
        y=y.astype(np.float64),
        true_lambda=true_lambda,
        true_mean_lambda=true_mean_lambda,
        true_var_lambda=true_var_lambda,
        true_quantiles=true_quantiles,
        scenario="poisson_lowdim",
    )


def make_poisson_dgp_highdim(
    seed: int,
    n: int = 2000,
    p: int = 50,
    p_relevant: int = 5,
) -> PoissonSimulationData:
    """High-dimensional Poisson DGP (sparse).

    True model: log(lambda(X)) = 0.3 + sum_{j=1}^5 0.2*(-1)^j * X_j + nonlinear

    Only the first `p_relevant` covariates matter. This tests the
    estimator's ability to handle irrelevant noise variables.

    Parameters
    ----------
    seed : int
        Random seed.
    n : int, default=2000
        Sample size.
    p : int, default=50
        Total number of covariates.
    p_relevant : int, default=5
        Number of relevant covariates.

    Returns
    -------
    PoissonSimulationData
        Container with data and true values.

    Examples
    --------
    >>> data = make_poisson_dgp_highdim(seed=42, n=2000, p=50)
    >>> print(f"True E[lambda]: {data.true_mean_lambda:.3f}")
    """
    rng = np.random.default_rng(seed)

    # Generate covariates
    X = rng.standard_normal((n, p))

    # True log-rate function (sparse + nonlinear)
    def log_lambda_func(X: np.ndarray) -> np.ndarray:
        result = 0.3 * np.ones(len(X))
        for j in range(min(p_relevant, p)):
            result += 0.2 * ((-1) ** j) * X[:, j]
            if j < 2:
                # Add some nonlinearity for the first 2 covariates
                result += 0.05 * X[:, j] ** 2
        return result

    log_lambda = log_lambda_func(X)
    true_lambda = np.exp(log_lambda)

    # Generate counts
    y = rng.poisson(true_lambda)

    # True targets
    true_mean_lambda = float(np.mean(true_lambda))
    true_var_lambda = float(np.var(true_lambda, ddof=0))

    quantiles_to_compute = (0.1, 0.25, 0.5, 0.75, 0.9)
    true_quantiles = {q: float(np.quantile(true_lambda, q)) for q in quantiles_to_compute}

    return PoissonSimulationData(
        X=X,
        y=y.astype(np.float64),
        true_lambda=true_lambda,
        true_mean_lambda=true_mean_lambda,
        true_var_lambda=true_var_lambda,
        true_quantiles=true_quantiles,
        scenario=f"poisson_highdim_p{p}",
    )


def make_poisson_dgp_nonlinear(
    seed: int,
    n: int = 2000,
    p: int = 5,
) -> PoissonSimulationData:
    """Highly nonlinear Poisson DGP.

    True model: log(lambda(X)) = sin(X1) + 0.3*X2^2 - 0.5*exp(-X3^2) + 0.2*X1*X2

    This tests the estimator's ability to capture complex nonlinearities.

    Parameters
    ----------
    seed : int
        Random seed.
    n : int, default=2000
        Sample size.
    p : int, default=5
        Number of covariates.

    Returns
    -------
    PoissonSimulationData
        Container with data and true values.
    """
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n, p))

    def log_lambda_func(X: np.ndarray) -> np.ndarray:
        return (
            0.5 * np.sin(2 * X[:, 0])
            + 0.3 * X[:, 1] ** 2
            - 0.5 * np.exp(-X[:, 2] ** 2)
            + 0.2 * X[:, 0] * X[:, 1]
        )

    log_lambda = log_lambda_func(X)
    true_lambda = np.exp(log_lambda)

    y = rng.poisson(true_lambda)

    true_mean_lambda = float(np.mean(true_lambda))
    true_var_lambda = float(np.var(true_lambda, ddof=0))

    quantiles_to_compute = (0.1, 0.25, 0.5, 0.75, 0.9)
    true_quantiles = {q: float(np.quantile(true_lambda, q)) for q in quantiles_to_compute}

    return PoissonSimulationData(
        X=X,
        y=y.astype(np.float64),
        true_lambda=true_lambda,
        true_mean_lambda=true_mean_lambda,
        true_var_lambda=true_var_lambda,
        true_quantiles=true_quantiles,
        scenario="poisson_nonlinear",
    )


def make_poisson_dgp_high_heterogeneity(
    seed: int,
    n: int = 2000,
    p: int = 5,
) -> PoissonSimulationData:
    """Poisson DGP with high heterogeneity in lambda.

    True model: log(lambda(X)) = 0.5 + 0.8*X1 - 0.6*X2

    Larger coefficients lead to more spread in lambda values,
    testing the variance estimation.

    Parameters
    ----------
    seed : int
        Random seed.
    n : int, default=2000
        Sample size.
    p : int, default=5
        Number of covariates.

    Returns
    -------
    PoissonSimulationData
        Container with data and true values.
    """
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n, p))

    def log_lambda_func(X: np.ndarray) -> np.ndarray:
        return 0.5 + 0.8 * X[:, 0] - 0.6 * X[:, 1]

    log_lambda = log_lambda_func(X)
    true_lambda = np.exp(log_lambda)

    y = rng.poisson(true_lambda)

    true_mean_lambda = float(np.mean(true_lambda))
    true_var_lambda = float(np.var(true_lambda, ddof=0))

    quantiles_to_compute = (0.1, 0.25, 0.5, 0.75, 0.9)
    true_quantiles = {q: float(np.quantile(true_lambda, q)) for q in quantiles_to_compute}

    return PoissonSimulationData(
        X=X,
        y=y.astype(np.float64),
        true_lambda=true_lambda,
        true_mean_lambda=true_mean_lambda,
        true_var_lambda=true_var_lambda,
        true_quantiles=true_quantiles,
        scenario="poisson_high_heterogeneity",
    )
