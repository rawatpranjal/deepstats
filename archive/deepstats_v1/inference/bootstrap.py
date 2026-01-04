"""Bootstrap standard error estimation.

This module implements bootstrap methods for standard error estimation
including pairs bootstrap, residual bootstrap, and wild bootstrap.

References
----------
- Efron, B. (1979). Bootstrap methods: another look at the jackknife.
- Wu, C. F. J. (1986). Jackknife, bootstrap and other resampling methods.
- Mammen, E. (1993). Bootstrap and wild bootstrap for high dimensional
  linear models.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .._typing import Float64Array


@dataclass
class BootstrapResult:
    """Container for bootstrap inference results.

    Attributes
    ----------
    se : ndarray
        Bootstrap standard errors (p,).
    vcov : ndarray
        Bootstrap variance-covariance matrix (p, p).
    samples : ndarray
        Bootstrap estimates (n_bootstrap, p).
    ci_lower : ndarray
        Percentile confidence interval lower bound (p,).
    ci_upper : ndarray
        Percentile confidence interval upper bound (p,).
    n_bootstrap : int
        Number of bootstrap replications.
    bootstrap_type : str
        Type of bootstrap used.
    """

    se: Float64Array
    vcov: Float64Array
    samples: Float64Array
    ci_lower: Float64Array
    ci_upper: Float64Array
    n_bootstrap: int
    bootstrap_type: str


def bootstrap_pairs(
    X: Float64Array,
    y: Float64Array,
    fit_fn: Callable[[Float64Array, Float64Array], Float64Array],
    n_bootstrap: int = 1000,
    random_state: int | None = None,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """Pairs (case) bootstrap for standard errors.

    Resamples (X_i, y_i) pairs with replacement and refits the model.
    Most robust to heteroskedasticity and model misspecification.

    Parameters
    ----------
    X : ndarray
        Feature matrix (n, p).
    y : ndarray
        Target vector (n,).
    fit_fn : callable
        Function that fits model and returns parameter estimates.
        Signature: fit_fn(X, y) -> params
    n_bootstrap : int, default=1000
        Number of bootstrap replications.
    random_state : int, optional
        Random seed for reproducibility.
    confidence_level : float, default=0.95
        Confidence level for intervals.

    Returns
    -------
    BootstrapResult
        Bootstrap inference results.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    # Get number of parameters from first fit
    params_original = fit_fn(X, y)
    p = len(params_original)

    boot_estimates = np.zeros((n_bootstrap, p))

    for b in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n, size=n, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]

        # Fit model and compute parameters
        params_boot = fit_fn(X_boot, y_boot)
        boot_estimates[b] = params_boot

    # Compute statistics
    se = np.std(boot_estimates, axis=0, ddof=1)
    vcov = np.cov(boot_estimates.T)

    # Percentile confidence intervals
    alpha = 1 - confidence_level
    ci_lower = np.percentile(boot_estimates, 100 * alpha / 2, axis=0)
    ci_upper = np.percentile(boot_estimates, 100 * (1 - alpha / 2), axis=0)

    return BootstrapResult(
        se=se,
        vcov=vcov,
        samples=boot_estimates,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_bootstrap=n_bootstrap,
        bootstrap_type="pairs",
    )


def bootstrap_residual(
    X: Float64Array,
    y: Float64Array,
    fitted_values: Float64Array,
    residuals: Float64Array,
    fit_fn: Callable[[Float64Array, Float64Array], Float64Array],
    n_bootstrap: int = 1000,
    random_state: int | None = None,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """Residual bootstrap for standard errors.

    Fixes X, resamples residuals with replacement, creates y* = fitted + e*.
    Assumes homoskedasticity; more efficient than pairs when valid.

    Parameters
    ----------
    X : ndarray
        Feature matrix (n, p).
    y : ndarray
        Target vector (n,).
    fitted_values : ndarray
        Model predictions (n,).
    residuals : ndarray
        Model residuals (n,).
    fit_fn : callable
        Function that fits model and returns parameter estimates.
    n_bootstrap : int, default=1000
        Number of bootstrap replications.
    random_state : int, optional
        Random seed.
    confidence_level : float, default=0.95
        Confidence level.

    Returns
    -------
    BootstrapResult
        Bootstrap inference results.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    # Get number of parameters from original fit
    params_original = fit_fn(X, y)
    p = len(params_original)

    # Center residuals
    residuals_centered = residuals - np.mean(residuals)

    boot_estimates = np.zeros((n_bootstrap, p))

    for b in range(n_bootstrap):
        # Resample residuals
        indices = rng.choice(n, size=n, replace=True)
        resid_boot = residuals_centered[indices]

        # Create bootstrap response
        y_boot = fitted_values + resid_boot

        # Fit model on bootstrap sample
        params_boot = fit_fn(X, y_boot)
        boot_estimates[b] = params_boot

    # Compute statistics
    se = np.std(boot_estimates, axis=0, ddof=1)
    vcov = np.cov(boot_estimates.T)

    alpha = 1 - confidence_level
    ci_lower = np.percentile(boot_estimates, 100 * alpha / 2, axis=0)
    ci_upper = np.percentile(boot_estimates, 100 * (1 - alpha / 2), axis=0)

    return BootstrapResult(
        se=se,
        vcov=vcov,
        samples=boot_estimates,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_bootstrap=n_bootstrap,
        bootstrap_type="residual",
    )


def bootstrap_wild(
    X: Float64Array,
    y: Float64Array,
    fitted_values: Float64Array,
    residuals: Float64Array,
    fit_fn: Callable[[Float64Array, Float64Array], Float64Array],
    n_bootstrap: int = 1000,
    random_state: int | None = None,
    confidence_level: float = 0.95,
    distribution: str = "rademacher",
) -> BootstrapResult:
    """Wild bootstrap for standard errors.

    Robust to heteroskedasticity. Creates y* = fitted + w * residuals
    where w is drawn from Rademacher (+1/-1) or Mammen distribution.

    Parameters
    ----------
    X : ndarray
        Feature matrix (n, p).
    y : ndarray
        Target vector (n,).
    fitted_values : ndarray
        Model predictions (n,).
    residuals : ndarray
        Model residuals (n,).
    fit_fn : callable
        Function that fits model and returns parameter estimates.
    n_bootstrap : int, default=1000
        Number of bootstrap replications.
    random_state : int, optional
        Random seed.
    confidence_level : float, default=0.95
        Confidence level.
    distribution : str, default="rademacher"
        Wild bootstrap distribution: "rademacher" or "mammen".

    Returns
    -------
    BootstrapResult
        Bootstrap inference results.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    # Get number of parameters
    params_original = fit_fn(X, y)
    p = len(params_original)

    boot_estimates = np.zeros((n_bootstrap, p))

    for b in range(n_bootstrap):
        if distribution == "rademacher":
            # Rademacher: +1 or -1 with equal probability
            w = rng.choice([-1, 1], size=n).astype(np.float64)
        elif distribution == "mammen":
            # Mammen two-point distribution (better for skewness)
            sqrt5 = np.sqrt(5)
            p_neg = (sqrt5 + 1) / (2 * sqrt5)
            w = np.where(
                rng.random(n) < p_neg,
                -(sqrt5 - 1) / 2,
                (sqrt5 + 1) / 2,
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Create bootstrap response
        y_boot = fitted_values + w * residuals

        # Fit model on bootstrap sample
        params_boot = fit_fn(X, y_boot)
        boot_estimates[b] = params_boot

    # Compute statistics
    se = np.std(boot_estimates, axis=0, ddof=1)
    vcov = np.cov(boot_estimates.T)

    alpha = 1 - confidence_level
    ci_lower = np.percentile(boot_estimates, 100 * alpha / 2, axis=0)
    ci_upper = np.percentile(boot_estimates, 100 * (1 - alpha / 2), axis=0)

    return BootstrapResult(
        se=se,
        vcov=vcov,
        samples=boot_estimates,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_bootstrap=n_bootstrap,
        bootstrap_type=f"wild_{distribution}",
    )


def create_nn_fit_function(
    network: nn.Module,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    compute_marginal_effects: bool = True,
) -> Callable[[Float64Array, Float64Array], Float64Array]:
    """Create a fit function for neural network bootstrap.

    Parameters
    ----------
    network : nn.Module
        Original trained network (used as template).
    epochs : int, default=50
        Number of epochs for bootstrap refitting.
    lr : float, default=1e-3
        Learning rate.
    batch_size : int, default=128
        Batch size.
    compute_marginal_effects : bool, default=True
        If True, return average marginal effects.
        If False, return network predictions.

    Returns
    -------
    callable
        Function that fits model and returns parameters.
    """

    def fit_fn(X: Float64Array, y: Float64Array) -> Float64Array:
        """Fit network and return marginal effects."""
        # Clone network architecture
        new_network = copy.deepcopy(network)

        # Device
        device = next(new_network.parameters()).device

        # Move to device if needed
        X_tensor = torch.from_numpy(X).float().to(device)
        y_tensor = torch.from_numpy(y).float().to(device).unsqueeze(1)

        # Optimizer
        optimizer = optim.Adam(new_network.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train
        new_network.train()
        for _ in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                y_pred = new_network(batch_X)
                loss = loss_fn(y_pred, batch_y)
                loss.backward()
                optimizer.step()

        if compute_marginal_effects:
            # Compute average marginal effects
            return _compute_marginal_effects(new_network, X, device)
        else:
            # Return fitted values
            new_network.eval()
            with torch.no_grad():
                return new_network(X_tensor).squeeze().cpu().numpy()

    return fit_fn


def _compute_marginal_effects(
    network: nn.Module,
    X: Float64Array,
    device: torch.device,
) -> Float64Array:
    """Compute average marginal effects via numerical differentiation."""
    eps = 1e-4
    n, p = X.shape
    effects = np.zeros(p)

    network.eval()
    with torch.no_grad():
        for j in range(p):
            X_plus = X.copy()
            X_minus = X.copy()
            X_plus[:, j] += eps
            X_minus[:, j] -= eps

            X_plus_t = torch.from_numpy(X_plus).float().to(device)
            X_minus_t = torch.from_numpy(X_minus).float().to(device)

            y_plus = network(X_plus_t).squeeze().cpu().numpy()
            y_minus = network(X_minus_t).squeeze().cpu().numpy()

            gradient = (y_plus - y_minus) / (2 * eps)
            effects[j] = np.mean(gradient)

    return effects
