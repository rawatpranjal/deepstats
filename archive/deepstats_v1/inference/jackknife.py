"""Jackknife standard error estimation.

This module implements jackknife (leave-one-out and delete-d) methods
for standard error estimation.

References
----------
- Quenouille, M. H. (1956). Notes on bias in estimation.
- Tukey, J. W. (1958). Bias and confidence in not quite large samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .._typing import Float64Array


@dataclass
class JackknifeResult:
    """Container for jackknife inference results.

    Attributes
    ----------
    se : ndarray
        Jackknife standard errors (p,).
    vcov : ndarray
        Jackknife variance-covariance matrix (p, p).
    leave_one_out_estimates : ndarray
        Leave-one-out estimates (n, p).
    bias_estimate : ndarray
        Jackknife bias estimate (p,).
    """

    se: Float64Array
    vcov: Float64Array
    leave_one_out_estimates: Float64Array
    bias_estimate: Float64Array


def jackknife_se(
    X: Float64Array,
    y: Float64Array,
    fit_fn: Callable[[Float64Array, Float64Array], Float64Array],
    params: Float64Array,
) -> JackknifeResult:
    """Compute jackknife (leave-one-out) standard errors.

    The jackknife estimator is related to HC3 standard errors and provides
    a finite-sample correction. However, it is computationally expensive
    as it requires n model refits.

    Parameters
    ----------
    X : ndarray
        Feature matrix (n, p).
    y : ndarray
        Target vector (n,).
    fit_fn : callable
        Function that fits model and returns parameter estimates.
        Signature: fit_fn(X, y) -> params
    params : ndarray
        Full-sample parameter estimates (p,).

    Returns
    -------
    JackknifeResult
        Jackknife inference results.

    Notes
    -----
    For large n, consider using delete-d jackknife or bootstrap instead,
    as leave-one-out requires n model refits.
    """
    n = X.shape[0]
    p = len(params)

    loo_estimates = np.zeros((n, p))

    # Leave-one-out loop
    for i in range(n):
        # Create leave-one-out sample
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_loo = X[mask]
        y_loo = y[mask]

        # Refit and compute parameters
        params_loo = fit_fn(X_loo, y_loo)
        loo_estimates[i] = params_loo

    # Jackknife variance formula: (n-1)/n * sum((theta_-i - theta_bar)^2)
    theta_bar = np.mean(loo_estimates, axis=0)
    deviations = loo_estimates - theta_bar
    vcov = ((n - 1) / n) * (deviations.T @ deviations)
    se = np.sqrt(np.diag(vcov))

    # Bias estimate: (n-1) * (theta_bar - theta)
    bias = (n - 1) * (theta_bar - params)

    return JackknifeResult(
        se=se,
        vcov=vcov,
        leave_one_out_estimates=loo_estimates,
        bias_estimate=bias,
    )


def delete_d_jackknife_se(
    X: Float64Array,
    y: Float64Array,
    fit_fn: Callable[[Float64Array, Float64Array], Float64Array],
    params: Float64Array,
    d: int = 10,
    n_groups: int | None = None,
    random_state: int | None = None,
) -> JackknifeResult:
    """Compute delete-d jackknife standard errors.

    A computationally efficient alternative to leave-one-out that
    deletes d observations at a time. When d = n/g for g groups,
    this is equivalent to g-fold cross-validation variance estimation.

    Parameters
    ----------
    X : ndarray
        Feature matrix (n, p).
    y : ndarray
        Target vector (n,).
    fit_fn : callable
        Function that fits model and returns parameter estimates.
    params : ndarray
        Full-sample parameter estimates (p,).
    d : int, default=10
        Number of observations to delete per group.
    n_groups : int, optional
        Number of jackknife groups. If None, uses n // d.
    random_state : int, optional
        Random seed for group assignment.

    Returns
    -------
    JackknifeResult
        Jackknife inference results.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    p = len(params)

    # Determine number of groups
    if n_groups is None:
        n_groups = max(n // d, 2)

    # Create random group assignments
    indices = np.arange(n)
    rng.shuffle(indices)

    # Split into groups
    groups = np.array_split(indices, n_groups)

    group_estimates = np.zeros((n_groups, p))

    for g, group_indices in enumerate(groups):
        # Create delete-d sample (remove this group)
        mask = np.ones(n, dtype=bool)
        mask[group_indices] = False
        X_del = X[mask]
        y_del = y[mask]

        # Refit and compute parameters
        params_del = fit_fn(X_del, y_del)
        group_estimates[g] = params_del

    # Delete-d jackknife variance formula
    # V = ((n-d)/(d)) * (1/g) * sum((theta_-g - theta_bar)^2)
    theta_bar = np.mean(group_estimates, axis=0)
    deviations = group_estimates - theta_bar

    # Scaling factor for delete-d
    scale = (n - d) / d / n_groups

    vcov = scale * (deviations.T @ deviations)
    se = np.sqrt(np.diag(vcov))

    # Bias estimate
    bias = (n_groups - 1) * (theta_bar - params)

    return JackknifeResult(
        se=se,
        vcov=vcov,
        leave_one_out_estimates=group_estimates,
        bias_estimate=bias,
    )


def infinitesimal_jackknife_se(
    X: Float64Array,
    residuals: Float64Array,
    gradients: Float64Array,
) -> Float64Array:
    """Compute infinitesimal jackknife (IJ) standard errors.

    The IJ estimator is equivalent to HC3 for linear models and provides
    an efficient approximation to the full jackknife.

    For a linear model with gradients (design matrix) G and residuals e:
    IJ variance = (1/n^2) * sum_i (psi_i^2 / (1 - h_ii)^2)

    where psi_i = G_i * e_i and h_ii is the leverage.

    Parameters
    ----------
    X : ndarray
        Feature matrix (n, p).
    residuals : ndarray
        Model residuals (n,).
    gradients : ndarray
        Gradient matrix (n, p), typically marginal effects.

    Returns
    -------
    ndarray
        Standard errors (p,).
    """
    n, p = X.shape

    # Compute leverage values
    # h_ii = diag(G @ (G'G)^{-1} @ G')
    GtG = gradients.T @ gradients
    GtG_inv = np.linalg.pinv(GtG)
    leverage = np.sum((gradients @ GtG_inv) * gradients, axis=1)

    # Compute adjusted residuals
    adjustment = 1.0 / np.maximum((1 - leverage) ** 2, 1e-10)

    # Compute variance
    psi = gradients * residuals[:, np.newaxis]
    psi_adj = psi * np.sqrt(adjustment)[:, np.newaxis]

    vcov = (psi_adj.T @ psi_adj) / n**2
    se = np.sqrt(np.diag(vcov))

    return se
