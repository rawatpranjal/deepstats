"""Robust standard error estimation.

This module implements heteroskedasticity-robust standard errors (HC0-HC3)
and clustered standard errors following econometrics best practices.

The implementations are designed to match R's `fixest` and Stata output
to at least 6 decimal places.

References
----------
- White (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator"
- MacKinnon & White (1985). "Some heteroskedasticity-consistent covariance
  matrix estimators with improved finite sample properties"
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import linalg

from .._typing import Float64Array

SEType = Literal["iid", "HC0", "HC1", "HC2", "HC3", "cluster"]


def compute_bread_matrix(X: Float64Array) -> Float64Array:
    """Compute the bread matrix (X'X)^{-1}.

    Uses scipy.linalg.solve for numerical stability instead of
    numpy.linalg.inv.

    Parameters
    ----------
    X : Float64Array
        Design matrix (n, p).

    Returns
    -------
    Float64Array
        (X'X)^{-1} matrix (p, p).
    """
    XtX = X.T @ X
    # Use Cholesky decomposition for symmetric positive definite
    try:
        L = linalg.cholesky(XtX, lower=True)
        bread = linalg.cho_solve((L, True), np.eye(XtX.shape[0]))
    except linalg.LinAlgError:
        # Fall back to general solver if not positive definite
        bread = linalg.solve(XtX, np.eye(XtX.shape[0]), assume_a="sym")
    return bread


def compute_leverage(X: Float64Array, bread: Float64Array | None = None) -> Float64Array:
    """Compute leverage (hat) values.

    h_ii = x_i' (X'X)^{-1} x_i

    Parameters
    ----------
    X : Float64Array
        Design matrix (n, p).
    bread : Float64Array, optional
        Pre-computed (X'X)^{-1}. Computed if not provided.

    Returns
    -------
    Float64Array
        Leverage values (n,).
    """
    if bread is None:
        bread = compute_bread_matrix(X)
    # h_ii = diag(X @ bread @ X')
    return np.sum((X @ bread) * X, axis=1)


def compute_vcov_iid(
    X: Float64Array,
    residuals: Float64Array,
    bread: Float64Array | None = None,
) -> Float64Array:
    """Compute homoskedastic (iid) variance-covariance matrix.

    V = sigma^2 * (X'X)^{-1}

    Parameters
    ----------
    X : Float64Array
        Design matrix (n, p).
    residuals : Float64Array
        Model residuals (n,).
    bread : Float64Array, optional
        Pre-computed (X'X)^{-1}.

    Returns
    -------
    Float64Array
        Variance-covariance matrix (p, p).
    """
    n, p = X.shape
    if bread is None:
        bread = compute_bread_matrix(X)

    sigma2 = np.sum(residuals**2) / (n - p)
    return sigma2 * bread


def compute_vcov_hc0(
    X: Float64Array,
    residuals: Float64Array,
    bread: Float64Array | None = None,
) -> Float64Array:
    """Compute HC0 (White) heteroskedasticity-robust variance-covariance.

    V = (X'X)^{-1} X' diag(e^2) X (X'X)^{-1}

    This is the original White (1980) estimator, which is consistent
    but biased in finite samples.

    Parameters
    ----------
    X : Float64Array
        Design matrix (n, p).
    residuals : Float64Array
        Model residuals (n,).
    bread : Float64Array, optional
        Pre-computed (X'X)^{-1}.

    Returns
    -------
    Float64Array
        Variance-covariance matrix (p, p).
    """
    if bread is None:
        bread = compute_bread_matrix(X)

    # Meat: X' diag(e^2) X
    e2 = residuals**2
    meat = X.T @ (X * e2[:, np.newaxis])

    # Sandwich: bread @ meat @ bread
    return bread @ meat @ bread


def compute_vcov_hc1(
    X: Float64Array,
    residuals: Float64Array,
    bread: Float64Array | None = None,
) -> Float64Array:
    """Compute HC1 heteroskedasticity-robust variance-covariance.

    V = (n / (n - p)) * HC0

    HC1 applies a degrees-of-freedom correction to HC0.
    This is the default in Stata's `robust` option.

    Parameters
    ----------
    X : Float64Array
        Design matrix (n, p).
    residuals : Float64Array
        Model residuals (n,).
    bread : Float64Array, optional
        Pre-computed (X'X)^{-1}.

    Returns
    -------
    Float64Array
        Variance-covariance matrix (p, p).
    """
    n, p = X.shape
    hc0 = compute_vcov_hc0(X, residuals, bread)
    return (n / (n - p)) * hc0


def compute_vcov_hc2(
    X: Float64Array,
    residuals: Float64Array,
    bread: Float64Array | None = None,
) -> Float64Array:
    """Compute HC2 heteroskedasticity-robust variance-covariance.

    V = (X'X)^{-1} X' diag(e^2 / (1 - h_ii)) X (X'X)^{-1}

    HC2 adjusts residuals by leverage, providing less bias than HC0/HC1.

    Parameters
    ----------
    X : Float64Array
        Design matrix (n, p).
    residuals : Float64Array
        Model residuals (n,).
    bread : Float64Array, optional
        Pre-computed (X'X)^{-1}.

    Returns
    -------
    Float64Array
        Variance-covariance matrix (p, p).
    """
    if bread is None:
        bread = compute_bread_matrix(X)

    h = compute_leverage(X, bread)
    # Prevent division by zero for high-leverage points
    adjustment = 1.0 / np.maximum(1 - h, 1e-10)
    e2_adj = (residuals**2) * adjustment

    meat = X.T @ (X * e2_adj[:, np.newaxis])
    return bread @ meat @ bread


def compute_vcov_hc3(
    X: Float64Array,
    residuals: Float64Array,
    bread: Float64Array | None = None,
) -> Float64Array:
    """Compute HC3 heteroskedasticity-robust variance-covariance.

    V = (X'X)^{-1} X' diag(e^2 / (1 - h_ii)^2) X (X'X)^{-1}

    HC3 is the most conservative estimator, recommended for small samples.
    It corresponds to a jackknife estimator.

    Parameters
    ----------
    X : Float64Array
        Design matrix (n, p).
    residuals : Float64Array
        Model residuals (n,).
    bread : Float64Array, optional
        Pre-computed (X'X)^{-1}.

    Returns
    -------
    Float64Array
        Variance-covariance matrix (p, p).
    """
    if bread is None:
        bread = compute_bread_matrix(X)

    h = compute_leverage(X, bread)
    # Prevent division by zero for high-leverage points
    adjustment = 1.0 / np.maximum((1 - h) ** 2, 1e-10)
    e2_adj = (residuals**2) * adjustment

    meat = X.T @ (X * e2_adj[:, np.newaxis])
    return bread @ meat @ bread


def compute_vcov(
    X: Float64Array,
    residuals: Float64Array,
    se_type: SEType = "HC1",
    cluster: Float64Array | None = None,
) -> Float64Array:
    """Compute variance-covariance matrix with specified standard error type.

    Parameters
    ----------
    X : Float64Array
        Design matrix (n, p).
    residuals : Float64Array
        Model residuals (n,).
    se_type : str, default="HC1"
        Standard error type: "iid", "HC0", "HC1", "HC2", "HC3", "cluster".
    cluster : Float64Array, optional
        Cluster identifiers for clustered SEs. Required if se_type="cluster".

    Returns
    -------
    Float64Array
        Variance-covariance matrix (p, p).

    Raises
    ------
    ValueError
        If se_type is unknown or cluster is missing for clustered SEs.

    Examples
    --------
    >>> vcov = compute_vcov(X, residuals, se_type="HC1")
    >>> std_errors = np.sqrt(np.diag(vcov))
    """
    bread = compute_bread_matrix(X)

    if se_type == "iid":
        return compute_vcov_iid(X, residuals, bread)
    elif se_type == "HC0":
        return compute_vcov_hc0(X, residuals, bread)
    elif se_type == "HC1":
        return compute_vcov_hc1(X, residuals, bread)
    elif se_type == "HC2":
        return compute_vcov_hc2(X, residuals, bread)
    elif se_type == "HC3":
        return compute_vcov_hc3(X, residuals, bread)
    elif se_type == "cluster":
        if cluster is None:
            raise ValueError("cluster array required for se_type='cluster'")
        return compute_vcov_cluster(X, residuals, cluster, bread)
    else:
        raise ValueError(
            f"Unknown se_type: '{se_type}'. "
            "Choose from: 'iid', 'HC0', 'HC1', 'HC2', 'HC3', 'cluster'"
        )


def compute_vcov_cluster(
    X: Float64Array,
    residuals: Float64Array,
    cluster: Float64Array,
    bread: Float64Array | None = None,
) -> Float64Array:
    """Compute cluster-robust variance-covariance matrix.

    V = (X'X)^{-1} B (X'X)^{-1}
    where B = sum_g (X_g' e_g)(X_g' e_g)'

    Parameters
    ----------
    X : Float64Array
        Design matrix (n, p).
    residuals : Float64Array
        Model residuals (n,).
    cluster : Float64Array
        Cluster identifiers (n,).
    bread : Float64Array, optional
        Pre-computed (X'X)^{-1}.

    Returns
    -------
    Float64Array
        Variance-covariance matrix (p, p).
    """
    n, p = X.shape
    if bread is None:
        bread = compute_bread_matrix(X)

    unique_clusters = np.unique(cluster)
    G = len(unique_clusters)

    # Compute meat matrix
    meat = np.zeros((p, p))
    for g in unique_clusters:
        mask = cluster == g
        X_g = X[mask]
        e_g = residuals[mask]
        score_g = X_g.T @ e_g  # (p,)
        meat += np.outer(score_g, score_g)

    # Small sample correction: G/(G-1) * (n-1)/(n-p)
    correction = (G / (G - 1)) * ((n - 1) / (n - p))

    return correction * bread @ meat @ bread
