"""Influence function standard error estimation.

This module implements influence function-based standard errors following
Farrell, Liang, Misra (2021) "Deep Neural Networks for Estimation and Inference".

The orthogonal score approach provides valid inference for average marginal
effects even with highly flexible neural network estimators.

Cross-Fitting for Bias Reduction
--------------------------------
Cross-fitting (Chernozhukov et al. 2018) splits data into K folds:
- Fit nuisance models on K-1 folds
- Predict on held-out fold
- Avoids overfitting bias when using flexible ML methods

This is implemented in `_compute_influence_se_crossfit()` which:
1. Splits data into K folds
2. For each fold, estimates Lambda (Hessian) from training folds only
3. Computes influence scores on held-out observations
4. Aggregates for final variance estimate

The cross-fitting approach is critical when using neural networks because:
- Neural nets can overfit to training data
- Using same data for nuisance estimation and inference introduces bias
- Cross-fitting breaks this dependence, ensuring valid coverage

This technique is also central to Double/Debiased Machine Learning (DML)
for causal inference. While DeepHTE uses a different structural approach
(enriched models Y = a(X) + b(X)*T), cross-fitting remains useful for
influence function inference on the average treatment effect.

References
----------
- Farrell, M. H., Liang, T., & Misra, S. (2021). "Deep Neural Networks for
  Estimation and Inference." Econometrica, 89(1), 181-213.
- Chernozhukov, V., et al. (2018). "Double/Debiased Machine Learning for
  Treatment and Structural Parameters." The Econometrics Journal, 21(1), C1-C68.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from .._typing import Float64Array


@dataclass
class InfluenceFunctionResult:
    """Container for influence function inference results.

    Attributes
    ----------
    se : ndarray
        Standard errors (p,).
    vcov : ndarray
        Variance-covariance matrix (p, p).
    influence_scores : ndarray
        Individual influence scores (n, p).
    lambda_matrix : ndarray
        Conditional Hessian estimate (p, p).
    """

    se: Float64Array
    vcov: Float64Array
    influence_scores: Float64Array
    lambda_matrix: Float64Array


def compute_influence_function_se(
    X: Float64Array,
    y: Float64Array,
    network: nn.Module,
    fitted_values: Float64Array,
    residuals: Float64Array,
    cross_fit: bool = True,
    n_folds: int = 5,
    random_state: int | None = None,
) -> InfluenceFunctionResult:
    """Compute influence function-based standard errors.

    Implements the orthogonal score approach from Farrell, Liang, Misra (2021)
    for valid inference on average marginal effects with neural networks.

    For the MSE loss with AME target H(x, theta) = theta(x):
        psi_i = theta_hat(x_i) - Lambda_hat^{-1} * l_theta(w_i)

    where l_theta is the gradient of the loss w.r.t. theta.

    Parameters
    ----------
    X : ndarray
        Feature matrix (n, p).
    y : ndarray
        Target vector (n,).
    network : nn.Module
        Trained neural network.
    fitted_values : ndarray
        Model predictions (n,).
    residuals : ndarray
        Model residuals y - fitted (n,).
    cross_fit : bool, default=True
        Whether to use cross-fitting to avoid overfitting bias.
    n_folds : int, default=5
        Number of cross-fitting folds if cross_fit=True.
    random_state : int, optional
        Random seed for cross-fitting.

    Returns
    -------
    InfluenceFunctionResult
        Influence function inference results.
    """
    device = next(network.parameters()).device
    n, p = X.shape

    if cross_fit:
        return _compute_influence_se_crossfit(
            X, y, network, residuals, n_folds, random_state, device
        )
    else:
        return _compute_influence_se_full(X, y, network, residuals, device)


def _compute_influence_se_full(
    X: Float64Array,
    y: Float64Array,
    network: nn.Module,
    residuals: Float64Array,
    device: torch.device,
) -> InfluenceFunctionResult:
    """Compute influence function SEs without cross-fitting.

    For MSE loss with target mu = E[theta(X)] where theta(x) are marginal effects:

    The orthogonal score is:
        psi_i = theta_hat(x_i) - Lambda^{-1} * l_theta_i

    For simple averaging (H = identity on theta), this becomes:
        psi_i = theta_hat(x_i) - Lambda^{-1} * (-2 * residual_i * gradient_i)

    where gradient_i = df(x_i)/dx_j (the individual marginal effects).
    """
    n, p = X.shape

    # Step 1: Compute individual marginal effects theta_hat(x_i)
    theta_i = _compute_individual_marginal_effects(X, network, device)  # (n, p)

    # Step 2: Compute loss gradients l_theta_i
    # For MSE: l_theta = -2 * residual * gradient
    l_theta = -2 * residuals[:, np.newaxis] * theta_i  # (n, p)

    # Step 3: Compute Lambda (average Hessian)
    # For MSE: Lambda = 2 * E[(gradient)^T @ gradient]
    Lambda = 2 * (theta_i.T @ theta_i) / n  # (p, p)

    # Step 4: Compute Lambda^{-1}
    Lambda_inv = np.linalg.pinv(Lambda)  # (p, p)

    # Step 5: Construct influence scores
    # psi_i = theta_i - Lambda^{-1} @ l_theta_i
    psi = theta_i - (l_theta @ Lambda_inv.T)  # (n, p)

    # Step 6: Compute variance from influence scores
    # Var(mu_hat) = (1/n) * Var(psi)
    psi_centered = psi - np.mean(psi, axis=0)
    vcov = (psi_centered.T @ psi_centered) / (n * (n - 1))  # (p, p)
    se = np.sqrt(np.diag(vcov))

    return InfluenceFunctionResult(
        se=se,
        vcov=vcov,
        influence_scores=psi,
        lambda_matrix=Lambda,
    )


def _compute_influence_se_crossfit(
    X: Float64Array,
    y: Float64Array,
    network: nn.Module,
    residuals: Float64Array,
    n_folds: int,
    random_state: int | None,
    device: torch.device,
) -> InfluenceFunctionResult:
    """Compute influence function SEs with cross-fitting.

    Cross-fitting avoids overfitting bias by computing influence scores
    on held-out folds using Lambda estimated from training folds.
    """
    n, p = X.shape
    psi_full = np.zeros((n, p))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for train_idx, test_idx in kf.split(X):
        # Compute individual marginal effects on both sets
        X_train = X[train_idx]
        X_test = X[test_idx]
        residuals_test = residuals[test_idx]

        theta_train = _compute_individual_marginal_effects(X_train, network, device)
        theta_test = _compute_individual_marginal_effects(X_test, network, device)

        # Compute Lambda from training data only
        Lambda = 2 * (theta_train.T @ theta_train) / len(train_idx)
        Lambda_inv = np.linalg.pinv(Lambda)

        # Loss gradient on test set
        l_theta_test = -2 * residuals_test[:, np.newaxis] * theta_test

        # Influence scores for test observations
        psi_full[test_idx] = theta_test - (l_theta_test @ Lambda_inv.T)

    # Compute SE from cross-fitted scores
    psi_centered = psi_full - np.mean(psi_full, axis=0)
    vcov = (psi_centered.T @ psi_centered) / (n * (n - 1))
    se = np.sqrt(np.diag(vcov))

    # Compute full-sample Lambda for reference
    theta_full = _compute_individual_marginal_effects(X, network, device)
    Lambda_full = 2 * (theta_full.T @ theta_full) / n

    return InfluenceFunctionResult(
        se=se,
        vcov=vcov,
        influence_scores=psi_full,
        lambda_matrix=Lambda_full,
    )


def _compute_individual_marginal_effects(
    X: Float64Array,
    network: nn.Module,
    device: torch.device,
    eps: float = 1e-4,
) -> Float64Array:
    """Compute marginal effects for each observation (not averaged).

    Parameters
    ----------
    X : ndarray
        Feature matrix (n, p).
    network : nn.Module
        Neural network.
    device : torch.device
        Computation device.
    eps : float
        Numerical differentiation step.

    Returns
    -------
    ndarray
        Individual marginal effects (n, p).
    """
    n, p = X.shape
    gradients = np.zeros((n, p))

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

            gradients[:, j] = (y_plus - y_minus) / (2 * eps)

    return gradients
