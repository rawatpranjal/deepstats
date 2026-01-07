"""Core DML algorithm with proper cross-fitting and splitting."""

import torch
from torch import Tensor
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Tuple

from .autodiff import compute_gradient, compute_hessian, detect_theta_dependence
from .lambda_estimator import LambdaEstimator, AggregateLambdaEstimator
from ..models import StructuralNet, train_structural_net, TrainingHistory
from ..utils import batch_inverse


@dataclass
class DMLResult:
    """Result of structural DML estimation."""

    mu_hat: float                          # Point estimate
    se: float                              # Standard error
    ci_lower: float                        # 95% CI lower bound
    ci_upper: float                        # 95% CI upper bound
    psi_values: np.ndarray                 # Influence function values
    theta_hat: np.ndarray                  # Estimated parameters theta(x) for all x
    mu_naive: float                        # Naive estimate (for comparison)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def structural_dml_core(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    target_fn: Callable[[Tensor, Tensor], Tensor],
    theta_dim: int,
    n_folds: int = 50,
    hidden_dims: List[int] = [64, 32],
    epochs: int = 100,
    lr: float = 0.01,
    three_way: Optional[bool] = None,
    gradient_fn: Optional[Callable] = None,
    hessian_fn: Optional[Callable] = None,
    per_obs_target_fn: Optional[Callable] = None,
    per_obs_target_grad_fn: Optional[Callable] = None,
    ridge: float = 1e-4,
    lambda_method: str = 'mlp',
    verbose: bool = False,
) -> DMLResult:
    """
    Core structural DML algorithm with proper cross-fitting.

    Algorithm:
    1. Split data into K folds
    2. For each fold k:
       a. Train theta_k on training data (or theta-split for 3-way)
       b. Compute Hessians on lambda-split
       c. Fit Lambda_k(x) nonparametrically (or aggregate for 2-way)
       d. For held-out i in I_k:
          - Compute psi_i = H - H_theta @ Lambda_k(x_i)^{-1} @ l_theta
    3. Aggregate: mu_hat = mean(psi), SE via within-fold formula

    Args:
        Y: (n,) outcomes
        T: (n,) treatments
        X: (n, d_x) covariates
        loss_fn: Structural loss (y, t, theta) -> (n,) losses
        target_fn: Target function (x, theta) -> scalar
        theta_dim: Dimension of parameter vector
        n_folds: Number of cross-validation folds
        hidden_dims: Neural network hidden layer sizes
        epochs: Training epochs
        lr: Learning rate
        three_way: Use three-way splitting (auto-detect if None)
        gradient_fn: Optional closed-form gradient (falls back to autodiff)
        hessian_fn: Optional closed-form Hessian (falls back to autodiff)
        per_obs_target_fn: Per-observation target h(theta_i) (default: beta_i)
        per_obs_target_grad_fn: Gradient of h w.r.t. theta (default: (0, 1))
        ridge: Ridge regularization for Hessian inversion
        lambda_method: Method for Lambda estimation ('mlp', 'rf', 'ridge')
        verbose: Print progress

    Returns:
        DMLResult with estimates and diagnostics
    """
    n = len(Y)
    d_x = X.shape[1]

    # Convert to tensors
    Y_t = torch.tensor(Y, dtype=torch.float32)
    T_t = torch.tensor(T, dtype=torch.float32)
    X_t = torch.tensor(X, dtype=torch.float32)

    # Auto-detect three-way splitting need
    if three_way is None:
        three_way = detect_theta_dependence(loss_fn, Y_t[:100], T_t[:100], theta_dim)
        if verbose:
            print(f"Auto-detected three_way={three_way}")

    # Create fold indices
    fold_indices = np.zeros(n, dtype=np.int32)
    perm = np.random.permutation(n)
    fold_size = n // n_folds

    for k in range(n_folds):
        start = k * fold_size
        if k == n_folds - 1:
            end = n
        else:
            end = (k + 1) * fold_size
        fold_indices[perm[start:end]] = k

    # Storage for results
    psi_values = np.zeros(n)
    theta_hat_all = np.zeros((n, theta_dim))
    corrections = np.zeros(n)
    lambda_cond_numbers = []
    lambda_min_eigenvalues = []
    n_regularized = 0  # Count of observations needing extra regularization
    histories = []

    # Default per-obs target: h(theta) = beta
    if per_obs_target_fn is None:
        def per_obs_target_fn(theta):
            return theta[:, 1]

    # Default per-obs target gradient: (0, 1)
    if per_obs_target_grad_fn is None:
        def per_obs_target_grad_fn(theta):
            n = theta.shape[0]
            grad = torch.zeros(n, theta_dim, dtype=theta.dtype, device=theta.device)
            grad[:, 1] = 1.0
            return grad

    # Cross-fitting loop
    for k in range(n_folds):
        if verbose:
            print(f"Processing fold {k+1}/{n_folds}")

        # Get fold indices
        eval_mask = fold_indices == k
        train_mask = ~eval_mask

        eval_idx = np.where(eval_mask)[0]
        train_idx = np.where(train_mask)[0]

        X_eval = X_t[eval_idx]
        T_eval = T_t[eval_idx]
        Y_eval = Y_t[eval_idx]

        X_train = X_t[train_idx]
        T_train = T_t[train_idx]
        Y_train = Y_t[train_idx]

        # Three-way splitting: split train into theta-train and lambda-train
        if three_way:
            n_train = len(train_idx)
            n_theta = int(0.6 * n_train)
            perm_train = np.random.permutation(n_train)
            theta_idx_local = perm_train[:n_theta]
            lambda_idx_local = perm_train[n_theta:]

            X_theta = X_train[theta_idx_local]
            T_theta = T_train[theta_idx_local]
            Y_theta = Y_train[theta_idx_local]

            X_lambda = X_train[lambda_idx_local]
            T_lambda = T_train[lambda_idx_local]
            Y_lambda = Y_train[lambda_idx_local]
        else:
            # Two-way: use all training data for both
            X_theta = X_train
            T_theta = T_train
            Y_theta = Y_train

            X_lambda = X_train
            T_lambda = T_train
            Y_lambda = Y_train

        # Train structural network
        model = StructuralNet(
            input_dim=d_x,
            theta_dim=theta_dim,
            hidden_dims=hidden_dims,
        )

        history = train_structural_net(
            model=model,
            X=X_theta,
            T=T_theta,
            Y=Y_theta,
            loss_fn=loss_fn,
            epochs=epochs,
            lr=lr,
            verbose=False,
        )
        histories.append(history)

        # Get theta predictions on lambda data and eval data
        model.eval()
        with torch.no_grad():
            theta_lambda = model(X_lambda)
            theta_eval = model(X_eval)

        # Compute Hessians on lambda data
        if hessian_fn is not None:
            hessians_lambda = hessian_fn(Y_lambda, T_lambda, theta_lambda)
        else:
            hessians_lambda = compute_hessian(loss_fn, Y_lambda, T_lambda, theta_lambda)

        # Fit Lambda estimator
        if three_way:
            if lambda_method == 'aggregate':
                # Use aggregate even for three-way (ensures full-rank for binary T)
                lambda_est = AggregateLambdaEstimator(theta_dim=theta_dim)
                lambda_est.fit(X_lambda, hessians_lambda)
            else:
                # Nonparametric Lambda(x)
                lambda_est = LambdaEstimator(method=lambda_method, theta_dim=theta_dim)
                lambda_est.fit(X_lambda, hessians_lambda)
            Lambda_eval = lambda_est.predict(X_eval)  # (n_eval, d, d)
        else:
            # Aggregate Lambda (same for all x)
            lambda_est = AggregateLambdaEstimator(theta_dim=theta_dim)
            lambda_est.fit(X_lambda, hessians_lambda)
            Lambda_eval = lambda_est.predict(X_eval)  # (n_eval, d, d)

        # Invert Lambda matrices
        Lambda_inv_eval = batch_inverse(Lambda_eval, ridge=ridge)

        # Track condition numbers and min eigenvalues
        for i in range(len(Lambda_eval)):
            s = torch.linalg.svdvals(Lambda_eval[i])
            if s.min() > 1e-10:
                cond = (s.max() / s.min()).item()
            else:
                cond = float('inf')
            lambda_cond_numbers.append(cond)

            # Track minimum eigenvalue
            try:
                eigvals = torch.linalg.eigvalsh(Lambda_eval[i])
                min_eig = eigvals.min().item()
                lambda_min_eigenvalues.append(min_eig)
                if min_eig < 1e-6:
                    n_regularized += 1
            except RuntimeError:
                lambda_min_eigenvalues.append(0.0)
                n_regularized += 1

        # Compute gradients on eval data
        # Need theta with grad for autodiff
        theta_eval_grad = theta_eval.clone().requires_grad_(True)

        if gradient_fn is not None:
            l_theta_eval = gradient_fn(Y_eval, T_eval, theta_eval)
        else:
            l_theta_eval = compute_gradient(loss_fn, Y_eval, T_eval, theta_eval)

        # Compute per-observation target and gradient
        h_eval = per_obs_target_fn(theta_eval)  # (n_eval,)
        h_grad_eval = per_obs_target_grad_fn(theta_eval)  # (n_eval, d) or (d,)

        # Handle both per-obs and constant target gradient
        if h_grad_eval.dim() == 1:
            h_grad_eval = h_grad_eval.unsqueeze(0).expand(len(eval_idx), -1)

        # Compute influence function: psi = h - h_grad @ Lambda_inv @ l_theta
        for i in range(len(eval_idx)):
            global_idx = eval_idx[i]

            h_i = h_eval[i].item()
            h_grad_i = h_grad_eval[i]  # (d,)
            Lambda_inv_i = Lambda_inv_eval[i]  # (d, d)
            l_theta_i = l_theta_eval[i]  # (d,)

            # Correction term: h_grad @ Lambda_inv @ l_theta
            correction_i = (h_grad_i @ Lambda_inv_i @ l_theta_i).item()

            psi_i = h_i - correction_i

            psi_values[global_idx] = psi_i
            theta_hat_all[global_idx] = theta_eval[i].detach().numpy()
            corrections[global_idx] = correction_i

    # Aggregate results
    mu_hat = psi_values.mean()

    # Within-fold variance (paper formula)
    variance_sum = 0.0
    for k in range(n_folds):
        psi_k = psi_values[fold_indices == k]
        mu_k = psi_k.mean()
        variance_k = ((psi_k - mu_k) ** 2).mean()
        variance_sum += variance_k

    psi_variance = variance_sum / n_folds
    se = np.sqrt(psi_variance / n)

    # 95% CI
    ci_lower = mu_hat - 1.96 * se
    ci_upper = mu_hat + 1.96 * se

    # Naive estimate for comparison
    mu_naive = theta_hat_all[:, 1].mean()  # Just average beta

    # Diagnostics
    min_lambda_eigenvalue = min(lambda_min_eigenvalues) if lambda_min_eigenvalues else 0.0
    mean_lambda_eigenvalue = np.mean(lambda_min_eigenvalues) if lambda_min_eigenvalues else 0.0

    diagnostics = {
        'lambda_cond_numbers': lambda_cond_numbers,
        'mean_cond_number': np.mean([c for c in lambda_cond_numbers if c < float('inf')]),
        'min_lambda_eigenvalue': min_lambda_eigenvalue,
        'mean_lambda_eigenvalue': mean_lambda_eigenvalue,
        'n_regularized': n_regularized,
        'pct_regularized': 100 * n_regularized / n if n > 0 else 0,
        'correction_mean': corrections.mean(),
        'correction_std': corrections.std(),
        'correction_ratio': corrections.std() / se if se > 0 else 0,
        'three_way': three_way,
        'n_folds': n_folds,
        'histories': histories,
    }

    # Warning for potential instability
    import warnings
    if n_regularized > 0.1 * n:
        warnings.warn(
            f"High Lambda regularization rate ({diagnostics['pct_regularized']:.1f}% of observations). "
            "This may indicate numerical instability. Consider: "
            "(1) larger sample size, (2) more regularization, or (3) simpler model.",
            UserWarning
        )

    if diagnostics['correction_ratio'] > 2.0:
        warnings.warn(
            f"High correction variance ratio ({diagnostics['correction_ratio']:.2f}). "
            "This suggests the influence function correction dominates the estimate variance. "
            "Consider using more cross-fitting folds (K >= 50).",
            UserWarning
        )

    return DMLResult(
        mu_hat=mu_hat,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        psi_values=psi_values,
        theta_hat=theta_hat_all,
        mu_naive=mu_naive,
        diagnostics=diagnostics,
    )


def compute_coverage(mu_true: float, result: DMLResult) -> bool:
    """Check if true mu is within 95% CI."""
    return result.ci_lower <= mu_true <= result.ci_upper


def compute_se_ratio(mu_true: float, results: List[DMLResult]) -> float:
    """
    Compute SE calibration ratio.

    SE_ratio = mean(estimated SE) / empirical SD of point estimates
    Should be close to 1.0 for well-calibrated SEs.
    """
    mu_hats = np.array([r.mu_hat for r in results])
    se_hats = np.array([r.se for r in results])

    empirical_sd = mu_hats.std()
    mean_se = se_hats.mean()

    return mean_se / empirical_sd if empirical_sd > 0 else float('inf')
