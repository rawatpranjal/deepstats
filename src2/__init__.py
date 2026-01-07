"""
src2: Clean Structural Deep Learning Implementation

A from-scratch implementation of the Farrell-Liang-Misra framework
for structural deep learning with valid inference.

Usage:
    # Pre-built family
    from src2 import structural_dml
    result = structural_dml(Y, T, X, family='logit')

    # Custom loss function
    def my_loss(y, t, theta):
        alpha, beta = theta[:, 0], theta[:, 1]
        mu = alpha + beta * t
        return (y - mu) ** 2

    result = structural_dml(Y, T, X, loss_fn=my_loss, theta_dim=2)

    # Access results
    print(f"Estimate: {result.mu_hat:.4f} +/- {result.se:.4f}")
    print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
"""

import numpy as np
from typing import Callable, List, Optional
from torch import Tensor

from .core import structural_dml_core, DMLResult
from .families import get_family, FAMILY_REGISTRY, BaseFamily


def structural_dml(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    family: Optional[str] = None,
    loss_fn: Optional[Callable] = None,
    target_fn: Optional[Callable] = None,
    theta_dim: Optional[int] = None,
    n_folds: int = 20,
    hidden_dims: List[int] = [64, 32],
    epochs: int = 100,
    lr: float = 0.01,
    verbose: bool = False,
    **kwargs,
) -> DMLResult:
    """
    Structural deep learning with valid inference.

    Implements the Farrell-Liang-Misra framework for estimating
    heterogeneous structural parameters with neural networks,
    using influence functions for valid inference.

    Args:
        Y: (n,) outcome vector
        T: (n,) treatment vector
        X: (n, d) covariate matrix
        family: Pre-built family name ('linear', 'logit', etc.)
        loss_fn: Custom loss function (y, t, theta) -> (n,) losses
        target_fn: Custom target function (x, theta) -> scalar
        theta_dim: Dimension of parameter vector (required if custom loss)
        n_folds: Number of cross-fitting folds (recommend 20-50)
        hidden_dims: Neural network hidden layer sizes
        epochs: Training epochs per fold
        lr: Learning rate
        verbose: Print progress
        **kwargs: Additional arguments to structural_dml_core

    Returns:
        DMLResult with:
            - mu_hat: Point estimate
            - se: Standard error
            - ci_lower, ci_upper: 95% confidence interval
            - psi_values: Influence function values
            - theta_hat: Estimated parameters for all observations
            - diagnostics: Training and estimation diagnostics

    Examples:
        # Binary outcome with heterogeneous effects
        result = structural_dml(Y, T, X, family='logit', n_folds=50)

        # Continuous outcome
        result = structural_dml(Y, T, X, family='linear')

        # Custom structural model
        def tobit_loss(y, t, theta):
            import torch
            alpha, beta = theta[:, 0], theta[:, 1]
            mu = alpha + beta * t
            sigma = 1.0
            # Tobit log-likelihood
            censored = (y <= 0).float()
            uncensored = 1 - censored
            z = -mu / sigma
            ll = censored * torch.distributions.Normal(0, 1).cdf(z).log()
            ll += uncensored * (-0.5 * ((y - mu) / sigma) ** 2 - 0.5 * np.log(2 * np.pi) - np.log(sigma))
            return -ll

        result = structural_dml(Y, T, X, loss_fn=tobit_loss, theta_dim=2)
    """
    # Validate inputs
    if family is None and loss_fn is None:
        raise ValueError("Must provide either 'family' or 'loss_fn'")

    if family is not None and loss_fn is not None:
        raise ValueError("Cannot provide both 'family' and 'loss_fn'")

    if loss_fn is not None and theta_dim is None:
        raise ValueError("Must provide 'theta_dim' when using custom loss_fn")

    # Get family or use custom functions
    if family is not None:
        fam = get_family(family)
        loss_fn = fam.loss
        theta_dim = fam.theta_dim
        three_way = fam.hessian_depends_on_theta()

        # Use closed-form functions if available
        gradient_fn = fam.gradient if hasattr(fam, 'gradient') and fam.gradient(
            Tensor([0.0]), Tensor([0.0]), Tensor([[0.0, 0.0]])
        ) is not None else None

        hessian_fn = fam.hessian if hasattr(fam, 'hessian') and fam.hessian(
            Tensor([0.0]), Tensor([0.0]), Tensor([[0.0, 0.0]])
        ) is not None else None

        # Use family's target functions
        target_fn = fam.default_target
        per_obs_target_fn = fam.per_obs_target
        per_obs_target_grad_fn = fam.per_obs_target_gradient
    else:
        # Fully automatic mode
        three_way = kwargs.pop('three_way', None)  # Auto-detect
        gradient_fn = None
        hessian_fn = None
        per_obs_target_fn = None
        per_obs_target_grad_fn = None

        # Default target if not provided
        if target_fn is None:
            def target_fn(x, theta):
                return theta[:, 1].mean()

    return structural_dml_core(
        Y=Y,
        T=T,
        X=X,
        loss_fn=loss_fn,
        target_fn=target_fn,
        theta_dim=theta_dim,
        n_folds=n_folds,
        hidden_dims=hidden_dims,
        epochs=epochs,
        lr=lr,
        three_way=three_way,
        gradient_fn=gradient_fn,
        hessian_fn=hessian_fn,
        per_obs_target_fn=per_obs_target_fn,
        per_obs_target_grad_fn=per_obs_target_grad_fn,
        verbose=verbose,
        **kwargs,
    )


# Re-export key classes
from .core import DMLResult, compute_coverage, compute_se_ratio
from .families import LinearFamily, LogitFamily, BaseFamily

__all__ = [
    # Main API
    'structural_dml',
    # Result class
    'DMLResult',
    # Families
    'LinearFamily',
    'LogitFamily',
    'BaseFamily',
    'get_family',
    'FAMILY_REGISTRY',
    # Utilities
    'compute_coverage',
    'compute_se_ratio',
]
