"""
Hessian computation via autodiff.

ℓ_θθ = ∂²ℓ/∂θ²

Provides both loop-based and vmap-based implementations.
"""

import torch
from torch import Tensor
from typing import Callable


def compute_hessian(
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    y: Tensor,
    t: Tensor,
    theta: Tensor,
    use_vmap: bool = True,
) -> Tensor:
    """
    Compute per-observation Hessian of loss w.r.t. theta.

    Args:
        loss_fn: Loss function for single observation
                 loss_fn(y_i, t_i, theta_i) -> scalar
        y: (n,) outcomes
        t: (n,) or (n, d_t) treatments
        theta: (n, d_theta) parameters
        use_vmap: Whether to use vmap (faster) or loop (more compatible)

    Returns:
        (n, d_theta, d_theta) Hessian tensor
    """
    if use_vmap:
        return compute_hessian_vmap(loss_fn, y, t, theta)
    else:
        return compute_hessian_loop(loss_fn, y, t, theta)


def compute_hessian_vmap(
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    y: Tensor,
    t: Tensor,
    theta: Tensor,
) -> Tensor:
    """
    Batched Hessian computation via vmap.

    Uses torch.func.hessian + torch.vmap for efficient batched Hessians.

    Args:
        loss_fn: Loss function for single observation
                 loss_fn(y_i, t_i, theta_i) -> scalar
        y: (n,) outcomes
        t: (n,) or (n, d_t) treatments
        theta: (n, d_theta) parameters

    Returns:
        (n, d_theta, d_theta) Hessian tensor
    """
    from torch.func import hessian, vmap

    # Define Hessian of loss w.r.t. theta for single observation
    def single_hessian(y_i: Tensor, t_i: Tensor, theta_i: Tensor) -> Tensor:
        """Hessian for one observation."""
        return hessian(lambda th: loss_fn(y_i, t_i, th))(theta_i)

    # vmap over batch dimension
    batched_hessian = vmap(single_hessian, in_dims=(0, 0, 0))

    return batched_hessian(y, t, theta)


def compute_hessian_loop(
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    y: Tensor,
    t: Tensor,
    theta: Tensor,
) -> Tensor:
    """
    Loop-based Hessian computation (fallback).

    Slower but more compatible with complex loss functions.

    Args:
        loss_fn: Loss function for single observation
        y: (n,) outcomes
        t: (n,) or (n, d_t) treatments
        theta: (n, d_theta) parameters

    Returns:
        (n, d_theta, d_theta) Hessian tensor
    """
    n = theta.shape[0]
    d_theta = theta.shape[1]
    device = theta.device
    dtype = theta.dtype

    hessians = torch.zeros(n, d_theta, d_theta, dtype=dtype, device=device)

    for i in range(n):
        theta_i = theta[i].clone().requires_grad_(True)
        y_i = y[i]
        t_i = t[i]

        # Compute loss
        loss_i = loss_fn(y_i, t_i, theta_i)

        # Compute gradient with create_graph=True for second derivatives
        grad_i = torch.autograd.grad(loss_i, theta_i, create_graph=True)[0]

        # Compute Hessian row by row
        for j in range(d_theta):
            grad_j = torch.autograd.grad(
                grad_i[j], theta_i, retain_graph=(j < d_theta - 1)
            )[0]
            hessians[i, j, :] = grad_j.detach()

    return hessians


def detect_hessian_theta_dependence(
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    y_sample: Tensor,
    t_sample: Tensor,
    theta_dim: int,
    n_test: int = 10,
    tol: float = 0.01,
) -> bool:
    """
    Detect if Hessian depends on theta values.

    If ℓ_θθ varies with θ, we need three-way splitting for cross-fitting.

    Args:
        loss_fn: Loss function for single observation
        y_sample: Sample outcomes for testing
        t_sample: Sample treatments for testing
        theta_dim: Dimension of parameter vector
        n_test: Number of test points
        tol: Tolerance for detecting dependence

    Returns:
        True if Hessian depends on theta (need three-way splitting)
    """
    n = min(len(y_sample), n_test)
    y = y_sample[:n]
    t = t_sample[:n]

    # Create two different theta values
    theta_1 = torch.randn(n, theta_dim)
    theta_2 = torch.randn(n, theta_dim) * 2 + 1  # Different scale and shift

    # Compute Hessians at each (use loop for compatibility)
    H_1 = compute_hessian_loop(loss_fn, y, t, theta_1)
    H_2 = compute_hessian_loop(loss_fn, y, t, theta_2)

    # Check if they differ (relative tolerance)
    rel_diff = torch.abs(H_1 - H_2) / (torch.abs(H_1) + 1e-8)
    max_rel_diff = rel_diff.max().item()

    return max_rel_diff > tol
