"""
Target Jacobian computation via autodiff.

H_θ = ∂H/∂θ

Provides both loop-based and vmap-based implementations.
"""

import torch
from torch import Tensor
from typing import Callable, Optional


def compute_target_jacobian(
    target_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    x: Tensor,
    theta: Tensor,
    t_tilde: Tensor,
    use_vmap: bool = True,
) -> Tensor:
    """
    Compute per-observation Jacobian of target w.r.t. theta.

    Args:
        target_fn: Target function for single observation
                   target_fn(x_i, theta_i, t_tilde_i) -> scalar or (d_mu,)
        x: (n, d_x) covariates
        theta: (n, d_theta) parameters
        t_tilde: (n,) or (n, d_t) or scalar evaluation point
        use_vmap: Whether to use vmap (faster) or loop (more compatible)

    Returns:
        (n, d_theta) if target is scalar
        (n, d_mu, d_theta) if target is vector
    """
    if use_vmap:
        return compute_target_jacobian_vmap(target_fn, x, theta, t_tilde)
    else:
        return compute_target_jacobian_loop(target_fn, x, theta, t_tilde)


def compute_target_jacobian_vmap(
    target_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    x: Tensor,
    theta: Tensor,
    t_tilde: Tensor,
) -> Tensor:
    """
    Batched Jacobian computation via vmap.

    Uses torch.func.jacrev + torch.vmap for efficient batched Jacobians.

    Args:
        target_fn: Target function for single observation
                   target_fn(x_i, theta_i, t_tilde_i) -> scalar or (d_mu,)
        x: (n, d_x) covariates
        theta: (n, d_theta) parameters
        t_tilde: (n,) or (n, d_t) or scalar evaluation point

    Returns:
        (n, d_theta) if target is scalar
        (n, d_mu, d_theta) if target is vector
    """
    from torch.func import jacrev, vmap

    n = theta.shape[0]

    # Handle scalar t_tilde (broadcast to all observations)
    if t_tilde.dim() == 0:
        t_tilde = t_tilde.expand(n)
    elif t_tilde.dim() == 1 and t_tilde.shape[0] == 1:
        t_tilde = t_tilde.expand(n)

    # Define Jacobian of target w.r.t. theta for single observation
    def single_jacobian(x_i: Tensor, theta_i: Tensor, t_i: Tensor) -> Tensor:
        """Jacobian for one observation."""
        return jacrev(lambda th: target_fn(x_i, th, t_i))(theta_i)

    # vmap over batch dimension
    batched_jacobian = vmap(single_jacobian, in_dims=(0, 0, 0))

    return batched_jacobian(x, theta, t_tilde)


def compute_target_jacobian_loop(
    target_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    x: Tensor,
    theta: Tensor,
    t_tilde: Tensor,
) -> Tensor:
    """
    Loop-based Jacobian computation (fallback).

    Slower but more compatible with complex target functions.

    Args:
        target_fn: Target function for single observation
        x: (n, d_x) covariates
        theta: (n, d_theta) parameters
        t_tilde: (n,) or (n, d_t) or scalar evaluation point

    Returns:
        (n, d_theta) if target is scalar
        (n, d_mu, d_theta) if target is vector
    """
    n = theta.shape[0]
    d_theta = theta.shape[1]
    device = theta.device
    dtype = theta.dtype

    # Handle scalar t_tilde
    if t_tilde.dim() == 0:
        t_tilde = t_tilde.expand(n)
    elif t_tilde.dim() == 1 and t_tilde.shape[0] == 1:
        t_tilde = t_tilde.expand(n)

    # Probe output dimension
    with torch.no_grad():
        sample_out = target_fn(x[0], theta[0], t_tilde[0])
    is_scalar = sample_out.dim() == 0
    d_mu = 1 if is_scalar else sample_out.shape[0]

    if is_scalar:
        jacobians = torch.zeros(n, d_theta, dtype=dtype, device=device)
    else:
        jacobians = torch.zeros(n, d_mu, d_theta, dtype=dtype, device=device)

    for i in range(n):
        theta_i = theta[i].clone().requires_grad_(True)
        x_i = x[i]
        t_i = t_tilde[i]

        h_i = target_fn(x_i, theta_i, t_i)

        if is_scalar:
            grad_i = torch.autograd.grad(h_i, theta_i)[0]
            jacobians[i] = grad_i
        else:
            # Vector output: compute Jacobian row by row
            for j in range(d_mu):
                grad_ij = torch.autograd.grad(
                    h_i[j], theta_i, retain_graph=(j < d_mu - 1)
                )[0]
                jacobians[i, j, :] = grad_ij

    return jacobians


def compute_gradient_for_scalar_target(
    target_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    x: Tensor,
    theta: Tensor,
    t_tilde: Tensor,
    use_vmap: bool = True,
) -> Tensor:
    """
    Compute gradient for scalar target (special case of Jacobian).

    When H is scalar, H_θ is a vector (gradient), not a matrix.

    Args:
        target_fn: Target function returning scalar
        x: (n, d_x) covariates
        theta: (n, d_theta) parameters
        t_tilde: Evaluation point

    Returns:
        (n, d_theta) gradient tensor
    """
    return compute_target_jacobian(target_fn, x, theta, t_tilde, use_vmap=use_vmap)


# Convenience alias
compute_H_theta = compute_target_jacobian
