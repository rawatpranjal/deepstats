"""
Score (gradient) computation via autodiff.

ℓ_θ = ∂ℓ/∂θ

Provides both loop-based and vmap-based implementations.
"""

import torch
from torch import Tensor
from typing import Callable, Optional


def compute_score(
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    y: Tensor,
    t: Tensor,
    theta: Tensor,
    use_vmap: bool = True,
) -> Tensor:
    """
    Compute per-observation gradient of loss w.r.t. theta.

    Args:
        loss_fn: Loss function (y, t, theta) -> scalar loss for single obs
                 Signature: loss_fn(y_i, t_i, theta_i) -> scalar
        y: (n,) outcomes
        t: (n,) or (n, d_t) treatments
        theta: (n, d_theta) parameters
        use_vmap: Whether to use vmap (faster) or loop (more compatible)

    Returns:
        (n, d_theta) gradient tensor
    """
    if use_vmap:
        return compute_score_vmap(loss_fn, y, t, theta)
    else:
        return compute_score_loop(loss_fn, y, t, theta)


def compute_score_vmap(
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    y: Tensor,
    t: Tensor,
    theta: Tensor,
) -> Tensor:
    """
    Batched gradient computation via vmap.

    Uses torch.func.grad + torch.vmap for efficient batched gradients.

    Args:
        loss_fn: Loss function for single observation
                 loss_fn(y_i, t_i, theta_i) -> scalar
        y: (n,) outcomes
        t: (n,) or (n, d_t) treatments
        theta: (n, d_theta) parameters

    Returns:
        (n, d_theta) gradient tensor
    """
    from torch.func import grad, vmap

    # Define gradient of loss w.r.t. theta for single observation
    def single_grad(y_i: Tensor, t_i: Tensor, theta_i: Tensor) -> Tensor:
        """Gradient for one observation."""
        return grad(lambda th: loss_fn(y_i, t_i, th))(theta_i)

    # vmap over batch dimension
    batched_grad = vmap(single_grad, in_dims=(0, 0, 0))

    return batched_grad(y, t, theta)


def compute_score_loop(
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    y: Tensor,
    t: Tensor,
    theta: Tensor,
) -> Tensor:
    """
    Loop-based gradient computation (fallback).

    Slower but more compatible with complex loss functions.

    Args:
        loss_fn: Loss function for single observation
        y: (n,) outcomes
        t: (n,) or (n, d_t) treatments
        theta: (n, d_theta) parameters

    Returns:
        (n, d_theta) gradient tensor
    """
    n = theta.shape[0]
    d_theta = theta.shape[1]
    device = theta.device
    dtype = theta.dtype

    gradients = torch.zeros(n, d_theta, dtype=dtype, device=device)

    for i in range(n):
        theta_i = theta[i].clone().requires_grad_(True)
        y_i = y[i]
        t_i = t[i]

        loss_i = loss_fn(y_i, t_i, theta_i)
        grad_i = torch.autograd.grad(loss_i, theta_i)[0]
        gradients[i] = grad_i

    return gradients
