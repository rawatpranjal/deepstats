"""Automatic differentiation utilities for structural deep learning."""

import torch
from torch import Tensor
from typing import Callable, Optional


def compute_gradient(
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    y: Tensor,
    t: Tensor,
    theta: Tensor,
) -> Tensor:
    """
    Compute gradient of loss with respect to theta via autodiff.

    ℓ_θ = ∂ℓ/∂θ

    Args:
        loss_fn: Loss function (y, t, theta) -> (n,) per-observation losses
        y: (n,) outcomes
        t: (n,) treatments
        theta: (n, d_theta) parameters

    Returns:
        (n, d_theta) gradient tensor
    """
    n, d_theta = theta.shape

    # Ensure theta requires grad
    theta_grad = theta.clone().requires_grad_(True)

    # Compute per-observation losses
    losses = loss_fn(y, t, theta_grad)  # (n,)

    # Compute gradient for each observation
    gradients = torch.zeros(n, d_theta, dtype=theta.dtype, device=theta.device)

    for i in range(n):
        if theta_grad.grad is not None:
            theta_grad.grad.zero_()

        # Backward on single observation's loss
        losses[i].backward(retain_graph=(i < n - 1))
        gradients[i] = theta_grad.grad[i].clone()

    return gradients


def compute_gradient_vectorized(
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    y: Tensor,
    t: Tensor,
    theta: Tensor,
) -> Tensor:
    """
    Vectorized gradient computation using jacobian.

    More efficient for small batches.

    Args:
        loss_fn: Loss function (y, t, theta) -> (n,) per-observation losses
        y: (n,) outcomes
        t: (n,) treatments
        theta: (n, d_theta) parameters

    Returns:
        (n, d_theta) gradient tensor
    """
    theta_grad = theta.clone().requires_grad_(True)
    losses = loss_fn(y, t, theta_grad)  # (n,)

    # Sum losses and compute gradient (gives sum of per-obs gradients)
    # This works because loss is separable across observations
    total_loss = losses.sum()
    grad = torch.autograd.grad(total_loss, theta_grad)[0]

    # For separable losses, this gives correct per-observation gradients
    # since ∂(Σℓᵢ)/∂θᵢ = ∂ℓᵢ/∂θᵢ when θᵢ only affects ℓᵢ
    return grad


def compute_hessian(
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    y: Tensor,
    t: Tensor,
    theta: Tensor,
) -> Tensor:
    """
    Compute Hessian of loss with respect to theta via autodiff.

    ℓ_θθ = ∂²ℓ/∂θ²

    Args:
        loss_fn: Loss function (y, t, theta) -> (n,) per-observation losses
        y: (n,) outcomes
        t: (n,) treatments
        theta: (n, d_theta) parameters

    Returns:
        (n, d_theta, d_theta) Hessian tensor (per observation)
    """
    n, d_theta = theta.shape
    hessians = torch.zeros(n, d_theta, d_theta, dtype=theta.dtype, device=theta.device)

    for i in range(n):
        # Get single observation
        y_i = y[i:i+1]
        t_i = t[i:i+1]
        theta_i = theta[i:i+1].clone().requires_grad_(True)

        # Compute loss for this observation
        loss_i = loss_fn(y_i, t_i, theta_i)[0]

        # Compute gradient
        grad_i = torch.autograd.grad(loss_i, theta_i, create_graph=True)[0][0]  # (d_theta,)

        # Compute Hessian row by row
        for j in range(d_theta):
            if theta_i.grad is not None:
                theta_i.grad.zero_()

            grad_j = torch.autograd.grad(
                grad_i[j], theta_i, retain_graph=(j < d_theta - 1)
            )[0][0]  # (d_theta,)

            hessians[i, j, :] = grad_j

    return hessians


def compute_hessian_functional(
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    y: Tensor,
    t: Tensor,
    theta: Tensor,
) -> Tensor:
    """
    Compute Hessian using torch.func for better efficiency.

    Args:
        loss_fn: Loss function (y, t, theta) -> (n,) per-observation losses
        y: (n,) outcomes
        t: (n,) treatments
        theta: (n, d_theta) parameters

    Returns:
        (n, d_theta, d_theta) Hessian tensor
    """
    n, d_theta = theta.shape
    hessians = torch.zeros(n, d_theta, d_theta, dtype=theta.dtype, device=theta.device)

    # Define loss for single observation
    def single_loss(theta_single, y_single, t_single):
        return loss_fn(y_single.unsqueeze(0), t_single.unsqueeze(0), theta_single.unsqueeze(0))[0]

    for i in range(n):
        # Use torch.func.hessian if available
        try:
            from torch.func import hessian
            hess_fn = hessian(lambda th: single_loss(th, y[i], t[i]))
            hessians[i] = hess_fn(theta[i])
        except ImportError:
            # Fall back to manual computation
            theta_i = theta[i:i+1].clone().requires_grad_(True)
            loss_i = loss_fn(y[i:i+1], t[i:i+1], theta_i)[0]
            grad_i = torch.autograd.grad(loss_i, theta_i, create_graph=True)[0][0]

            for j in range(d_theta):
                grad_j = torch.autograd.grad(grad_i[j], theta_i, retain_graph=(j < d_theta - 1))[0][0]
                hessians[i, j, :] = grad_j

    return hessians


def compute_target_gradient(
    target_fn: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    theta: Tensor,
) -> Tensor:
    """
    Compute gradient of target function with respect to theta.

    H_θ = ∂H/∂θ

    For average target H = (1/n)Σh(θᵢ), this returns per-observation gradients.

    Args:
        target_fn: Target function (x, theta) -> scalar target value
        x: (n, d_x) covariates
        theta: (n, d_theta) parameters

    Returns:
        (n, d_theta) gradient tensor, or (d_theta,) if target is scalar
    """
    n, d_theta = theta.shape

    theta_grad = theta.clone().requires_grad_(True)
    target_val = target_fn(x, theta_grad)

    # Compute gradient
    grad = torch.autograd.grad(target_val, theta_grad)[0]

    return grad


def compute_per_obs_target_gradient(
    h_fn: Callable[[Tensor], Tensor],
    theta: Tensor,
) -> Tensor:
    """
    Compute per-observation gradient of h(θ) with respect to θ.

    For H = (1/n)Σh(θᵢ), we need ∂h(θᵢ)/∂θᵢ for each i.

    Args:
        h_fn: Per-observation target h(theta_i) -> scalar
        theta: (n, d_theta) parameters

    Returns:
        (n, d_theta) gradient tensor
    """
    n, d_theta = theta.shape
    gradients = torch.zeros(n, d_theta, dtype=theta.dtype, device=theta.device)

    for i in range(n):
        theta_i = theta[i:i+1].clone().requires_grad_(True)
        h_i = h_fn(theta_i)
        grad_i = torch.autograd.grad(h_i, theta_i)[0]
        gradients[i] = grad_i[0]

    return gradients


def detect_theta_dependence(
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    y_sample: Tensor,
    t_sample: Tensor,
    theta_dim: int,
    n_test: int = 10,
) -> bool:
    """
    Automatically detect if the Hessian depends on theta.

    If ℓ_θθ varies with θ, we need three-way splitting.

    Args:
        loss_fn: Loss function (y, t, theta) -> (n,) losses
        y_sample: Sample outcomes for testing
        t_sample: Sample treatments for testing
        theta_dim: Dimension of parameter vector
        n_test: Number of test points

    Returns:
        True if Hessian depends on theta (need three-way splitting)
    """
    n = min(len(y_sample), n_test)
    y = y_sample[:n]
    t = t_sample[:n]

    # Create two different theta values
    theta_1 = torch.randn(n, theta_dim)
    theta_2 = torch.randn(n, theta_dim) * 2 + 1  # Different scale and shift

    # Compute Hessians at each
    H_1 = compute_hessian(loss_fn, y, t, theta_1)
    H_2 = compute_hessian(loss_fn, y, t, theta_2)

    # Check if they differ
    # Use relative tolerance for numerical stability
    rel_diff = torch.abs(H_1 - H_2) / (torch.abs(H_1) + 1e-8)
    max_rel_diff = rel_diff.max().item()

    # If Hessians differ by more than 1%, they depend on theta
    return max_rel_diff > 0.01
