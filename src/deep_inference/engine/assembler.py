"""
Influence function assembler.

Computes: ψ = H - H_θ @ Λ⁻¹ @ ℓ_θ

This is the core mathematical operation that enables valid inference.
"""

from typing import TYPE_CHECKING, Optional
import torch
from torch import Tensor

if TYPE_CHECKING:
    from deep_inference.models import StructuralModel
    from deep_inference.targets import Target


def assemble_influence(
    h: Tensor,
    h_jacobian: Tensor,
    lambda_inv: Tensor,
    score: Tensor,
) -> Tensor:
    """
    Assemble influence function values.

    ψ_i = h_i - h_jacobian_i @ lambda_inv_i @ score_i

    Args:
        h: (n,) target values H(x, θ, t̃)
        h_jacobian: (n, d_theta) target Jacobians H_θ
        lambda_inv: (n, d_theta, d_theta) inverse Lambda matrices
        score: (n, d_theta) score vectors ℓ_θ

    Returns:
        (n,) influence function values ψ
    """
    n = h.shape[0]

    # Compute correction term: H_θ @ Λ⁻¹ @ ℓ_θ
    # For each observation: (d_theta,) @ (d_theta, d_theta) @ (d_theta,)
    correction = torch.zeros_like(h)

    for i in range(n):
        # (d_theta,) @ (d_theta, d_theta) = (d_theta,)
        # (d_theta,) @ (d_theta,) = scalar
        temp = h_jacobian[i] @ lambda_inv[i]  # (d_theta,)
        correction[i] = temp @ score[i]  # scalar

    # ψ = H - correction
    psi = h - correction

    return psi


def assemble_influence_batched(
    h: Tensor,
    h_jacobian: Tensor,
    lambda_inv: Tensor,
    score: Tensor,
) -> Tensor:
    """
    Batched assembly using einsum.

    Faster than loop for large n.

    Args:
        h: (n,) target values
        h_jacobian: (n, d_theta) Jacobians
        lambda_inv: (n, d_theta, d_theta) inverse Lambdas
        score: (n, d_theta) scores

    Returns:
        (n,) influence function values
    """
    # H_θ @ Λ⁻¹: (n, d_theta) @ (n, d_theta, d_theta) -> (n, d_theta)
    # Use einsum: 'nd,nde->ne'
    temp = torch.einsum('nd,nde->ne', h_jacobian, lambda_inv)

    # temp @ score: (n, d_theta) * (n, d_theta) -> (n,) (dot product per row)
    correction = (temp * score).sum(dim=1)

    psi = h - correction

    return psi


def compute_psi(
    Y: Tensor,
    T: Tensor,
    X: Tensor,
    theta_hat: Tensor,
    t_tilde: Tensor,
    lambda_matrices: Tensor,
    model: "StructuralModel",
    target: "Target",
    ridge: float = 1e-4,
) -> Tensor:
    """
    Compute influence function values for a set of observations.

    Full pipeline:
    1. Compute target values h
    2. Compute target Jacobians H_θ
    3. Compute scores ℓ_θ
    4. Invert Lambda matrices
    5. Assemble: ψ = h - H_θ @ Λ⁻¹ @ ℓ_θ

    Args:
        Y: (n,) outcomes
        T: (n,) treatments
        X: (n, d_x) covariates
        theta_hat: (n, d_theta) estimated parameters
        t_tilde: Evaluation point (scalar or (n,))
        lambda_matrices: (n, d_theta, d_theta) Lambda matrices
        model: Structural model
        target: Target functional
        ridge: Regularization for matrix inversion

    Returns:
        (n,) influence function values
    """
    from deep_inference.utils.linalg import batch_inverse

    n = theta_hat.shape[0]
    d_theta = theta_hat.shape[1]
    device = theta_hat.device
    dtype = theta_hat.dtype

    # Handle t_tilde broadcasting
    if t_tilde.dim() == 0:
        t_tilde = t_tilde.expand(n)

    # 1. Compute target values h(x, θ, t̃)
    h = torch.zeros(n, dtype=dtype, device=device)
    for i in range(n):
        h[i] = target.h(X[i], theta_hat[i], t_tilde[i])

    # 2. Compute target Jacobians H_θ
    h_jacobian = torch.zeros(n, d_theta, dtype=dtype, device=device)
    for i in range(n):
        jac = target.jacobian(X[i], theta_hat[i], t_tilde[i])
        if jac is not None:
            h_jacobian[i] = jac
        else:
            # Fall back to autodiff
            theta_i = theta_hat[i].clone().requires_grad_(True)
            h_i = target.h(X[i], theta_i, t_tilde[i])
            grad_i = torch.autograd.grad(h_i, theta_i)[0]
            h_jacobian[i] = grad_i

    # 3. Compute scores ℓ_θ
    score = torch.zeros(n, d_theta, dtype=dtype, device=device)
    for i in range(n):
        s = model.score(Y[i], T[i], theta_hat[i])
        if s is not None:
            score[i] = s
        else:
            # Fall back to autodiff
            theta_i = theta_hat[i].clone().requires_grad_(True)
            loss_i = model.loss(Y[i], T[i], theta_i)
            grad_i = torch.autograd.grad(loss_i, theta_i)[0]
            score[i] = grad_i

    # 4. Invert Lambda matrices
    lambda_inv = batch_inverse(lambda_matrices, ridge=ridge)

    # 5. Assemble influence function
    psi = assemble_influence_batched(h, h_jacobian, lambda_inv, score)

    return psi


def compute_psi_vmap(
    Y: Tensor,
    T: Tensor,
    X: Tensor,
    theta_hat: Tensor,
    t_tilde: Tensor,
    lambda_matrices: Tensor,
    model: "StructuralModel",
    target: "Target",
    ridge: float = 1e-4,
) -> Tensor:
    """
    Compute influence function values using vmap for full batching.

    Faster than compute_psi for large n, but requires torch.func.

    Args:
        (same as compute_psi)

    Returns:
        (n,) influence function values
    """
    from deep_inference.utils.linalg import batch_inverse
    from deep_inference.autodiff.score import compute_score_vmap
    from deep_inference.autodiff.jacobian import compute_target_jacobian_vmap

    n = theta_hat.shape[0]

    # Handle t_tilde broadcasting
    if t_tilde.dim() == 0:
        t_tilde = t_tilde.expand(n)

    # 1. Compute target values (vmap)
    from torch.func import vmap
    h_single = lambda x, th, t: target.h(x, th, t)
    h = vmap(h_single)(X, theta_hat, t_tilde)

    # 2. Compute target Jacobians (vmap)
    h_jacobian = compute_target_jacobian_vmap(target.h, X, theta_hat, t_tilde)

    # 3. Compute scores (vmap)
    score = compute_score_vmap(model.loss, Y, T, theta_hat)

    # 4. Invert Lambda matrices
    lambda_inv = batch_inverse(lambda_matrices, ridge=ridge)

    # 5. Assemble
    psi = assemble_influence_batched(h, h_jacobian, lambda_inv, score)

    return psi
