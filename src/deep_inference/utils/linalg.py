"""Linear algebra utilities with numerical stability."""

import torch
from torch import Tensor
from typing import Optional, Literal
from enum import Enum


class RegularizationStrategy(Enum):
    """Strategy for regularizing matrix inversion."""

    ABSOLUTE = "absolute"
    """Current behavior: add ridge when min_eig < threshold."""

    RELATIVE = "relative"
    """Bound condition number: min_eig >= max_eig / max_condition."""

    TIKHONOV = "tikhonov"
    """Scale-aware Tikhonov: (Λ + εI)⁻¹ where ε = scale * trace(Λ)/d."""


def safe_inverse(
    matrix: Tensor,
    ridge: float = 1e-4,
    min_eigenvalue: float = 1e-6,
) -> Tensor:
    """
    Compute numerically stable inverse of a matrix.

    Uses adaptive eigenvalue regularization when the matrix is near-singular.
    Monitors eigenvalues and increases ridge if needed to ensure stability.

    Args:
        matrix: (d, d) symmetric matrix to invert
        ridge: Base ridge regularization coefficient
        min_eigenvalue: Minimum acceptable eigenvalue threshold

    Returns:
        (d, d) inverse matrix
    """
    d = matrix.shape[0]
    eye = torch.eye(d, dtype=matrix.dtype, device=matrix.device)

    # Check minimum eigenvalue
    try:
        eigvals = torch.linalg.eigvalsh(matrix)
        min_eig = eigvals.min().item()

        # Adaptive regularization: if min eigenvalue is too small, increase ridge
        if min_eig < min_eigenvalue:
            effective_ridge = max(ridge, min_eigenvalue - min_eig + 1e-6)
        else:
            effective_ridge = ridge
    except RuntimeError:
        # If eigenvalue computation fails, use default ridge
        effective_ridge = ridge

    # Add regularization
    regularized = matrix + effective_ridge * eye

    # Use pseudo-inverse for numerical stability
    try:
        # Try direct inverse first
        inv = torch.linalg.inv(regularized)
    except RuntimeError:
        # Fall back to pseudo-inverse
        inv = torch.linalg.pinv(regularized)

    return inv


def batch_inverse(
    matrices: Tensor,
    strategy: RegularizationStrategy = RegularizationStrategy.TIKHONOV,
    ridge: float = 1e-4,
    min_eigenvalue: float = 1e-6,
    max_condition: float = 100.0,
    tikhonov_scale: float = 0.01,
) -> Tensor:
    """
    Compute inverse of a batch of matrices with configurable regularization.

    Supports three regularization strategies:
    - ABSOLUTE: Legacy behavior with fixed eigenvalue threshold
    - RELATIVE: Bound condition number for scale-invariant regularization
    - TIKHONOV: (Λ + εI)⁻¹ with scale-aware ε = scale * trace(Λ)/d

    Args:
        matrices: (n, d, d) batch of symmetric matrices
        strategy: Regularization strategy (default: TIKHONOV)
        ridge: Base ridge for ABSOLUTE strategy
        min_eigenvalue: Minimum eigenvalue threshold for ABSOLUTE strategy
        max_condition: Maximum condition number for RELATIVE strategy
        tikhonov_scale: ε scale factor for TIKHONOV strategy

    Returns:
        (n, d, d) batch of inverse matrices
    """
    n, d, _ = matrices.shape
    eye = torch.eye(d, dtype=matrices.dtype, device=matrices.device)
    inv = torch.zeros_like(matrices)

    for i in range(n):
        mat = matrices[i]

        if strategy == RegularizationStrategy.TIKHONOV:
            # Scale-aware Tikhonov: ε = scale * trace(Λ)/d
            trace = torch.trace(mat).item()
            epsilon = tikhonov_scale * trace / d
            # Enforce minimum epsilon for numerical stability
            epsilon = max(epsilon, 1e-10)
            regularized = mat + epsilon * eye

        elif strategy == RegularizationStrategy.RELATIVE:
            # Bound condition number by clamping min eigenvalue
            try:
                eigvals, eigvecs = torch.linalg.eigh(mat)
                max_eig = eigvals[-1].item()
                min_allowed = max_eig / max_condition
                min_allowed = max(min_allowed, 1e-10)
                eigvals_clamped = torch.clamp(eigvals, min=min_allowed)
                regularized = eigvecs @ torch.diag(eigvals_clamped) @ eigvecs.T
            except RuntimeError:
                # Fallback to ridge if eigendecomposition fails
                regularized = mat + ridge * eye

        else:  # ABSOLUTE (legacy)
            try:
                eigvals = torch.linalg.eigvalsh(mat)
                min_eig = eigvals.min().item()
                if min_eig < min_eigenvalue:
                    effective_ridge = max(ridge, min_eigenvalue - min_eig + 1e-6)
                else:
                    effective_ridge = ridge
            except RuntimeError:
                effective_ridge = ridge
            regularized = mat + effective_ridge * eye

        # Compute inverse
        try:
            inv[i] = torch.linalg.inv(regularized)
        except RuntimeError:
            inv[i] = torch.linalg.pinv(regularized)

    return inv


def batch_inverse_legacy(
    matrices: Tensor,
    ridge: float = 1e-4,
    min_eigenvalue: float = 1e-6,
) -> Tensor:
    """
    Legacy batch_inverse for backward compatibility.

    Uses ABSOLUTE strategy with original parameters.
    """
    return batch_inverse(
        matrices,
        strategy=RegularizationStrategy.ABSOLUTE,
        ridge=ridge,
        min_eigenvalue=min_eigenvalue,
    )


def condition_number(matrix: Tensor) -> float:
    """
    Compute condition number of a matrix.

    Args:
        matrix: (d, d) matrix

    Returns:
        Condition number (ratio of largest to smallest singular value)
    """
    s = torch.linalg.svdvals(matrix)
    if s.min() < 1e-10:
        return float('inf')
    return (s.max() / s.min()).item()
