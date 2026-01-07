"""Linear algebra utilities with numerical stability."""

import torch
from torch import Tensor
from typing import Optional


def safe_inverse(
    matrix: Tensor,
    ridge: float = 1e-4,
    min_eigenvalue: float = 1e-6,
) -> Tensor:
    """
    Compute numerically stable inverse of a matrix.

    Uses eigenvalue regularization when the matrix is near-singular.

    Args:
        matrix: (d, d) symmetric matrix to invert
        ridge: Ridge regularization coefficient
        min_eigenvalue: Minimum eigenvalue threshold

    Returns:
        (d, d) inverse matrix
    """
    d = matrix.shape[0]

    # Add ridge regularization
    regularized = matrix + ridge * torch.eye(d, dtype=matrix.dtype, device=matrix.device)

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
    ridge: float = 1e-4,
) -> Tensor:
    """
    Compute inverse of a batch of matrices.

    Args:
        matrices: (n, d, d) batch of symmetric matrices
        ridge: Ridge regularization coefficient

    Returns:
        (n, d, d) batch of inverse matrices
    """
    n, d, _ = matrices.shape

    # Add ridge regularization
    eye = torch.eye(d, dtype=matrices.dtype, device=matrices.device)
    regularized = matrices + ridge * eye.unsqueeze(0)

    # Batch pseudo-inverse
    try:
        inv = torch.linalg.inv(regularized)
    except RuntimeError:
        # Fall back to per-matrix pseudo-inverse
        inv = torch.zeros_like(matrices)
        for i in range(n):
            inv[i] = torch.linalg.pinv(regularized[i])

    return inv


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


def batch_condition_numbers(matrices: Tensor) -> Tensor:
    """
    Compute condition numbers for a batch of matrices.

    Args:
        matrices: (n, d, d) batch of matrices

    Returns:
        (n,) condition numbers
    """
    n = matrices.shape[0]
    conds = torch.zeros(n, dtype=matrices.dtype, device=matrices.device)

    for i in range(n):
        conds[i] = condition_number(matrices[i])

    return conds
