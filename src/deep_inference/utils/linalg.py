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
    ridge: float = 1e-4,
    min_eigenvalue: float = 1e-6,
) -> Tensor:
    """
    Compute inverse of a batch of matrices with adaptive regularization.

    Uses per-matrix eigenvalue monitoring to adaptively increase regularization
    for near-singular matrices, preventing extreme corrections in Logit models.

    Args:
        matrices: (n, d, d) batch of symmetric matrices
        ridge: Base ridge regularization coefficient
        min_eigenvalue: Minimum acceptable eigenvalue threshold

    Returns:
        (n, d, d) batch of inverse matrices
    """
    n, d, _ = matrices.shape
    eye = torch.eye(d, dtype=matrices.dtype, device=matrices.device)
    inv = torch.zeros_like(matrices)

    # Process each matrix with adaptive regularization
    for i in range(n):
        mat = matrices[i]

        # Check minimum eigenvalue for this matrix
        try:
            eigvals = torch.linalg.eigvalsh(mat)
            min_eig = eigvals.min().item()

            # Adaptive regularization
            if min_eig < min_eigenvalue:
                effective_ridge = max(ridge, min_eigenvalue - min_eig + 1e-6)
            else:
                effective_ridge = ridge
        except RuntimeError:
            effective_ridge = ridge

        # Add regularization
        regularized = mat + effective_ridge * eye

        # Compute inverse
        try:
            inv[i] = torch.linalg.inv(regularized)
        except RuntimeError:
            inv[i] = torch.linalg.pinv(regularized)

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
