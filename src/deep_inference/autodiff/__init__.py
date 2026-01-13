"""
Automatic differentiation utilities using torch.func for batched computation.

This module provides vmap-based batched derivatives:
- score: ℓ_θ = ∂ℓ/∂θ (gradient of loss)
- hessian: ℓ_θθ = ∂²ℓ/∂θ² (Hessian of loss)
- jacobian: H_θ = ∂H/∂θ (Jacobian of target)

All functions are batched via vmap for efficiency with large datasets.
"""

from .score import compute_score, compute_score_vmap
from .hessian import compute_hessian, compute_hessian_vmap
from .jacobian import compute_target_jacobian, compute_target_jacobian_vmap

__all__ = [
    "compute_score",
    "compute_score_vmap",
    "compute_hessian",
    "compute_hessian_vmap",
    "compute_target_jacobian",
    "compute_target_jacobian_vmap",
]
