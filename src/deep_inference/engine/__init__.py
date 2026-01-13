"""
Core inference engine.

The engine orchestrates:
- Cross-fitting (2-way or 3-way)
- Influence function assembly
- Variance estimation
"""

from .crossfit import CrossFitter, run_crossfit
from .assembler import assemble_influence, compute_psi
from .variance import estimate_variance, compute_se

__all__ = [
    "CrossFitter",
    "run_crossfit",
    "assemble_influence",
    "compute_psi",
    "estimate_variance",
    "compute_se",
]
