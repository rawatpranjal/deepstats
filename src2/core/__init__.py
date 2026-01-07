"""Core DML algorithm components."""

from .autodiff import (
    compute_gradient,
    compute_hessian,
    compute_target_gradient,
    detect_theta_dependence,
)
from .lambda_estimator import LambdaEstimator
from .algorithm import structural_dml_core, DMLResult, compute_coverage, compute_se_ratio

__all__ = [
    "compute_gradient",
    "compute_hessian",
    "compute_target_gradient",
    "detect_theta_dependence",
    "LambdaEstimator",
    "structural_dml_core",
    "DMLResult",
    "compute_coverage",
    "compute_se_ratio",
]
