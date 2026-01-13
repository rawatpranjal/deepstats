"""
Regime detection and Lambda strategy selection.

This is the "brain" that analyzes model properties and data settings
to determine the optimal Lambda estimation approach.
"""

from typing import Optional, TYPE_CHECKING
import torch
from torch import Tensor

from .base import Regime, LambdaStrategy, REGIME_INFO

if TYPE_CHECKING:
    from deep_inference.models import StructuralModel


def detect_regime(
    model: "StructuralModel",
    is_randomized: bool,
    has_known_treatment_dist: bool = False,
) -> Regime:
    """
    Detect the appropriate Lambda regime.

    Decision tree:
    1. If hessian_depends_on_theta = False → Regime B (analytic)
    2. If randomized + has F_T + hessian_depends_on_y = False → Regime A (compute)
    3. Else → Regime C (estimate with 3-way split)

    Args:
        model: The structural model
        is_randomized: True if treatment is randomly assigned
        has_known_treatment_dist: True if F_T is known (for Regime A)

    Returns:
        Regime enum value
    """
    # Check model properties
    hess_depends_on_theta = model.hessian_depends_on_theta
    hess_depends_on_y = model.hessian_depends_on_y

    # Regime B: Linear model (Hessian doesn't depend on theta)
    if not hess_depends_on_theta:
        return Regime.B

    # Regime A: Randomized + can compute Λ
    if is_randomized and has_known_treatment_dist and not hess_depends_on_y:
        return Regime.A

    # Regime C: Everything else (must estimate Λ with 3-way split)
    return Regime.C


def select_lambda_strategy(
    model: "StructuralModel",
    is_randomized: bool = False,
    treatment_dist: Optional["TreatmentDistribution"] = None,
    lambda_method: Optional[str] = None,
    **kwargs,
) -> LambdaStrategy:
    """
    Select the appropriate Lambda strategy.

    Args:
        model: The structural model
        is_randomized: True if treatment is randomly assigned
        treatment_dist: Treatment distribution (for Regime A)
        lambda_method: Override auto-detection (for advanced users)
        **kwargs: Additional arguments for the strategy

    Returns:
        LambdaStrategy instance
    """
    from .compute import ComputeLambda
    from .analytic import AnalyticLambda
    from .estimate import EstimateLambda

    # Auto-detect regime
    has_known_dist = treatment_dist is not None
    regime = detect_regime(model, is_randomized, has_known_dist)

    # Allow manual override
    if lambda_method is not None:
        if lambda_method == "compute":
            regime = Regime.A
        elif lambda_method == "analytic":
            regime = Regime.B
        elif lambda_method == "estimate":
            regime = Regime.C
        elif lambda_method == "aggregate":
            # Special case: simple aggregate (like current implementation)
            return EstimateLambda(method="aggregate", **kwargs)
        elif lambda_method == "mlp":
            return EstimateLambda(method="mlp", **kwargs)
        elif lambda_method == "rf":
            return EstimateLambda(method="rf", **kwargs)
        elif lambda_method == "ridge":
            return EstimateLambda(method="ridge", **kwargs)

    # Create strategy based on regime
    if regime == Regime.A:
        return ComputeLambda(treatment_dist=treatment_dist, model=model, **kwargs)
    elif regime == Regime.B:
        return AnalyticLambda(**kwargs)
    else:  # Regime C
        return EstimateLambda(**kwargs)


def get_regime_info(regime: Regime):
    """Get information about a regime."""
    return REGIME_INFO[regime]


def needs_three_way_split(regime: Regime) -> bool:
    """Check if regime requires 3-way cross-fitting."""
    return REGIME_INFO[regime].requires_separate_fold


def describe_regime(regime: Regime) -> str:
    """Get human-readable description of regime."""
    descriptions = {
        Regime.A: (
            "Regime A: Randomized experiment with known F_T. "
            "Lambda computed via Monte Carlo integration. 2-way split."
        ),
        Regime.B: (
            "Regime B: Linear model (Hessian doesn't depend on theta). "
            "Lambda = E[TT'|X] estimated independently of theta. 2-way split."
        ),
        Regime.C: (
            "Regime C: Observational data or nonlinear model. "
            "Lambda estimated via neural network regression. 3-way split required."
        ),
    }
    return descriptions[regime]
