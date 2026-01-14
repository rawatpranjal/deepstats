"""
Base protocol and types for Lambda strategies.
"""

from typing import Protocol, Optional, runtime_checkable, Literal
from enum import Enum, auto
from dataclasses import dataclass, field
import torch
from torch import Tensor


class Regime(Enum):
    """
    The three Lambda estimation regimes.

    Regime determines:
    - How Λ(x) is obtained (compute vs estimate)
    - Cross-fitting strategy (2-way vs 3-way split)
    """

    A = auto()  # Randomized + Hessian doesn't depend on Y → COMPUTE
    B = auto()  # Linear model → ANALYTIC (Λ = E[TT'|X])
    C = auto()  # Observational + Nonlinear → ESTIMATE (3-way split)


@dataclass
class RegimeInfo:
    """Information about a regime."""

    regime: Regime
    requires_theta: bool  # Does Λ depend on θ̂?
    requires_separate_fold: bool  # Need 3-way split?
    lambda_method: str  # "compute", "analytic", "estimate"


REGIME_INFO = {
    Regime.A: RegimeInfo(
        regime=Regime.A,
        requires_theta=True,  # Need θ̂ to compute Λ
        requires_separate_fold=False,  # 2-way split sufficient
        lambda_method="compute",
    ),
    Regime.B: RegimeInfo(
        regime=Regime.B,
        requires_theta=False,  # Λ = E[TT'|X] doesn't need θ!
        requires_separate_fold=False,  # 2-way split
        lambda_method="analytic",
    ),
    Regime.C: RegimeInfo(
        regime=Regime.C,
        requires_theta=True,  # Need θ̂ to compute Hessians
        requires_separate_fold=True,  # 3-way split required!
        lambda_method="estimate",
    ),
}


@dataclass
class RegularizationConfig:
    """
    Configuration for Lambda regularization.

    Controls how Λ matrices are regularized for numerical stability:
    1. Eigenvalue floor: How to handle small/negative eigenvalues
    2. Shrinkage: Bias-variance tradeoff via Ledoit-Wolf style shrinkage
    3. Inversion: How to compute Λ⁻¹ safely

    Default settings use relative eigenvalue floor + Tikhonov inversion,
    which provides scale-invariant regularization.
    """

    # Eigenvalue floor strategy
    eigenvalue_strategy: Literal["absolute", "relative"] = "relative"
    """'absolute': floor at fixed value; 'relative': floor at max_eig/max_condition"""

    absolute_floor: float = 1e-4
    """Minimum eigenvalue when using absolute strategy."""

    max_condition: float = 100.0
    """Maximum condition number when using relative strategy."""

    # Shrinkage (Ledoit-Wolf style)
    apply_shrinkage: bool = False
    """Whether to apply shrinkage toward a well-conditioned target."""

    shrinkage_intensity: float = 0.1
    """Shrinkage intensity α: Λ_shrunk = (1-α)Λ + α·target"""

    shrinkage_target: Literal["scaled_identity", "diagonal"] = "scaled_identity"
    """Shrinkage target: 'scaled_identity' = (trace/d)·I, 'diagonal' = diag(Λ)"""

    # Inversion strategy
    inversion_strategy: Literal["direct", "tikhonov"] = "tikhonov"
    """'direct': Λ⁻¹ with eigenvalue clamping; 'tikhonov': (Λ + εI)⁻¹"""

    tikhonov_scale: float = 0.01
    """ε = tikhonov_scale · trace(Λ)/d for scale-aware regularization."""


# Default config for backward compatibility
DEFAULT_REGULARIZATION_CONFIG = RegularizationConfig()


@runtime_checkable
class LambdaStrategy(Protocol):
    """
    Protocol for Lambda estimation strategies.

    A Lambda strategy provides Λ(x) = E[ℓ_θθ | X=x].
    """

    requires_theta: bool
    """True if Λ depends on θ̂ estimates."""

    requires_separate_fold: bool
    """True if 3-way split is needed (Regime C)."""

    def fit(
        self,
        X: Tensor,
        T: Tensor,
        Y: Tensor,
        theta_hat: Optional[Tensor],
        model: "StructuralModel",
    ) -> None:
        """
        Fit the Lambda estimator/computer.

        Args:
            X: (n, d_x) covariates
            T: (n,) or (n, d_t) treatments
            Y: (n,) outcomes
            theta_hat: (n, d_theta) estimated parameters (if requires_theta)
            model: The structural model (for computing Hessians)
        """
        ...

    def predict(self, X: Tensor, theta_hat: Optional[Tensor] = None) -> Tensor:
        """
        Predict Λ(x) for new observations.

        Args:
            X: (n, d_x) covariates
            theta_hat: (n, d_theta) theta estimates (if requires_theta)

        Returns:
            (n, d_theta, d_theta) Lambda matrices
        """
        ...


class BaseLambdaStrategy:
    """Base class for Lambda strategies."""

    requires_theta: bool = True
    requires_separate_fold: bool = False

    def fit(
        self,
        X: Tensor,
        T: Tensor,
        Y: Tensor,
        theta_hat: Optional[Tensor],
        model: "StructuralModel",
    ) -> None:
        """Must be implemented by subclasses."""
        raise NotImplementedError()

    def predict(self, X: Tensor, theta_hat: Optional[Tensor] = None) -> Tensor:
        """Must be implemented by subclasses."""
        raise NotImplementedError()
