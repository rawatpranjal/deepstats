"""
Base protocol and types for Lambda strategies.
"""

from typing import Protocol, Optional, runtime_checkable
from enum import Enum, auto
from dataclasses import dataclass
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
