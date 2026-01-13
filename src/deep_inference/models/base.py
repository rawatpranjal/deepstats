"""
Base protocol and classes for structural models.

A StructuralModel defines the loss function and its properties for inference.
"""

from typing import Protocol, Optional, Callable, runtime_checkable
from dataclasses import dataclass
import torch
from torch import Tensor


@runtime_checkable
class StructuralModel(Protocol):
    """
    Protocol for structural models.

    A structural model defines:
    - loss(y, t, theta): The loss function
    - Metadata about how Hessians depend on parameters

    Implementations can optionally override score() and hessian()
    for closed-form computation (falls back to autodiff if not provided).
    """

    theta_dim: int
    """Dimension of parameter vector theta."""

    hessian_depends_on_theta: bool
    """
    True if ℓ_θθ varies with θ values.

    Examples:
    - Linear: False (Hessian = TT' doesn't depend on θ)
    - Logit: True (Hessian = σ(1-σ)TT' depends on θ through σ)

    Affects:
    - If False → Regime B (analytic Λ) or simpler estimation
    - If True → May need Regime A (compute) or C (estimate)
    """

    hessian_depends_on_y: bool
    """
    True if ℓ_θθ varies with Y values.

    Examples:
    - Linear: False
    - Logit: False
    - Most GLMs: False (Hessian is Fisher information, doesn't use Y)

    Affects:
    - If False + randomized → Can compute Λ via Monte Carlo (Regime A)
    - If True → Must estimate Λ even under randomization
    """

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Compute per-observation loss.

        Args:
            y: Single observation outcome (scalar or vector)
            t: Single observation treatment (scalar or vector)
            theta: Single observation parameters (d_theta,)

        Returns:
            Scalar loss value for this observation
        """
        ...

    def score(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """
        Compute gradient of loss w.r.t. theta (optional).

        If not implemented (returns None), autodiff will be used.

        Args:
            y: Outcome (scalar)
            t: Treatment (scalar or vector)
            theta: Parameters (d_theta,)

        Returns:
            (d_theta,) gradient tensor, or None to use autodiff
        """
        ...

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """
        Compute Hessian of loss w.r.t. theta (optional).

        If not implemented (returns None), autodiff will be used.

        Args:
            y: Outcome (scalar)
            t: Treatment (scalar or vector)
            theta: Parameters (d_theta,)

        Returns:
            (d_theta, d_theta) Hessian tensor, or None to use autodiff
        """
        ...


@dataclass
class ModelMetadata:
    """Metadata about a structural model's properties."""

    theta_dim: int
    hessian_depends_on_theta: bool
    hessian_depends_on_y: bool
    name: str = "custom"


class BaseModel:
    """
    Base class for structural models.

    Provides default implementations and autodiff fallbacks.
    """

    theta_dim: int = 2
    hessian_depends_on_theta: bool = True  # Conservative default
    hessian_depends_on_y: bool = False  # True for most GLMs

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement loss()")

    def score(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """Default: use autodiff (return None)."""
        return None

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """Default: use autodiff (return None)."""
        return None

    def get_metadata(self) -> ModelMetadata:
        """Return metadata about this model."""
        return ModelMetadata(
            theta_dim=self.theta_dim,
            hessian_depends_on_theta=self.hessian_depends_on_theta,
            hessian_depends_on_y=self.hessian_depends_on_y,
            name=self.__class__.__name__,
        )


class CustomModel(BaseModel):
    """
    Wrapper for user-provided loss functions.

    Enables users to pass arbitrary loss functions and get inference.
    """

    def __init__(
        self,
        loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
        theta_dim: int,
        hessian_depends_on_theta: bool = True,
        hessian_depends_on_y: bool = False,
        score_fn: Optional[Callable[[Tensor, Tensor, Tensor], Tensor]] = None,
        hessian_fn: Optional[Callable[[Tensor, Tensor, Tensor], Tensor]] = None,
    ):
        """
        Create a custom model from a loss function.

        Args:
            loss_fn: Loss function (y, t, theta) -> scalar
            theta_dim: Dimension of theta vector
            hessian_depends_on_theta: True if Hessian varies with theta
            hessian_depends_on_y: True if Hessian varies with Y
            score_fn: Optional closed-form score function
            hessian_fn: Optional closed-form Hessian function
        """
        self._loss_fn = loss_fn
        self.theta_dim = theta_dim
        self.hessian_depends_on_theta = hessian_depends_on_theta
        self.hessian_depends_on_y = hessian_depends_on_y
        self._score_fn = score_fn
        self._hessian_fn = hessian_fn

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        return self._loss_fn(y, t, theta)

    def score(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        if self._score_fn is not None:
            return self._score_fn(y, t, theta)
        return None

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        if self._hessian_fn is not None:
            return self._hessian_fn(y, t, theta)
        return None


def model_from_loss(
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    theta_dim: int,
    hessian_depends_on_theta: Optional[bool] = None,
    hessian_depends_on_y: Optional[bool] = None,
) -> CustomModel:
    """
    Create a model from a loss function with automatic property detection.

    Args:
        loss_fn: Loss function (y, t, theta) -> scalar
        theta_dim: Dimension of theta vector
        hessian_depends_on_theta: Override automatic detection if provided
        hessian_depends_on_y: Override automatic detection if provided

    Returns:
        CustomModel instance
    """
    from deep_inference.autodiff.hessian import (
        detect_hessian_theta_dependence,
        detect_hessian_y_dependence,
    )

    # Auto-detect if not provided
    if hessian_depends_on_theta is None or hessian_depends_on_y is None:
        # Create sample data for detection
        n_test = 10
        y_sample = torch.randn(n_test)
        t_sample = torch.randn(n_test)

        if hessian_depends_on_theta is None:
            hessian_depends_on_theta = detect_hessian_theta_dependence(
                loss_fn, y_sample, t_sample, theta_dim
            )

        if hessian_depends_on_y is None:
            hessian_depends_on_y = detect_hessian_y_dependence(
                loss_fn, y_sample, t_sample, theta_dim
            )

    return CustomModel(
        loss_fn=loss_fn,
        theta_dim=theta_dim,
        hessian_depends_on_theta=hessian_depends_on_theta,
        hessian_depends_on_y=hessian_depends_on_y,
    )
