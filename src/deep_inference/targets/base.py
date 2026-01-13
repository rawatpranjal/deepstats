"""
Base protocol and classes for target functionals.

A Target defines what quantity we want inference on.
"""

from typing import Protocol, Optional, Callable, runtime_checkable
from dataclasses import dataclass
import torch
from torch import Tensor


@runtime_checkable
class Target(Protocol):
    """
    Protocol for target functionals.

    A target defines:
    - h(x, theta, t_tilde): The target value H(x, θ; t̃)
    - jacobian: ∂H/∂θ (optional, falls back to autodiff)

    Common targets:
    - AverageParameter: H = θ_k (e.g., treatment effect β)
    - AverageMarginalEffect: H = ∂G/∂t * β (e.g., AME in logit)
    - Elasticity: H = (1-G) * β * t (price elasticity)
    """

    output_dim: int
    """Dimension of target output (1 for scalar targets)."""

    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """
        Compute target value (single observation).

        Args:
            x: Covariates (d_x,)
            theta: Parameters (d_theta,)
            t_tilde: Evaluation point (scalar or d_t)

        Returns:
            Scalar or (output_dim,) target value
        """
        ...

    def jacobian(
        self, x: Tensor, theta: Tensor, t_tilde: Tensor
    ) -> Optional[Tensor]:
        """
        Compute Jacobian of target w.r.t. theta (optional).

        If not implemented (returns None), autodiff will be used.

        Args:
            x: Covariates (d_x,)
            theta: Parameters (d_theta,)
            t_tilde: Evaluation point (scalar or d_t)

        Returns:
            (d_theta,) if output_dim=1 (gradient)
            (output_dim, d_theta) if output_dim>1 (Jacobian)
            None to use autodiff
        """
        ...


@dataclass
class TargetMetadata:
    """Metadata about a target functional."""

    output_dim: int
    name: str = "custom"
    is_linear_in_theta: bool = False  # If H is linear in θ, jacobian is constant


class BaseTarget:
    """
    Base class for target functionals.

    Provides default implementations and autodiff fallbacks.
    """

    output_dim: int = 1  # Most targets are scalar

    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement h()")

    def jacobian(
        self, x: Tensor, theta: Tensor, t_tilde: Tensor
    ) -> Optional[Tensor]:
        """Default: use autodiff (return None)."""
        return None

    def get_metadata(self) -> TargetMetadata:
        """Return metadata about this target."""
        return TargetMetadata(
            output_dim=self.output_dim,
            name=self.__class__.__name__,
        )


class CustomTarget(BaseTarget):
    """
    Wrapper for user-provided target functions.

    Enables users to pass arbitrary target functions and get inference.
    """

    def __init__(
        self,
        h_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
        output_dim: int = 1,
        jacobian_fn: Optional[
            Callable[[Tensor, Tensor, Tensor], Tensor]
        ] = None,
    ):
        """
        Create a custom target from a function.

        Args:
            h_fn: Target function (x, theta, t_tilde) -> scalar or (output_dim,)
            output_dim: Dimension of output
            jacobian_fn: Optional closed-form Jacobian function
        """
        self._h_fn = h_fn
        self.output_dim = output_dim
        self._jacobian_fn = jacobian_fn

    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        return self._h_fn(x, theta, t_tilde)

    def jacobian(
        self, x: Tensor, theta: Tensor, t_tilde: Tensor
    ) -> Optional[Tensor]:
        if self._jacobian_fn is not None:
            return self._jacobian_fn(x, theta, t_tilde)
        return None


def target_from_fn(
    h_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    output_dim: int = 1,
) -> CustomTarget:
    """
    Create a target from a function.

    Args:
        h_fn: Target function (x, theta, t_tilde) -> value
        output_dim: Dimension of output

    Returns:
        CustomTarget instance
    """
    return CustomTarget(h_fn=h_fn, output_dim=output_dim)
