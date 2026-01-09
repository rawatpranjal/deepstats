"""Base family class and protocol definition."""

import torch
from torch import Tensor
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class Family(Protocol):
    """Protocol that all families must implement."""

    theta_dim: int

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Compute per-observation loss.

        Args:
            y: (n,) outcomes
            t: (n,) treatments
            theta: (n, theta_dim) parameters

        Returns:
            (n,) per-observation losses
        """
        ...

    def hessian_depends_on_theta(self) -> bool:
        """
        Whether the Hessian depends on theta.

        If True, three-way splitting is required.
        """
        ...


class BaseFamily:
    """
    Base implementation with autodiff fallbacks.

    Subclasses can override gradient() and hessian() for closed-form
    implementations that are more numerically stable.
    """

    theta_dim: int = 2

    def __init__(self, **kwargs):
        """Base constructor - accepts and ignores kwargs for compatibility."""
        # Subclasses can override to use kwargs like target='ame'
        pass

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Compute per-observation loss.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement loss()")

    def gradient(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """
        Optionally provide closed-form gradient ℓ_θ.

        Returns None to use autodiff.

        Args:
            y: (n,) outcomes
            t: (n,) treatments
            theta: (n, theta_dim) parameters

        Returns:
            (n, theta_dim) gradient tensor, or None to use autodiff
        """
        return None

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """
        Optionally provide closed-form Hessian ℓ_θθ.

        Returns None to use autodiff.

        Args:
            y: (n,) outcomes
            t: (n,) treatments
            theta: (n, theta_dim) parameters

        Returns:
            (n, theta_dim, theta_dim) Hessian tensor, or None to use autodiff
        """
        return None

    def hessian_depends_on_theta(self) -> bool:
        """
        Whether the Hessian depends on theta.

        Conservative default is True (use three-way splitting).
        Override to False for linear-in-parameters models.
        """
        return True

    def default_target(self, x: Tensor, theta: Tensor) -> Tensor:
        """
        Default target: E[β(X)] = mean of second parameter.

        Args:
            x: (n, d_x) covariates (unused in default)
            theta: (n, theta_dim) parameters

        Returns:
            Scalar target value
        """
        return theta[:, 1].mean()

    def per_obs_target(self, theta: Tensor, t: Tensor) -> Tensor:
        """
        Per-observation target h(θ, t) = β.

        For H = E[β(X)] = (1/n)Σβᵢ, this returns βᵢ.
        The t parameter enables T-dependent targets like AME.

        Args:
            theta: (n, theta_dim) parameters
            t: (n,) treatments

        Returns:
            (n,) per-observation target values
        """
        return theta[:, 1]

    def per_obs_target_gradient(self, theta: Tensor, t: Tensor) -> Tensor:
        """
        Gradient of per-observation target: ∂h/∂θ = (0, 1).

        For H = E[β(X)], the gradient with respect to (α, β) is (0, 1).
        The t parameter enables T-dependent targets like AME.

        Args:
            theta: (n, theta_dim) parameters
            t: (n,) treatments

        Returns:
            (n, theta_dim) or (theta_dim,) gradient
        """
        n = theta.shape[0]
        grad = torch.zeros(n, self.theta_dim, dtype=theta.dtype, device=theta.device)
        grad[:, 1] = 1.0  # ∂β/∂β = 1, ∂β/∂α = 0
        return grad
