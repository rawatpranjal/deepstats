"""
Linear regression model for structural estimation.

Y = α(X) + β(X) * T + ε

Loss: (y - α - β*t)² / 2
Score: (α + β*t - y) * [1, t]
Hessian: [[1, t], [t, t²]] (DOES NOT depend on θ!)

This is Regime B: Λ = E[TT'|X] doesn't depend on θ.
"""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseModel


class LinearModel(BaseModel):
    """
    Linear regression model.

    Model: E[Y|T,X] = α(X) + β(X) * T

    Key property: Hessian doesn't depend on theta (Regime B).
    This means Λ(x) = E[TT'|X] can be estimated independently of θ̂.
    """

    theta_dim: int = 2
    hessian_depends_on_theta: bool = False  # KEY: Enables Regime B
    hessian_depends_on_y: bool = False

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Squared error loss (single observation).

        Args:
            y: Outcome (scalar)
            t: Treatment (scalar)
            theta: Parameters (2,) = [α, β]

        Returns:
            Scalar loss: (y - α - β*t)² / 2
        """
        pred = theta[0] + theta[1] * t
        return 0.5 * (y - pred) ** 2

    def score(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Closed-form score (gradient of loss).

        ∂ℓ/∂θ = (pred - y) * [1, t]

        Args:
            y: Outcome (scalar)
            t: Treatment (scalar)
            theta: Parameters (2,)

        Returns:
            (2,) gradient vector
        """
        pred = theta[0] + theta[1] * t
        residual = pred - y
        return torch.stack([residual, residual * t])

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Closed-form Hessian (DOES NOT depend on theta or y!).

        ∂²ℓ/∂θ² = [[1, t], [t, t²]]

        Args:
            y: Outcome (scalar) - NOT USED
            t: Treatment (scalar)
            theta: Parameters (2,) - NOT USED

        Returns:
            (2, 2) Hessian matrix
        """
        # Hessian is constant in theta and y
        H = torch.zeros(2, 2, dtype=theta.dtype, device=theta.device)
        H[0, 0] = 1.0
        H[0, 1] = t
        H[1, 0] = t
        H[1, 1] = t * t
        return H


# Convenience alias
Linear = LinearModel
