"""Linear family implementation with closed-form optimizations."""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseFamily


class LinearFamily(BaseFamily):
    """
    Linear structural model.

    Model: E[Y | T, X] = alpha(X) + beta(X) * T

    Loss: (y - alpha - beta * t)^2

    Parameters:
        theta = (alpha, beta) where
        - alpha(x): baseline outcome
        - beta(x): treatment effect

    Target: E[beta(X)] (average treatment effect)
    """

    theta_dim = 2

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Squared error loss.

        Args:
            y: (n,) outcomes
            t: (n,) treatments
            theta: (n, 2) parameters [alpha, beta]

        Returns:
            (n,) per-observation losses
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]
        mu = alpha + beta * t
        return (y - mu) ** 2

    def gradient(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Closed-form gradient.

        l_theta = d/d(theta) (y - alpha - beta*t)^2
                = -2(y - alpha - beta*t) * (1, t)'

        Args:
            y: (n,) outcomes
            t: (n,) treatments
            theta: (n, 2) parameters [alpha, beta]

        Returns:
            (n, 2) gradient tensor
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]
        residual = y - alpha - beta * t

        grad_alpha = -2 * residual
        grad_beta = -2 * residual * t

        return torch.stack([grad_alpha, grad_beta], dim=1)

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Closed-form Hessian.

        l_theta_theta = d^2/d(theta)^2 (y - alpha - beta*t)^2
                      = 2 * [[1, t], [t, t^2]]

        Note: The Hessian does NOT depend on theta (only on t).
        This means two-way splitting is sufficient.

        Args:
            y: (n,) outcomes
            t: (n,) treatments
            theta: (n, 2) parameters [alpha, beta] (unused)

        Returns:
            (n, 2, 2) Hessian tensor
        """
        n = len(y)
        ones = torch.ones(n, dtype=y.dtype, device=y.device)

        H = torch.zeros(n, 2, 2, dtype=y.dtype, device=y.device)
        H[:, 0, 0] = 2 * ones
        H[:, 0, 1] = 2 * t
        H[:, 1, 0] = 2 * t
        H[:, 1, 1] = 2 * t ** 2

        return H

    def hessian_depends_on_theta(self) -> bool:
        """
        Linear Hessian does not depend on theta.

        This means two-way splitting is sufficient.
        """
        return False

    def residual(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Compute residual: y - mu.

        Args:
            y: (n,) outcomes
            t: (n,) treatments
            theta: (n, 2) parameters [alpha, beta]

        Returns:
            (n,) residuals
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]
        mu = alpha + beta * t
        return y - mu
