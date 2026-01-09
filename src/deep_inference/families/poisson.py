"""Poisson family implementation with closed-form optimizations."""

import torch
from torch import Tensor

from .base import BaseFamily


class PoissonFamily(BaseFamily):
    """
    Poisson structural model.

    Model: Y ~ Poisson(lambda), lambda = exp(alpha(X) + beta(X) * T)

    Loss: Poisson negative log-likelihood (deviance)
        L = lambda - Y * log(lambda)

    Parameters:
        theta = (alpha, beta) where
        - alpha(x): baseline log-rate
        - beta(x): treatment effect on log-rate

    Target: E[beta(X)] (average treatment effect on log-rate)

    Note: The Hessian depends on theta through lambda = exp(alpha + beta*T),
    so three-way splitting is required.
    """

    theta_dim = 2

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Poisson negative log-likelihood.

        L = lambda - Y * log(lambda)

        Args:
            y: (n,) count outcomes (non-negative integers)
            t: (n,) treatments
            theta: (n, 2) parameters [alpha, beta]

        Returns:
            (n,) per-observation losses
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]

        # Clamp linear predictor for numerical stability
        eta = torch.clamp(alpha + beta * t, -20, 20)
        lam = torch.exp(eta)

        # NLL = lambda - y * log(lambda)
        # Add small epsilon to avoid log(0)
        return lam - y * torch.log(lam + 1e-10)

    def gradient(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Closed-form gradient.

        l_theta = (lambda - Y) * (1, t)'

        where lambda = exp(alpha + beta*t)

        Args:
            y: (n,) outcomes
            t: (n,) treatments
            theta: (n, 2) parameters [alpha, beta]

        Returns:
            (n, 2) gradient tensor
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]

        eta = torch.clamp(alpha + beta * t, -20, 20)
        lam = torch.exp(eta)

        # Gradient: d/d_alpha = (lambda - y), d/d_beta = (lambda - y) * t
        residual = lam - y

        grad_alpha = residual
        grad_beta = residual * t

        return torch.stack([grad_alpha, grad_beta], dim=1)

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Closed-form Hessian.

        l_theta_theta = lambda * [[1, t], [t, t^2]]

        where lambda = exp(alpha + beta*t)

        Note: The Hessian DEPENDS on theta through lambda.
        This means three-way splitting is required.

        Args:
            y: (n,) outcomes (unused in Hessian)
            t: (n,) treatments
            theta: (n, 2) parameters [alpha, beta]

        Returns:
            (n, 2, 2) Hessian tensor
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]

        eta = torch.clamp(alpha + beta * t, -20, 20)
        lam = torch.exp(eta)

        n = len(y)
        H = torch.zeros(n, 2, 2, dtype=theta.dtype, device=theta.device)
        H[:, 0, 0] = lam
        H[:, 0, 1] = lam * t
        H[:, 1, 0] = lam * t
        H[:, 1, 1] = lam * t ** 2

        return H

    def hessian_depends_on_theta(self) -> bool:
        """
        Poisson Hessian depends on theta through lambda = exp(alpha + beta*t).

        This means three-way splitting is required.
        """
        return True

    def residual(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Compute residual: y - lambda.

        Args:
            y: (n,) outcomes
            t: (n,) treatments
            theta: (n, 2) parameters [alpha, beta]

        Returns:
            (n,) residuals
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]

        eta = torch.clamp(alpha + beta * t, -20, 20)
        lam = torch.exp(eta)

        return y - lam

    def predicted_rate(self, t: Tensor, theta: Tensor) -> Tensor:
        """
        Compute predicted rate lambda = exp(alpha + beta*t).

        Args:
            t: (n,) treatments
            theta: (n, 2) parameters [alpha, beta]

        Returns:
            (n,) predicted rates
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]

        eta = torch.clamp(alpha + beta * t, -20, 20)
        return torch.exp(eta)
