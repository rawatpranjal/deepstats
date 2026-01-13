"""Gaussian family implementation with closed-form optimizations."""

import math
import torch
from torch import Tensor

from .base import BaseFamily


class GaussianFamily(BaseFamily):
    """
    Gaussian structural model.

    Model: Y ~ N(mu, sigma^2) where mu = alpha(X) + beta(X) * T

    Loss: Gaussian NLL = (y - mu)^2 / (2*sigma^2) + log(sigma)

    Parameters:
        theta = (alpha, beta) where
        - alpha(x): baseline outcome
        - beta(x): treatment effect

    Target: E[beta(X)] (average treatment effect)

    Note: sigma is a fixed hyperparameter, not learned.
    """

    theta_dim = 2

    def __init__(self, sigma: float = 1.0, **kwargs):
        """
        Initialize GaussianFamily.

        Args:
            sigma: Standard deviation (fixed, not learned). Default 1.0.
        """
        super().__init__(**kwargs)
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = sigma

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Gaussian negative log-likelihood.

        NLL = (y - mu)^2 / (2*sigma^2) + log(sigma)

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

        return 0.5 * ((y - mu) / self.sigma) ** 2 + math.log(self.sigma)

    def gradient(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Closed-form gradient.

        l_theta = (mu - y) / sigma^2 * (1, t)'

        Args:
            y: (n,) outcomes
            t: (n,) treatments
            theta: (n, 2) parameters [alpha, beta]

        Returns:
            (n, 2) gradient tensor
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]
        mu = alpha + beta * t

        residual = (mu - y) / (self.sigma ** 2)

        grad_alpha = residual
        grad_beta = residual * t

        return torch.stack([grad_alpha, grad_beta], dim=1)

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Closed-form Hessian.

        l_theta_theta = 1/sigma^2 * [[1, t], [t, t^2]]

        Note: The Hessian does NOT depend on theta (only on t and sigma).
        This means two-way splitting is sufficient.

        Args:
            y: (n,) outcomes
            t: (n,) treatments
            theta: (n, 2) parameters [alpha, beta] (unused)

        Returns:
            (n, 2, 2) Hessian tensor
        """
        n = len(y)
        scale = 1.0 / (self.sigma ** 2)

        H = torch.zeros(n, 2, 2, dtype=theta.dtype, device=theta.device)
        H[:, 0, 0] = scale
        H[:, 0, 1] = scale * t
        H[:, 1, 0] = scale * t
        H[:, 1, 1] = scale * t ** 2

        return H

    def hessian_depends_on_theta(self) -> bool:
        """
        Gaussian Hessian does not depend on theta.

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
