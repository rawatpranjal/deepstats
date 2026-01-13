"""Gaussian family implementation with MLE for sigma."""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseFamily


class GaussianFamily(BaseFamily):
    """
    Gaussian structural model with MLE for sigma.

    Model: Y ~ N(mu, sigma^2) where mu = alpha(X) + beta(X) * T, sigma = exp(gamma(X))

    Loss: Gaussian NLL = (y - mu)^2 / (2*sigma^2) + log(sigma)
                       = (y - mu)^2 * exp(-2*gamma) / 2 + gamma

    Parameters:
        theta = (alpha, beta, gamma) where
        - alpha(x): baseline outcome
        - beta(x): treatment effect
        - gamma(x): log standard deviation (sigma = exp(gamma))

    Target: E[beta(X)] (average treatment effect)

    Note: Unlike Linear family, this estimates sigma via MLE.
    """

    theta_dim = 3

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Gaussian negative log-likelihood with learned sigma.

        NLL = (y - mu)^2 / (2*sigma^2) + log(sigma)
            = (y - mu)^2 * exp(-2*gamma) / 2 + gamma

        Args:
            y: (n,) outcomes
            t: (n,) treatments
            theta: (n, 3) parameters [alpha, beta, gamma]

        Returns:
            (n,) per-observation losses
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]
        gamma = torch.clamp(theta[:, 2], -10, 10)  # Clamp for stability

        mu = alpha + beta * t
        sigma_sq = torch.exp(2 * gamma)

        return 0.5 * (y - mu) ** 2 / sigma_sq + gamma

    def gradient(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """
        Closed-form gradient.

        dl/d(alpha) = (mu - y) / sigma^2
        dl/d(beta) = t * (mu - y) / sigma^2
        dl/d(gamma) = 1 - (y - mu)^2 / sigma^2

        Args:
            y: (n,) outcomes
            t: (n,) treatments
            theta: (n, 3) parameters [alpha, beta, gamma]

        Returns:
            (n, 3) gradient tensor
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]
        gamma = torch.clamp(theta[:, 2], -10, 10)

        mu = alpha + beta * t
        sigma_sq = torch.exp(2 * gamma)
        residual = (mu - y) / sigma_sq
        sq_residual = (y - mu) ** 2 / sigma_sq

        grad_alpha = residual
        grad_beta = residual * t
        grad_gamma = 1 - sq_residual

        return torch.stack([grad_alpha, grad_beta, grad_gamma], dim=1)

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """
        Closed-form Hessian.

        L = (y - mu)^2 * exp(-2*gamma) / 2 + gamma

        H = [[1/sigma^2,  t/sigma^2,    -2*(mu-y)/sigma^2     ],
             [t/sigma^2,  t^2/sigma^2,  -2*t*(mu-y)/sigma^2   ],
             [-2*(mu-y)/sigma^2, -2*t*(mu-y)/sigma^2, 2*(y-mu)^2/sigma^2]]

        Args:
            y: (n,) outcomes
            t: (n,) treatments
            theta: (n, 3) parameters [alpha, beta, gamma]

        Returns:
            (n, 3, 3) Hessian tensor
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]
        gamma = torch.clamp(theta[:, 2], -10, 10)

        mu = alpha + beta * t
        sigma_sq = torch.exp(2 * gamma)

        scale = 1.0 / sigma_sq
        residual = (mu - y) / sigma_sq
        sq_residual = (y - mu) ** 2 / sigma_sq

        n = len(y)
        H = torch.zeros(n, 3, 3, dtype=theta.dtype, device=theta.device)

        # d2L/d(alpha)^2, d2L/d(alpha)d(beta), d2L/d(beta)^2
        H[:, 0, 0] = scale
        H[:, 0, 1] = scale * t
        H[:, 1, 0] = scale * t
        H[:, 1, 1] = scale * t ** 2

        # d2L/d(alpha)d(gamma), d2L/d(beta)d(gamma)
        # d/d(gamma)[(mu-y)*exp(-2*gamma)] = (mu-y)*(-2)*exp(-2*gamma) = -2*(mu-y)/sigma^2
        H[:, 0, 2] = -2 * residual
        H[:, 2, 0] = -2 * residual
        H[:, 1, 2] = -2 * residual * t
        H[:, 2, 1] = -2 * residual * t

        # d2L/d(gamma)^2
        H[:, 2, 2] = 2 * sq_residual

        return H

    def hessian_depends_on_theta(self) -> bool:
        """
        Gaussian MLE Hessian depends on theta through sigma = exp(gamma).
        """
        return True

    def residual(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Compute residual: y - mu.

        Args:
            y: (n,) outcomes
            t: (n,) treatments
            theta: (n, 3) parameters [alpha, beta, gamma]

        Returns:
            (n,) residuals
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]
        mu = alpha + beta * t
        return y - mu
