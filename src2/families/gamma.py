"""Gamma family for structural deep learning."""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseFamily


class GammaFamily(BaseFamily):
    """Gamma family: Y ~ Gamma(shape, mu/shape) where mu = exp(alpha + beta*T).

    Model:
        E[Y | X, T] = mu = exp(alpha(X) + beta(X) * T)
        Var[Y | X, T] = mu^2 / shape

    Parameters:
        theta = [alpha, beta] where mu = exp(alpha + beta*T)

    Loss:
        Gamma deviance: Y/mu + log(mu) (up to constants)

    Target:
        E[beta(X)] - average effect on log-mean
    """

    theta_dim: int = 2

    def __init__(self, shape: float = 2.0, **kwargs):
        """Initialize GammaFamily.

        Args:
            shape: Shape parameter k > 0. Higher = less variance.
        """
        super().__init__(**kwargs)
        self.shape = shape

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """Gamma deviance loss: Y/mu + log(mu)."""
        alpha, beta = theta[:, 0], theta[:, 1]
        eta = torch.clamp(alpha + beta * t, -10, 10)
        mu = torch.exp(eta)
        return y / mu + torch.log(mu)

    def gradient(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """Closed-form gradient of loss.

        dl/d(alpha) = (mu - y) / mu = 1 - y/mu
        dl/d(beta) = t * (mu - y) / mu = t * (1 - y/mu)
        """
        alpha, beta = theta[:, 0], theta[:, 1]
        eta = torch.clamp(alpha + beta * t, -10, 10)
        mu = torch.exp(eta)
        residual = 1 - y / torch.clamp(mu, min=1e-6)
        grad = torch.stack([residual, residual * t], dim=1)
        return grad

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """Closed-form Hessian of loss.

        d2l/d(alpha)^2 = y/mu
        d2l/d(alpha)d(beta) = t * y/mu
        d2l/d(beta)^2 = t^2 * y/mu
        """
        alpha, beta = theta[:, 0], theta[:, 1]
        eta = torch.clamp(alpha + beta * t, -10, 10)
        mu = torch.exp(eta)
        w = y / torch.clamp(mu, min=1e-6)

        n = theta.shape[0]
        H = torch.zeros(n, 2, 2, dtype=theta.dtype, device=theta.device)
        H[:, 0, 0] = w
        H[:, 0, 1] = w * t
        H[:, 1, 0] = w * t
        H[:, 1, 1] = w * t * t
        return H

    def hessian_depends_on_theta(self) -> bool:
        """Hessian depends on theta through mu = exp(alpha + beta*T)."""
        return True
