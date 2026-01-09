"""Weibull family for structural deep learning."""

import math
import torch
from torch import Tensor
from typing import Optional

from .base import BaseFamily


class WeibullFamily(BaseFamily):
    """Weibull family: Y ~ Weibull(shape, lambda) where lambda = exp(alpha + beta*T).

    Model:
        Y ~ Weibull(k, lambda) where lambda = exp(alpha(X) + beta(X) * T)
        f(y) = (k/lambda) * (y/lambda)^(k-1) * exp(-(y/lambda)^k)
        E[Y] = lambda * Gamma(1 + 1/k)

    Parameters:
        theta = [alpha, beta] where lambda = exp(alpha + beta*T)

    Loss:
        Weibull NLL: -log(k) + k*log(lambda) - (k-1)*log(y) + (y/lambda)^k

    Target:
        E[beta(X)] - average effect on log-scale parameter
    """

    theta_dim: int = 2

    def __init__(self, shape: float = 2.0, **kwargs):
        """Initialize WeibullFamily.

        Args:
            shape: Shape parameter k > 0.
                   k < 1: decreasing hazard
                   k = 1: exponential (constant hazard)
                   k > 1: increasing hazard
        """
        super().__init__(**kwargs)
        if shape <= 0:
            raise ValueError(f"shape must be positive, got {shape}")
        self.shape = shape

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """Weibull NLL loss."""
        alpha, beta = theta[:, 0], theta[:, 1]
        eta = torch.clamp(alpha + beta * t, -20, 20)
        lam = torch.exp(eta)
        k = self.shape

        # Clamp y to avoid log(0)
        y_safe = torch.clamp(y, min=1e-10)
        z = (y_safe / lam) ** k

        # NLL: -log(k) + k*log(lambda) - (k-1)*log(y) + z
        return -math.log(k) + k * eta - (k - 1) * torch.log(y_safe) + z

    def gradient(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """Closed-form gradient of loss.

        dl/d(eta) = k - k * (y/lambda)^k = k * (1 - z)
        dl/d(alpha) = dl/d(eta)
        dl/d(beta) = dl/d(eta) * t
        """
        alpha, beta = theta[:, 0], theta[:, 1]
        eta = torch.clamp(alpha + beta * t, -20, 20)
        lam = torch.exp(eta)
        k = self.shape
        y_safe = torch.clamp(y, min=1e-10)
        z = (y_safe / lam) ** k

        dl_deta = k * (1 - z)
        grad = torch.stack([dl_deta, dl_deta * t], dim=1)
        return grad

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """Closed-form Hessian of loss.

        d2l/d(eta)^2 = k^2 * z = k^2 * (y/lambda)^k
        """
        alpha, beta = theta[:, 0], theta[:, 1]
        eta = torch.clamp(alpha + beta * t, -20, 20)
        lam = torch.exp(eta)
        k = self.shape
        y_safe = torch.clamp(y, min=1e-10)
        z = (y_safe / lam) ** k

        w = k ** 2 * z

        n = theta.shape[0]
        H = torch.zeros(n, 2, 2, dtype=theta.dtype, device=theta.device)
        H[:, 0, 0] = w
        H[:, 0, 1] = w * t
        H[:, 1, 0] = w * t
        H[:, 1, 1] = w * t * t
        return H

    def hessian_depends_on_theta(self) -> bool:
        """Hessian depends on theta through lambda = exp(alpha + beta*T)."""
        return True
