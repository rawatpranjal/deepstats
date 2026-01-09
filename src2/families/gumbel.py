"""Gumbel family for structural deep learning."""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseFamily


class GumbelFamily(BaseFamily):
    """Gumbel (Type I extreme value) family: Y ~ Gumbel(mu, scale).

    Model:
        Y ~ Gumbel(mu, scale) where mu = alpha(X) + beta(X) * T
        E[Y] = mu + scale * gamma (Euler-Mascheroni constant)
        Var[Y] = (pi * scale)^2 / 6

    Parameters:
        theta = [alpha, beta] where mu = alpha + beta*T

    Loss:
        Gumbel NLL: z + exp(-z) where z = (y - mu) / scale

    Target:
        E[beta(X)] - average effect on location parameter
    """

    theta_dim: int = 2

    def __init__(self, scale: float = 1.0, **kwargs):
        """Initialize GumbelFamily.

        Args:
            scale: Scale parameter > 0.
        """
        super().__init__(**kwargs)
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.scale = scale

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """Gumbel NLL: z + exp(-z) where z = (y - mu) / scale."""
        alpha, beta = theta[:, 0], theta[:, 1]
        mu = alpha + beta * t
        z = (y - mu) / self.scale
        return z + torch.exp(-z)

    def gradient(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """Closed-form gradient of loss.

        dl/dmu = -1/scale * (1 - exp(-z))
        dl/d(alpha) = dl/dmu
        dl/d(beta) = dl/dmu * t
        """
        alpha, beta = theta[:, 0], theta[:, 1]
        mu = alpha + beta * t
        z = (y - mu) / self.scale
        dl_dmu = -1.0 / self.scale * (1 - torch.exp(-z))
        grad = torch.stack([dl_dmu, dl_dmu * t], dim=1)
        return grad

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """Closed-form Hessian of loss.

        d2l/dmu^2 = -exp(-z) / scale^2
        """
        alpha, beta = theta[:, 0], theta[:, 1]
        mu = alpha + beta * t
        z = (y - mu) / self.scale
        w = torch.exp(-z) / (self.scale ** 2)

        n = theta.shape[0]
        H = torch.zeros(n, 2, 2, dtype=theta.dtype, device=theta.device)
        H[:, 0, 0] = w
        H[:, 0, 1] = w * t
        H[:, 1, 0] = w * t
        H[:, 1, 1] = w * t * t
        return H

    def hessian_depends_on_theta(self) -> bool:
        """Hessian depends on theta through z = (y - mu) / scale."""
        return True
