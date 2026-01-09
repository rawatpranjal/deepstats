"""Negative Binomial family for structural deep learning."""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseFamily


class NegBinFamily(BaseFamily):
    """Negative Binomial family: Y ~ NegBin(mu, r) where mu = exp(alpha + beta*T).

    Model:
        Y ~ NegBin(mu, 1/overdispersion)
        E[Y] = mu = exp(alpha(X) + beta(X) * T)
        Var[Y] = mu + overdispersion * mu^2

    Parameters:
        theta = [alpha, beta] where mu = exp(alpha + beta*T)

    Loss:
        Poisson-like loss: mu - Y * log(mu)
        (Simplified form; full NegBin NLL includes terms in overdispersion)

    Target:
        E[beta(X)] - average effect on log-mean
    """

    theta_dim: int = 2

    def __init__(self, overdispersion: float = 0.5, **kwargs):
        """Initialize NegBinFamily.

        Args:
            overdispersion: Overdispersion parameter alpha > 0.
                           Var[Y] = mu + alpha * mu^2
        """
        super().__init__(**kwargs)
        if overdispersion <= 0:
            raise ValueError(f"overdispersion must be positive, got {overdispersion}")
        self.overdispersion = overdispersion

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """NegBin loss (Poisson-like): mu - Y * log(mu)."""
        alpha, beta = theta[:, 0], theta[:, 1]
        eta = torch.clamp(alpha + beta * t, -20, 20)
        mu = torch.exp(eta)
        return mu - y * torch.log(mu + 1e-10)

    def gradient(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """Closed-form gradient of loss.

        For Poisson-like loss:
        dl/d(alpha) = (mu - y)
        dl/d(beta) = (mu - y) * t
        """
        alpha, beta = theta[:, 0], theta[:, 1]
        eta = torch.clamp(alpha + beta * t, -20, 20)
        mu = torch.exp(eta)
        residual = mu - y
        grad = torch.stack([residual, residual * t], dim=1)
        return grad

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """Closed-form Hessian of loss.

        For NegBin, using the working weight mu / (1 + alpha*mu):
        H_ij = W * T_i * T_j where T = [1, t]
        """
        alpha, beta = theta[:, 0], theta[:, 1]
        eta = torch.clamp(alpha + beta * t, -20, 20)
        mu = torch.exp(eta)
        # NegBin working weight
        w = mu / (1.0 + self.overdispersion * mu)

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
