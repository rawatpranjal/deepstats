"""Negative Binomial family for structural deep learning."""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseFamily


class NegBinFamily(BaseFamily):
    """Negative Binomial family: Y ~ NegBin(mu, r) where mu = exp(alpha + beta*T).

    Model:
        Y ~ NegBin(mu, r) where r = 1/overdispersion
        E[Y] = mu = exp(alpha(X) + beta(X) * T)
        Var[Y] = mu + overdispersion * mu^2 = mu + mu^2/r

    Parameters:
        theta = [alpha, beta] where mu = exp(alpha + beta*T)

    Loss:
        True NegBin NLL: -lgamma(y+r) + lgamma(r) + lgamma(y+1) + (r+y)*log(r+mu) - y*log(mu) - r*log(r)

    Target:
        E[beta(X)] - average effect on log-mean
    """

    theta_dim: int = 2

    def __init__(self, overdispersion: float = 0.5, **kwargs):
        """Initialize NegBinFamily.

        Args:
            overdispersion: Overdispersion parameter alpha > 0.
                           Var[Y] = mu + alpha * mu^2
                           r = 1/alpha (dispersion parameter)
        """
        super().__init__(**kwargs)
        if overdispersion <= 0:
            raise ValueError(f"overdispersion must be positive, got {overdispersion}")
        self.overdispersion = overdispersion
        self.r = 1.0 / overdispersion  # Dispersion parameter

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """True Negative Binomial NLL.

        NLL = -lgamma(y+r) + lgamma(r) + lgamma(y+1) + (r+y)*log(r+mu) - y*log(mu) - r*log(r)

        Note: We drop constant terms that don't depend on theta for optimization.
        Keeping: (r+y)*log(r+mu) - y*log(mu)
        """
        alpha, beta = theta[:, 0], theta[:, 1]
        eta = torch.clamp(alpha + beta * t, -20, 20)
        mu = torch.exp(eta)
        r = self.r

        # Full NLL (for proper likelihood)
        # We keep all terms for correctness
        y_safe = torch.clamp(y, min=0)
        mu_safe = torch.clamp(mu, min=1e-10)

        nll = (
            -torch.lgamma(y_safe + r)
            + torch.lgamma(torch.tensor(r, dtype=theta.dtype, device=theta.device))
            + torch.lgamma(y_safe + 1)
            + (r + y_safe) * torch.log(r + mu_safe)
            - y_safe * torch.log(mu_safe)
            - r * torch.log(torch.tensor(r, dtype=theta.dtype, device=theta.device))
        )
        return nll

    def gradient(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """Closed-form gradient of NegBin NLL.

        d(NLL)/d(eta) = r*(mu - y) / (r + mu)

        dl/d(alpha) = r*(mu - y) / (r + mu)
        dl/d(beta) = r*(mu - y) / (r + mu) * t
        """
        alpha, beta = theta[:, 0], theta[:, 1]
        eta = torch.clamp(alpha + beta * t, -20, 20)
        mu = torch.exp(eta)
        r = self.r

        # Gradient w.r.t. eta
        dl_deta = r * (mu - y) / (r + mu)

        grad = torch.stack([dl_deta, dl_deta * t], dim=1)
        return grad

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """Closed-form Hessian of NegBin NLL.

        d²(NLL)/d(eta)² = r * mu * (r + y) / (r + mu)²

        Note: This is the expected Fisher information when y is replaced by E[y]=mu:
              r * mu * (r + mu) / (r + mu)² = r * mu / (r + mu)
        """
        alpha, beta = theta[:, 0], theta[:, 1]
        eta = torch.clamp(alpha + beta * t, -20, 20)
        mu = torch.exp(eta)
        r = self.r

        # Use observed Hessian
        w = r * mu * (r + y) / (r + mu) ** 2

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
