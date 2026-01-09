"""Tobit family for structural deep learning."""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseFamily


class TobitFamily(BaseFamily):
    """Tobit (censored regression) family: Y = max(0, Y*) where Y* = alpha + beta*T + sigma*eps.

    Model:
        Y* = alpha(X) + beta(X) * T + sigma(X) * eps, eps ~ N(0, 1)
        Y = max(0, Y*)  (left-censored at 0)
        sigma = exp(gamma) where gamma is the third parameter

    Parameters:
        theta = [alpha, beta, gamma] where sigma = exp(gamma)

    Loss:
        Tobit NLL:
        - Uncensored (Y > 0): log(sigma) + (Y - mu)^2 / (2*sigma^2)
        - Censored (Y = 0): -log(Phi(-z)) where z = mu/sigma

    Targets:
        - 'latent': E[beta(X)] - effect on latent Y*
        - 'observed': E[beta(X) * Phi(z)] - average effect on observed outcome
    """

    theta_dim: int = 3

    def __init__(self, target: str = "latent", **kwargs):
        """Initialize TobitFamily.

        Args:
            target: 'latent' (effect on Y*) or 'observed' (effect on E[Y|X,T])
        """
        super().__init__(**kwargs)
        if target not in ("latent", "observed"):
            raise ValueError(f"target must be 'latent' or 'observed', got '{target}'")
        self.target = target

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """Tobit NLL loss."""
        alpha, beta, gamma = theta[:, 0], theta[:, 1], theta[:, 2]
        gamma = torch.clamp(gamma, -10, 10)
        sigma = torch.exp(gamma)
        mu = alpha + beta * t
        z = mu / sigma

        censored = (y <= 0)
        dist = torch.distributions.Normal(0, 1)

        # Uncensored: log(sigma) + (y-mu)^2 / (2*sigma^2)
        nll_uncensored = gamma + 0.5 * ((y - mu) / sigma) ** 2

        # Censored: -log(Phi(-z)) = -log(1 - Phi(z))
        log_Phi_neg_z = dist.cdf(-z).clamp(min=1e-10).log()
        nll_censored = -log_Phi_neg_z

        return torch.where(censored, nll_censored, nll_uncensored)

    def hessian_depends_on_theta(self) -> bool:
        """Hessian depends on theta through z = mu/sigma."""
        return True

    def default_target(self, x: Tensor, theta: Tensor) -> Tensor:
        """Default target based on target type."""
        if self.target == "observed":
            # E[beta * Phi(z)] - need T for this, approximate with mean
            return theta[:, 1].mean()
        return theta[:, 1].mean()

    def per_obs_target(self, theta: Tensor, t: Tensor) -> Tensor:
        """Per-observation target h(theta, t).

        For latent: h = beta
        For observed: h = beta * Phi(z) where z = (alpha + beta*t) / sigma
        """
        if self.target == "observed":
            alpha, beta, gamma = theta[:, 0], theta[:, 1], theta[:, 2]
            sigma = torch.exp(torch.clamp(gamma, -10, 10))
            z = (alpha + beta * t) / sigma
            Phi_z = torch.distributions.Normal(0, 1).cdf(z)
            return beta * Phi_z
        return theta[:, 1]

    def per_obs_target_gradient(self, theta: Tensor, t: Tensor) -> Tensor:
        """Gradient of per-observation target w.r.t. theta.

        For latent: grad = [0, 1, 0]
        For observed: grad = [beta*phi(z)/sigma, Phi(z) + beta*t*phi(z)/sigma, -beta*z*phi(z)]
        """
        n = theta.shape[0]

        if self.target == "observed":
            alpha, beta, gamma = theta[:, 0], theta[:, 1], theta[:, 2]
            sigma = torch.exp(torch.clamp(gamma, -10, 10))
            z = (alpha + beta * t) / sigma
            dist = torch.distributions.Normal(0, 1)
            phi_z = torch.exp(dist.log_prob(z))
            Phi_z = dist.cdf(z)

            grad = torch.zeros(n, 3, dtype=theta.dtype, device=theta.device)
            # d(beta*Phi)/d(alpha) = beta * phi(z) / sigma
            grad[:, 0] = beta * phi_z / sigma
            # d(beta*Phi)/d(beta) = Phi(z) + beta * t * phi(z) / sigma
            grad[:, 1] = Phi_z + beta * t * phi_z / sigma
            # d(beta*Phi)/d(gamma) = -beta * z * phi(z) (since d(sigma)/d(gamma) = sigma)
            grad[:, 2] = -beta * z * phi_z
            return grad

        # Latent target: grad = [0, 1, 0]
        grad = torch.zeros(n, 3, dtype=theta.dtype, device=theta.device)
        grad[:, 1] = 1.0
        return grad
