"""Probit family implementation."""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseFamily


class ProbitFamily(BaseFamily):
    """
    Probit structural model for binary outcomes.

    Model: Y ~ Bernoulli(p) where p = Phi(alpha(X) + beta(X) * T)
           Phi is the standard normal CDF

    Loss: Probit NLL = -y*log(Phi(eta)) - (1-y)*log(1-Phi(eta))

    Parameters:
        theta = (alpha, beta) where
        - alpha(x): baseline latent propensity
        - beta(x): treatment effect on latent propensity

    Target: E[beta(X)] (average treatment effect on latent scale)

    Note: Similar to logit but uses normal CDF instead of logistic sigmoid.
    """

    theta_dim = 2

    def __init__(self, target: str = "beta", **kwargs):
        """
        Initialize probit family.

        Args:
            target: Target functional. Options:
                - "beta": E[beta(X)] (default)
                - "ame": Average Marginal Effect phi(eta)*beta
        """
        super().__init__(**kwargs)
        if target not in ("beta", "ame"):
            raise ValueError(f"target must be 'beta' or 'ame', got '{target}'")
        self.target = target

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Probit negative log-likelihood.

        NLL = -y*log(Phi(eta)) - (1-y)*log(1-Phi(eta))

        Args:
            y: (n,) binary outcomes {0, 1}
            t: (n,) treatments
            theta: (n, 2) parameters [alpha, beta]

        Returns:
            (n,) per-observation losses
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]
        eta = alpha + beta * t

        # Standard normal distribution
        dist = torch.distributions.Normal(0.0, 1.0)

        # Clamp CDF away from boundaries for numerical stability
        Phi = dist.cdf(eta)
        Phi = torch.clamp(Phi, 1e-7, 1 - 1e-7)

        # NLL = -y*log(Phi) - (1-y)*log(1-Phi)
        return -y * torch.log(Phi) - (1 - y) * torch.log(1 - Phi)

    def gradient(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """
        Closed-form gradient using Mills ratio.

        dl/d(alpha) = -phi(eta) * [(y/Phi) - ((1-y)/(1-Phi))]
        dl/d(beta) = t * dl/d(alpha)

        Simplifies to: -phi(eta) * (y - Phi) / [Phi * (1 - Phi)]

        Args:
            y: (n,) outcomes
            t: (n,) treatments
            theta: (n, 2) parameters [alpha, beta]

        Returns:
            (n, 2) gradient tensor
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]
        eta = alpha + beta * t

        dist = torch.distributions.Normal(0.0, 1.0)
        phi = torch.exp(dist.log_prob(eta))  # PDF
        Phi = torch.clamp(dist.cdf(eta), 1e-7, 1 - 1e-7)  # CDF

        # Weight: phi / [Phi * (1 - Phi)]
        w = phi / (Phi * (1 - Phi))

        # Residual: Phi - y (note sign flip from standard)
        residual = Phi - y

        grad_alpha = w * residual
        grad_beta = w * residual * t

        return torch.stack([grad_alpha, grad_beta], dim=1)

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """
        Return None to use autodiff for Hessian.

        The probit Hessian involves derivatives of Mills ratio which are
        algebraically messy. Autodiff is cleaner and equally accurate.
        """
        return None

    def hessian_depends_on_theta(self) -> bool:
        """Probit Hessian depends on theta through Phi(eta)."""
        return True

    def per_obs_target(self, theta: Tensor, t: Tensor) -> Tensor:
        """
        Per-observation target.

        - "beta": just beta(x)
        - "ame": phi(eta) * beta (average marginal effect)
        """
        if self.target == "ame":
            alpha = theta[:, 0]
            beta = theta[:, 1]
            eta = alpha + beta * t
            dist = torch.distributions.Normal(0.0, 1.0)
            phi = torch.exp(dist.log_prob(eta))
            return phi * beta
        return theta[:, 1]

    def per_obs_target_gradient(self, theta: Tensor, t: Tensor) -> Tensor:
        """
        Gradient of per-observation target w.r.t. theta.

        - "beta": [0, 1]
        - "ame": d(phi*beta)/d(alpha,beta) involves phi'
        """
        n = theta.shape[0]

        if self.target == "ame":
            alpha = theta[:, 0]
            beta = theta[:, 1]
            eta = alpha + beta * t

            dist = torch.distributions.Normal(0.0, 1.0)
            phi = torch.exp(dist.log_prob(eta))

            # d(phi*beta)/d(alpha) = beta * phi' = beta * (-eta * phi)
            # d(phi*beta)/d(beta) = phi + beta * phi' * t = phi + beta * (-eta * phi) * t
            grad_alpha = -beta * eta * phi
            grad_beta = phi * (1 - beta * eta * t)

            return torch.stack([grad_alpha, grad_beta], dim=1)

        # Default: [0, 1]
        grad = torch.zeros(n, 2, dtype=theta.dtype, device=theta.device)
        grad[:, 1] = 1.0
        return grad

    def residual(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Compute residual: y - Phi(eta).

        Args:
            y: (n,) outcomes
            t: (n,) treatments
            theta: (n, 2) parameters [alpha, beta]

        Returns:
            (n,) residuals
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]
        eta = alpha + beta * t
        dist = torch.distributions.Normal(0.0, 1.0)
        Phi = dist.cdf(eta)
        return y - Phi
