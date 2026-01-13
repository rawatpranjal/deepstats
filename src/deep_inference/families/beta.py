"""Beta regression family implementation."""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseFamily


class BetaFamily(BaseFamily):
    """
    Beta regression structural model for proportions Y in (0, 1).

    Model: Y ~ Beta(mu*phi, (1-mu)*phi)
           where mu = sigmoid(alpha(X) + beta(X) * T)
           phi is a fixed precision parameter

    Loss: Beta NLL = lgamma(mu*phi) + lgamma((1-mu)*phi) - lgamma(phi)
                    - (mu*phi - 1)*log(y) - ((1-mu)*phi - 1)*log(1-y)

    Parameters:
        theta = (alpha, beta) where
        - alpha(x): baseline logit(mean)
        - beta(x): treatment effect on logit(mean)

    Target: E[beta(X)] (average treatment effect on logit scale)

    Reference: Ferrari & Cribari-Neto (2004), Stata betareg
    """

    theta_dim = 2

    def __init__(self, precision: float = 1.0, target: str = "beta", **kwargs):
        """
        Initialize beta regression family.

        Args:
            precision: Precision parameter phi > 0. Higher = less variance.
                       Var(Y) = mu(1-mu)/(1+phi)
            target: Target functional ("beta" for coefficient)
        """
        super().__init__(**kwargs)
        if precision <= 0:
            raise ValueError(f"precision must be positive, got {precision}")
        self.phi = precision
        self.target = target

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Beta negative log-likelihood.

        NLL = lgamma(a) + lgamma(b) - lgamma(phi)
              - (a-1)*log(y) - (b-1)*log(1-y)

        where a = mu*phi, b = (1-mu)*phi, mu = sigmoid(eta)

        Args:
            y: (n,) outcomes in (0, 1)
            t: (n,) treatments
            theta: (n, 2) parameters [alpha, beta]

        Returns:
            (n,) per-observation losses
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]
        eta = alpha + beta * t

        # Mean via logit link
        mu = torch.sigmoid(eta)
        mu = torch.clamp(mu, 1e-6, 1 - 1e-6)  # Stability

        # Beta shape parameters
        phi = self.phi
        a = mu * phi
        b = (1 - mu) * phi

        # Clamp y away from boundaries
        y_safe = torch.clamp(y, 1e-7, 1 - 1e-7)

        # Beta NLL (up to constant lgamma(phi))
        nll = (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(torch.tensor(phi))
               - (a - 1) * torch.log(y_safe)
               - (b - 1) * torch.log(1 - y_safe))

        return nll

    def gradient(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """
        Closed-form gradient using digamma function.

        dl/d(eta) = phi * [psi(a) - psi(b) - log(y) + log(1-y)] * mu*(1-mu)
        dl/d(alpha) = dl/d(eta)
        dl/d(beta) = dl/d(eta) * t

        where psi is the digamma function.

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

        mu = torch.sigmoid(eta)
        mu = torch.clamp(mu, 1e-6, 1 - 1e-6)

        phi = self.phi
        a = mu * phi
        b = (1 - mu) * phi

        y_safe = torch.clamp(y, 1e-7, 1 - 1e-7)

        # Digamma terms
        psi_a = torch.digamma(a)
        psi_b = torch.digamma(b)

        # Gradient w.r.t. eta
        # d(NLL)/d(mu) = phi * [psi(a) - psi(b) - log(y) + log(1-y)]
        # d(mu)/d(eta) = mu * (1 - mu)
        grad_mu = phi * (psi_a - psi_b - torch.log(y_safe) + torch.log(1 - y_safe))
        grad_eta = grad_mu * mu * (1 - mu)

        grad_alpha = grad_eta
        grad_beta = grad_eta * t

        return torch.stack([grad_alpha, grad_beta], dim=1)

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """
        Return None to use autodiff for Hessian.

        Beta regression Hessian involves trigamma functions and is messy.
        Autodiff is cleaner.
        """
        return None

    def hessian_depends_on_theta(self) -> bool:
        """Beta Hessian depends on theta through mu = sigmoid(eta)."""
        return True

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
        mu = torch.sigmoid(alpha + beta * t)
        return y - mu
