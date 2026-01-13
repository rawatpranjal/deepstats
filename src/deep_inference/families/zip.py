"""Zero-Inflated Poisson (ZIP) family implementation."""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseFamily


class ZIPFamily(BaseFamily):
    """
    Zero-Inflated Poisson structural model for count data with excess zeros.

    Model: Mixture of structural zeros and Poisson process
           P(Y=0) = pi + (1-pi)*exp(-lambda)
           P(Y=k) = (1-pi) * Poisson(k; lambda)  for k > 0

    Parameterization:
        lambda = exp(alpha(X) + beta(X) * T)     [Poisson rate]
        pi = sigmoid(gamma(X) + delta(X) * T)   [Zero-inflation probability]

    Parameters:
        theta = (alpha, beta, gamma, delta) where
        - alpha(x): baseline log-rate
        - beta(x): treatment effect on log-rate
        - gamma(x): baseline logit(zero-inflation)
        - delta(x): treatment effect on logit(zero-inflation)

    Target: E[beta(X)] (average treatment effect on log-rate)

    Reference: Lambert (1992), Stata zip
    """

    theta_dim = 4

    def __init__(self, target: str = "beta", **kwargs):
        """
        Initialize ZIP family.

        Args:
            target: Target functional.
                - "beta": E[beta(X)] - treatment effect on log-rate
                - "delta": E[delta(X)] - treatment effect on zero-inflation
        """
        super().__init__(**kwargs)
        if target not in ("beta", "delta"):
            raise ValueError(f"target must be 'beta' or 'delta', got '{target}'")
        self.target = target

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Zero-Inflated Poisson negative log-likelihood.

        For Y=0: NLL = -log(pi + (1-pi)*exp(-lambda))
        For Y>0: NLL = -log(1-pi) + lambda - y*log(lambda) + lgamma(y+1)

        Args:
            y: (n,) count outcomes (non-negative integers)
            t: (n,) treatments
            theta: (n, 4) parameters [alpha, beta, gamma, delta]

        Returns:
            (n,) per-observation losses
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]
        gamma = theta[:, 2]
        delta = theta[:, 3]

        # Poisson rate
        eta_rate = torch.clamp(alpha + beta * t, -20, 10)
        lam = torch.exp(eta_rate)

        # Zero-inflation probability
        eta_zero = gamma + delta * t
        pi = torch.sigmoid(eta_zero)
        pi = torch.clamp(pi, 1e-7, 1 - 1e-7)

        # Indicator for zeros
        is_zero = (y == 0)

        # P(Y=0) = pi + (1-pi)*exp(-lambda)
        prob_zero = pi + (1 - pi) * torch.exp(-lam)
        prob_zero = torch.clamp(prob_zero, 1e-10, 1.0)
        nll_zero = -torch.log(prob_zero)

        # P(Y=k>0) = (1-pi) * Poisson(k; lambda)
        # NLL = -log(1-pi) + lambda - y*log(lambda) + log(y!)
        nll_pos = (-torch.log(1 - pi)
                   + lam
                   - y * torch.log(lam + 1e-10)
                   + torch.lgamma(y + 1))

        return torch.where(is_zero, nll_zero, nll_pos)

    def gradient(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """
        Return None to use autodiff for gradient.

        ZIP gradient involves mixture weights and is complex.
        Autodiff is cleaner and equally accurate.
        """
        return None

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """
        Return None to use autodiff for Hessian.

        ZIP is a mixture model with complex cross-derivatives.
        """
        return None

    def hessian_depends_on_theta(self) -> bool:
        """ZIP Hessian depends on theta through lambda and pi."""
        return True

    def per_obs_target(self, theta: Tensor, t: Tensor) -> Tensor:
        """
        Per-observation target.

        - "beta": beta(x) - treatment effect on log-rate
        - "delta": delta(x) - treatment effect on zero-inflation
        """
        if self.target == "delta":
            return theta[:, 3]
        return theta[:, 1]  # Default: beta

    def per_obs_target_gradient(self, theta: Tensor, t: Tensor) -> Tensor:
        """
        Gradient of per-observation target w.r.t. theta.

        - "beta": [0, 1, 0, 0]
        - "delta": [0, 0, 0, 1]
        """
        n = theta.shape[0]
        grad = torch.zeros(n, 4, dtype=theta.dtype, device=theta.device)

        if self.target == "delta":
            grad[:, 3] = 1.0
        else:
            grad[:, 1] = 1.0

        return grad

    def residual(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Compute residual: y - E[Y].

        E[Y] = (1-pi) * lambda for ZIP model.

        Args:
            y: (n,) outcomes
            t: (n,) treatments
            theta: (n, 4) parameters [alpha, beta, gamma, delta]

        Returns:
            (n,) residuals
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]
        gamma = theta[:, 2]
        delta = theta[:, 3]

        lam = torch.exp(torch.clamp(alpha + beta * t, -20, 10))
        pi = torch.sigmoid(gamma + delta * t)

        expected = (1 - pi) * lam
        return y - expected
