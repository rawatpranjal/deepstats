"""Bernoulli/Binomial distribution family for binary outcomes.

Model: Y ~ Bernoulli(p)
where p = sigmoid(g(X)) (neural network output through sigmoid)

Link: logit (canonical)
    eta = log(mu / (1-mu))
    mu = 1 / (1 + exp(-eta))

Variance function: V(mu) = mu * (1 - mu)
Dispersion: phi = 1 (fixed for binomial)
"""

from __future__ import annotations

import math

import torch

from .base import ExponentialFamily


class Bernoulli(ExponentialFamily):
    """Bernoulli/Binomial distribution family for binary outcomes.

    Model: Y ~ Bernoulli(p)
    where p = sigmoid(g(X))

    Parameters
    ----------
    link : str, default="logit"
        Link function: "logit" (canonical) or "probit".

    Examples
    --------
    >>> from deepstats.families import Bernoulli
    >>> import torch
    >>>
    >>> family = Bernoulli()
    >>> eta = torch.tensor([0.0, 1.0, -1.0])
    >>> p = family.inverse_link(eta)  # p = sigmoid(eta)
    >>> V = family.variance(p)  # V = p * (1 - p)
    """

    name = "bernoulli"
    has_dispersion = False
    canonical_link = "logit"

    def __init__(self, link: str = "logit") -> None:
        if link not in ("logit", "probit"):
            raise ValueError("Bernoulli family supports 'logit' or 'probit' link")
        self._link = link

    def link(self, mu: torch.Tensor) -> torch.Tensor:
        """Logit or probit link.

        For logit: eta = log(mu / (1-mu))
        For probit: eta = Phi^{-1}(mu)
        """
        mu = torch.clamp(mu, min=1e-7, max=1 - 1e-7)
        if self._link == "logit":
            return torch.log(mu / (1 - mu))
        else:  # probit
            return torch.erfinv(2 * mu - 1) * math.sqrt(2)

    def inverse_link(self, eta: torch.Tensor) -> torch.Tensor:
        """Sigmoid (logit) or normal CDF (probit) mean function.

        For logit: mu = 1/(1+exp(-eta))
        For probit: mu = Phi(eta)
        """
        if self._link == "logit":
            return torch.sigmoid(eta)
        else:  # probit
            return 0.5 * (1 + torch.erf(eta / math.sqrt(2)))

    def variance(self, mu: torch.Tensor) -> torch.Tensor:
        """Binomial variance: V(mu) = mu * (1-mu)."""
        mu = torch.clamp(mu, min=1e-7, max=1 - 1e-7)
        return mu * (1 - mu)

    def link_derivative(self, mu: torch.Tensor) -> torch.Tensor:
        """Derivative of link function.

        For logit: g'(mu) = 1/(mu*(1-mu))
        For probit: g'(mu) = 1/phi(Phi^{-1}(mu))
        """
        mu = torch.clamp(mu, min=1e-7, max=1 - 1e-7)
        if self._link == "logit":
            return 1.0 / (mu * (1 - mu))
        else:  # probit
            # Derivative of probit is 1/phi(Phi^{-1}(mu))
            eta = self.link(mu)
            normal_pdf = torch.exp(-0.5 * eta**2) / math.sqrt(2 * math.pi)
            return 1.0 / torch.clamp(normal_pdf, min=1e-10)

    def log_likelihood(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
        dispersion: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """Bernoulli log-likelihood (binary cross-entropy).

        log f(y|mu) = y * log(mu) + (1-y) * log(1-mu)

        Parameters
        ----------
        y : torch.Tensor
            Binary outcomes (0 or 1).
        mu : torch.Tensor
            Predicted probabilities.
        dispersion : float
            Not used for Bernoulli (fixed at 1).

        Returns
        -------
        torch.Tensor
            Log-likelihood for each observation.
        """
        mu = torch.clamp(mu, min=1e-7, max=1 - 1e-7)
        return y * torch.log(mu) + (1 - y) * torch.log(1 - mu)

    def unit_deviance(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        """Unit deviance for Bernoulli.

        d_i = 2 * (y * log(y/mu) + (1-y) * log((1-y)/(1-mu)))

        Parameters
        ----------
        y : torch.Tensor
            Binary outcomes.
        mu : torch.Tensor
            Predicted probabilities.

        Returns
        -------
        torch.Tensor
            Unit deviance for each observation.
        """
        mu = torch.clamp(mu, min=1e-7, max=1 - 1e-7)
        # Handle y=0 and y=1 cases
        term1 = torch.where(
            y > 0.5, y * torch.log(y / mu), torch.zeros_like(y)
        )
        term2 = torch.where(
            y < 0.5, (1 - y) * torch.log((1 - y) / (1 - mu)), torch.zeros_like(y)
        )
        return 2 * (term1 + term2)

    def validate_response(self, y: torch.Tensor) -> None:
        """Bernoulli requires responses in [0, 1]."""
        if (y < 0).any() or (y > 1).any():
            raise ValueError("Bernoulli response must be in [0, 1]")
        if torch.isnan(y).any():
            raise ValueError("Response contains NaN values")
