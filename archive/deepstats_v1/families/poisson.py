"""Poisson distribution family for count data.

Model: Y ~ Poisson(lambda)
where lambda = exp(g(X)) (neural network output through exp)

Link: log (canonical)
    eta = log(mu)
    mu = exp(eta)

Variance function: V(mu) = mu
Dispersion: phi = 1 (fixed, but quasi-Poisson allows estimation)
"""

from __future__ import annotations

import torch

from .base import ExponentialFamily


class Poisson(ExponentialFamily):
    """Poisson distribution family for count data.

    Model: Y ~ Poisson(lambda)
    where lambda = exp(g(X))

    Parameters
    ----------
    link : str, default="log"
        Link function. Only "log" supported currently.

    Examples
    --------
    >>> from deepstats.families import Poisson
    >>> import torch
    >>>
    >>> family = Poisson()
    >>> eta = torch.tensor([0.0, 1.0, 2.0])
    >>> mu = family.inverse_link(eta)  # mu = exp(eta)
    >>> V = family.variance(mu)  # V = mu
    """

    name = "poisson"
    has_dispersion = False  # Fixed at 1, quasi-Poisson would be True
    canonical_link = "log"

    def __init__(self, link: str = "log") -> None:
        if link != "log":
            raise ValueError("Poisson family only supports log link currently")
        self._link = link

    def link(self, mu: torch.Tensor) -> torch.Tensor:
        """Log link: eta = log(mu)."""
        return torch.log(torch.clamp(mu, min=1e-10))

    def inverse_link(self, eta: torch.Tensor) -> torch.Tensor:
        """Exp mean function: mu = exp(eta).

        Clamp to prevent numerical overflow.
        """
        return torch.exp(torch.clamp(eta, max=20))

    def variance(self, mu: torch.Tensor) -> torch.Tensor:
        """Variance equals mean: V(mu) = mu."""
        return torch.clamp(mu, min=1e-10)

    def link_derivative(self, mu: torch.Tensor) -> torch.Tensor:
        """g'(mu) = 1/mu for log link."""
        return 1.0 / torch.clamp(mu, min=1e-10)

    def log_likelihood(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
        dispersion: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """Poisson log-likelihood.

        log f(y|mu) = y * log(mu) - mu - log(y!)

        Note: log(y!) computed via lgamma(y+1) for differentiability.

        Parameters
        ----------
        y : torch.Tensor
            Observed count values.
        mu : torch.Tensor
            Predicted means.
        dispersion : float
            Not used for Poisson (fixed at 1).

        Returns
        -------
        torch.Tensor
            Log-likelihood for each observation.
        """
        mu = torch.clamp(mu, min=1e-10)
        return y * torch.log(mu) - mu - torch.lgamma(y + 1)

    def unit_deviance(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        """Unit deviance for Poisson.

        d_i = 2 * (y * log(y/mu) - (y - mu))
        With convention 0 * log(0) = 0.

        Parameters
        ----------
        y : torch.Tensor
            Observed values.
        mu : torch.Tensor
            Predicted means.

        Returns
        -------
        torch.Tensor
            Unit deviance for each observation.
        """
        mu = torch.clamp(mu, min=1e-10)
        # Handle y=0 case: 0 * log(0/mu) = 0
        y_safe = torch.clamp(y, min=1e-10)
        term1 = torch.where(y > 0, y * torch.log(y_safe / mu), torch.zeros_like(y))
        term2 = y - mu
        return 2 * (term1 - term2)

    def validate_response(self, y: torch.Tensor) -> None:
        """Poisson requires non-negative responses."""
        if (y < 0).any():
            raise ValueError("Poisson response must be non-negative")
        if torch.isnan(y).any():
            raise ValueError("Response contains NaN values")
