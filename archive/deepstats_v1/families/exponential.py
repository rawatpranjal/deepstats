"""Exponential distribution family for positive continuous data.

Model: Y ~ Exponential(rate=1/mu)
where mu = g^{-1}(X) is the neural network output through inverse link.

This is a special case of Gamma with shape k=1.

Mean: E[Y] = mu
Variance: Var(Y) = mu²

Link functions:
- log (common, default): eta = log(mu), mu = exp(eta)
- inverse (canonical): eta = 1/mu, mu = 1/eta

Dispersion: Fixed at 1 (since shape k=1).

References
----------
- McCullagh & Nelder (1989). Generalized Linear Models, Ch 8.
"""

from __future__ import annotations

import torch

from .base import ExponentialFamily


class Exponential(ExponentialFamily):
    """Exponential distribution family for positive continuous data.

    A special case of Gamma with shape=1. Suitable for modeling
    time-to-event data (survival analysis), inter-arrival times,
    or any memoryless positive continuous process.

    The parameterization is:
        Y ~ Exponential(rate=1/mu)
        E[Y] = mu
        Var(Y) = mu²

    Parameters
    ----------
    link : str, default="log"
        Link function: "log" (common) or "inverse" (canonical).

    Examples
    --------
    >>> from deepstats.families import Exponential
    >>> import torch
    >>>
    >>> family = Exponential()
    >>> eta = torch.tensor([0.0, 1.0, 2.0])
    >>> mu = family.inverse_link(eta)  # mu = exp(eta)
    >>> V = family.variance(mu)  # V = mu²
    """

    name = "exponential"
    has_dispersion = False  # Fixed at 1 (shape k=1)
    canonical_link = "inverse"

    def __init__(self, link: str = "log") -> None:
        if link not in ("log", "inverse"):
            raise ValueError(f"Exponential family supports 'log' or 'inverse' link, got {link}")
        self._link = link

    def link(self, mu: torch.Tensor) -> torch.Tensor:
        """Link function: eta = g(mu).

        For log link: eta = log(mu)
        For inverse link: eta = 1/mu
        """
        mu = torch.clamp(mu, min=1e-10)
        if self._link == "log":
            return torch.log(mu)
        else:  # inverse
            return 1.0 / mu

    def inverse_link(self, eta: torch.Tensor) -> torch.Tensor:
        """Inverse link (mean function): mu = g^{-1}(eta).

        For log link: mu = exp(eta)
        For inverse link: mu = 1/eta

        Clamp to prevent numerical issues.
        """
        if self._link == "log":
            return torch.exp(torch.clamp(eta, min=-20, max=20))
        else:  # inverse
            eta = torch.clamp(eta, min=1e-10)
            return 1.0 / eta

    def variance(self, mu: torch.Tensor) -> torch.Tensor:
        """Variance function: V(mu) = mu²."""
        return torch.clamp(mu, min=1e-10) ** 2

    def link_derivative(self, mu: torch.Tensor) -> torch.Tensor:
        """Derivative of link function: g'(mu).

        For log link: g'(mu) = 1/mu
        For inverse link: g'(mu) = -1/mu²
        """
        mu = torch.clamp(mu, min=1e-10)
        if self._link == "log":
            return 1.0 / mu
        else:  # inverse
            return -1.0 / (mu ** 2)

    def log_likelihood(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
        dispersion: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """Exponential log-likelihood.

        Y ~ Exp(rate=1/mu)

        log f(y|mu) = -log(mu) - y/mu

        Parameters
        ----------
        y : torch.Tensor
            Observed positive values.
        mu : torch.Tensor
            Predicted means.
        dispersion : float
            Not used for Exponential (fixed at 1).

        Returns
        -------
        torch.Tensor
            Log-likelihood for each observation.
        """
        y = torch.clamp(y, min=1e-10)
        mu = torch.clamp(mu, min=1e-10)

        return -torch.log(mu) - y / mu

    def unit_deviance(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        """Unit deviance for Exponential.

        Same as Gamma with k=1:
        d_i = 2 * ((y - mu)/mu - log(y/mu))

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
        y = torch.clamp(y, min=1e-10)
        mu = torch.clamp(mu, min=1e-10)

        return 2 * ((y - mu) / mu - torch.log(y / mu))

    def validate_response(self, y: torch.Tensor) -> None:
        """Exponential requires strictly positive responses."""
        super().validate_response(y)
        if (y <= 0).any():
            raise ValueError("Exponential response must be strictly positive")
