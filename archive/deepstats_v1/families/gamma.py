"""Gamma distribution family for positive continuous data.

Model: Y ~ Gamma(shape=k, scale=mu/k)
where mu = g^{-1}(X) is the neural network output through inverse link.

Mean: E[Y] = mu
Variance: Var(Y) = mu² / k = mu² * dispersion

Link functions:
- log (common, default): eta = log(mu), mu = exp(eta)
- inverse (canonical): eta = 1/mu, mu = 1/eta

Dispersion: phi = 1/k where k is shape parameter

References
----------
- McCullagh & Nelder (1989). Generalized Linear Models, Ch 8.
"""

from __future__ import annotations

import torch

from .base import ExponentialFamily


class Gamma(ExponentialFamily):
    """Gamma distribution family for positive continuous data.

    Suitable for modeling positive continuous outcomes like durations,
    costs, or any positive-valued measurements with right skew.

    The parameterization is:
        Y ~ Gamma(shape=k, scale=mu/k)
        E[Y] = mu
        Var(Y) = mu² / k

    Parameters
    ----------
    link : str, default="log"
        Link function: "log" (common) or "inverse" (canonical).

    Examples
    --------
    >>> from deepstats.families import Gamma
    >>> import torch
    >>>
    >>> family = Gamma()
    >>> eta = torch.tensor([0.0, 1.0, 2.0])
    >>> mu = family.inverse_link(eta)  # mu = exp(eta)
    >>> V = family.variance(mu)  # V = mu²
    >>>
    >>> # With dispersion = 0.5 (shape k = 2):
    >>> # Var(Y) = mu² * 0.5
    """

    name = "gamma"
    has_dispersion = True  # Shape parameter k = 1/dispersion
    canonical_link = "inverse"

    def __init__(self, link: str = "log") -> None:
        if link not in ("log", "inverse"):
            raise ValueError(f"Gamma family supports 'log' or 'inverse' link, got {link}")
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
            # Ensure eta is positive for inverse link
            eta = torch.clamp(eta, min=1e-10)
            return 1.0 / eta

    def variance(self, mu: torch.Tensor) -> torch.Tensor:
        """Variance function: V(mu) = mu².

        Full variance is V(mu) * dispersion = mu² / k.
        """
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
        """Gamma log-likelihood.

        Using shape-scale parameterization:
            Y ~ Gamma(shape=k, scale=mu/k)
            where k = 1/dispersion

        log f(y|mu, k) = k*log(k) - k*log(mu) - log(Gamma(k))
                       + (k-1)*log(y) - k*y/mu

        Parameters
        ----------
        y : torch.Tensor
            Observed positive values.
        mu : torch.Tensor
            Predicted means.
        dispersion : float
            Dispersion parameter phi = 1/k.

        Returns
        -------
        torch.Tensor
            Log-likelihood for each observation.
        """
        y = torch.clamp(y, min=1e-10)
        mu = torch.clamp(mu, min=1e-10)

        # Shape parameter k = 1/phi
        if isinstance(dispersion, (int, float)):
            k = 1.0 / max(dispersion, 1e-10)
        else:
            k = 1.0 / torch.clamp(dispersion, min=1e-10)

        # Log-likelihood
        ll = (
            k * torch.log(torch.tensor(k, dtype=y.dtype, device=y.device))
            - k * torch.log(mu)
            - torch.lgamma(torch.tensor(k, dtype=y.dtype, device=y.device))
            + (k - 1) * torch.log(y)
            - k * y / mu
        )
        return ll

    def unit_deviance(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        """Unit deviance for Gamma.

        d_i = 2 * (-log(y/mu) + (y - mu)/mu)
            = 2 * ((y - mu)/mu - log(y/mu))

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

        # d = 2 * ((y - mu)/mu - log(y/mu))
        return 2 * ((y - mu) / mu - torch.log(y / mu))

    def validate_response(self, y: torch.Tensor) -> None:
        """Gamma requires strictly positive responses."""
        super().validate_response(y)
        if (y <= 0).any():
            raise ValueError("Gamma response must be strictly positive")
