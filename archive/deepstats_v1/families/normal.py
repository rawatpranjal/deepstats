"""Normal (Gaussian) distribution family.

Model: Y ~ N(mu, sigma^2)
where mu = g(X) (neural network output)

Link: identity (canonical)
    eta = mu

Variance function: V(mu) = 1
Dispersion: sigma^2 (needs to be estimated)
"""

from __future__ import annotations

import math

import torch

from .base import ExponentialFamily


class Normal(ExponentialFamily):
    """Gaussian/Normal distribution family.

    Model: Y ~ N(mu, sigma^2)
    where mu = g(X) (neural network output)

    Parameters
    ----------
    link : str, default="identity"
        Link function. Only "identity" supported for Normal.

    Examples
    --------
    >>> from deepstats.families import Normal
    >>> import torch
    >>>
    >>> family = Normal()
    >>> mu = torch.tensor([1.0, 2.0, 3.0])
    >>> eta = family.link(mu)  # eta = mu for identity link
    >>> V = family.variance(mu)  # V = 1 (constant)
    """

    name = "normal"
    has_dispersion = True
    canonical_link = "identity"

    def __init__(self, link: str = "identity") -> None:
        if link != "identity":
            raise ValueError("Normal family only supports identity link")
        self._link = link

    def link(self, mu: torch.Tensor) -> torch.Tensor:
        """Identity link: eta = mu."""
        return mu

    def inverse_link(self, eta: torch.Tensor) -> torch.Tensor:
        """Inverse identity: mu = eta."""
        return eta

    def variance(self, mu: torch.Tensor) -> torch.Tensor:
        """Constant variance: V(mu) = 1."""
        return torch.ones_like(mu)

    def link_derivative(self, mu: torch.Tensor) -> torch.Tensor:
        """g'(mu) = 1 for identity link."""
        return torch.ones_like(mu)

    def log_likelihood(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
        dispersion: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """Normal log-likelihood.

        log f(y|mu, sigma^2) = -0.5 * log(2*pi*sigma^2) - (y-mu)^2/(2*sigma^2)

        Parameters
        ----------
        y : torch.Tensor
            Observed values.
        mu : torch.Tensor
            Predicted means.
        dispersion : float
            Variance sigma^2.

        Returns
        -------
        torch.Tensor
            Log-likelihood for each observation.
        """
        sigma2 = dispersion
        if isinstance(sigma2, (int, float)):
            sigma2 = torch.tensor(sigma2, dtype=mu.dtype, device=mu.device)
        return -0.5 * (
            math.log(2 * math.pi) + torch.log(sigma2) + (y - mu) ** 2 / sigma2
        )

    def unit_deviance(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        """Unit deviance for Normal: (y - mu)^2."""
        return (y - mu) ** 2

    def validate_response(self, y: torch.Tensor) -> None:
        """Normal accepts any real-valued response."""
        if torch.isnan(y).any():
            raise ValueError("Response contains NaN values")
        if torch.isinf(y).any():
            raise ValueError("Response contains infinite values")
