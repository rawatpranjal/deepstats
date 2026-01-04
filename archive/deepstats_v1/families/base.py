"""Base class for exponential family distributions.

This module defines the abstract base class for GLM distribution families,
following the exponential family formulation.

References
----------
- McCullagh & Nelder (1989). Generalized Linear Models.
- Farrell, Liang, Misra (2021). Deep Neural Networks for Estimation and Inference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class ExponentialFamily(ABC):
    """Abstract base class for exponential family distributions.

    Following GLM theory, an exponential family has density:
        f(y|theta, phi) = exp((y*theta - b(theta))/a(phi) + c(y, phi))

    where:
        - theta is the natural (canonical) parameter
        - phi is the dispersion parameter
        - b(theta) is the cumulant function (log-partition)
        - a(phi) = phi/w for known weights w

    The link function g connects mean mu to linear predictor eta:
        eta = g(mu)

    Attributes
    ----------
    name : str
        Distribution family name.
    has_dispersion : bool
        Whether family has unknown dispersion parameter.
    canonical_link : str
        Name of canonical link function.
    """

    name: str
    has_dispersion: bool = False
    canonical_link: str

    @abstractmethod
    def link(self, mu: torch.Tensor) -> torch.Tensor:
        """Link function: eta = g(mu).

        Maps the mean to the linear predictor space.

        Parameters
        ----------
        mu : torch.Tensor
            Mean values.

        Returns
        -------
        torch.Tensor
            Linear predictor values.
        """
        pass

    @abstractmethod
    def inverse_link(self, eta: torch.Tensor) -> torch.Tensor:
        """Inverse link (mean function): mu = g^{-1}(eta).

        Maps linear predictor to mean space.

        Parameters
        ----------
        eta : torch.Tensor
            Linear predictor values.

        Returns
        -------
        torch.Tensor
            Mean values.
        """
        pass

    @abstractmethod
    def variance(self, mu: torch.Tensor) -> torch.Tensor:
        """Variance function: V(mu).

        Variance as a function of the mean (up to dispersion).

        For Normal: V(mu) = 1
        For Poisson: V(mu) = mu
        For Bernoulli: V(mu) = mu(1-mu)

        Parameters
        ----------
        mu : torch.Tensor
            Mean values.

        Returns
        -------
        torch.Tensor
            Variance values.
        """
        pass

    @abstractmethod
    def log_likelihood(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
        dispersion: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """Log-likelihood contribution for each observation.

        Parameters
        ----------
        y : torch.Tensor
            Observed values (n,) or (n, 1).
        mu : torch.Tensor
            Predicted means (n,) or (n, 1).
        dispersion : torch.Tensor or float
            Dispersion parameter (phi). For Normal, this is sigma^2.

        Returns
        -------
        torch.Tensor
            Log-likelihood for each observation (n,).
        """
        pass

    @abstractmethod
    def unit_deviance(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        """Unit deviance: 2 * (log f(y|y) - log f(y|mu)).

        The deviance measures goodness of fit relative to saturated model.

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
        pass

    def deviance_residuals(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        """Deviance residuals: sign(y-mu) * sqrt(d_i).

        Where d_i is the unit deviance for observation i.

        Parameters
        ----------
        y : torch.Tensor
            Observed values.
        mu : torch.Tensor
            Predicted means.

        Returns
        -------
        torch.Tensor
            Deviance residuals.
        """
        d = self.unit_deviance(y, mu)
        sign = torch.sign(y - mu)
        return sign * torch.sqrt(torch.clamp(d, min=0))

    def deviance(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
        dispersion: float = 1.0,
    ) -> torch.Tensor:
        """Scaled deviance: sum of unit deviances / dispersion.

        Parameters
        ----------
        y : torch.Tensor
            Observed values.
        mu : torch.Tensor
            Predicted means.
        dispersion : float
            Dispersion parameter.

        Returns
        -------
        torch.Tensor
            Scaled deviance (scalar).
        """
        return self.unit_deviance(y, mu).sum() / dispersion

    def nll_loss(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
        dispersion: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """Negative log-likelihood loss (for training).

        Returns mean NLL across observations.

        Parameters
        ----------
        y : torch.Tensor
            Observed values.
        mu : torch.Tensor
            Predicted means.
        dispersion : torch.Tensor or float
            Dispersion parameter.

        Returns
        -------
        torch.Tensor
            Mean negative log-likelihood (scalar).
        """
        return -self.log_likelihood(y, mu, dispersion).mean()

    def link_derivative(self, mu: torch.Tensor) -> torch.Tensor:
        """Derivative of link function: g'(mu).

        Used in working weights for IRLS and Fisher information.

        Parameters
        ----------
        mu : torch.Tensor
            Mean values.

        Returns
        -------
        torch.Tensor
            Derivative values.
        """
        raise NotImplementedError("Subclass must implement link_derivative")

    def weight_matrix(
        self,
        mu: torch.Tensor,
        dispersion: float = 1.0,
    ) -> torch.Tensor:
        """Working weights W_ii for observation i.

        W_ii = 1 / (V(mu_i) * g'(mu_i)^2 * phi)

        This is used in the "meat" of the sandwich estimator.

        Parameters
        ----------
        mu : torch.Tensor
            Mean values.
        dispersion : float
            Dispersion parameter.

        Returns
        -------
        torch.Tensor
            Working weights.
        """
        g_prime = self.link_derivative(mu)
        V = self.variance(mu)
        return 1.0 / (V * g_prime**2 * dispersion)

    def estimate_dispersion(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
        df_resid: int,
    ) -> float:
        """Estimate dispersion parameter from Pearson residuals.

        phi_hat = sum((y - mu)^2 / V(mu)) / df_resid

        Parameters
        ----------
        y : torch.Tensor
            Observed values.
        mu : torch.Tensor
            Predicted means.
        df_resid : int
            Residual degrees of freedom.

        Returns
        -------
        float
            Estimated dispersion.
        """
        V = self.variance(mu)
        pearson_resid_sq = (y - mu) ** 2 / V
        return float(pearson_resid_sq.sum() / df_resid)

    def validate_response(self, y: torch.Tensor) -> None:
        """Validate response variable is in valid range for family.

        Parameters
        ----------
        y : torch.Tensor
            Response values to validate.

        Raises
        ------
        ValueError
            If response is invalid for this family.
        """
        if torch.isnan(y).any():
            raise ValueError("Response contains NaN values")
