"""
Average Marginal Effect (AME) target.

For logit model: H = p(1-p) * β where p = σ(α + β*t̃)

This captures the marginal effect of treatment on the outcome probability,
averaged across the population.

The Jacobian is model-specific and computed via autodiff by default.
"""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseTarget


class AverageMarginalEffect(BaseTarget):
    """
    Target: Average Marginal Effect (AME).

    For logit: H = p(1-p) * β_k where p = σ(θ'T)

    This is the marginal effect of treatment on the outcome probability,
    evaluated at t_tilde.

    AME = E[∂P(Y=1)/∂T | X] = E[p(1-p) * β | X]
    """

    output_dim: int = 1

    def __init__(
        self,
        param_index: int = 1,
        model_type: str = "logit",
    ):
        """
        Initialize AME target.

        Args:
            param_index: Index of treatment parameter (default: 1 for β)
            model_type: Type of model ("logit", "probit", "linear")
        """
        self.param_index = param_index
        self.model_type = model_type

    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """
        Compute AME at evaluation point.

        For logit: H = p(1-p) * β where p = σ(α + β*t̃)

        Args:
            x: Covariates (not used directly)
            theta: Parameters (d_theta,) = [α, β, ...]
            t_tilde: Evaluation point (scalar)

        Returns:
            Scalar: marginal effect at t_tilde
        """
        if self.model_type == "logit":
            # p = σ(α + β*t)
            logits = theta[0] + theta[self.param_index] * t_tilde
            p = torch.sigmoid(logits)
            # AME = p(1-p) * β
            return p * (1 - p) * theta[self.param_index]

        elif self.model_type == "linear":
            # Linear: marginal effect is just β
            return theta[self.param_index]

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def jacobian(
        self, x: Tensor, theta: Tensor, t_tilde: Tensor
    ) -> Optional[Tensor]:
        """
        Compute Jacobian of AME w.r.t. theta.

        For logit with θ = (α, β):
            ∂H/∂α = β * p(1-p)(1-2p)
            ∂H/∂β = p(1-p) + β * t̃ * p(1-p)(1-2p)

        Returns None by default to use autodiff.

        Args:
            x: Covariates
            theta: Parameters
            t_tilde: Evaluation point

        Returns:
            (d_theta,) Jacobian or None for autodiff
        """
        if self.model_type == "logit":
            # Compute closed-form Jacobian for logit
            logits = theta[0] + theta[self.param_index] * t_tilde
            p = torch.sigmoid(logits)
            pp = p * (1 - p)  # p(1-p)
            pp_prime = pp * (1 - 2 * p)  # p(1-p)(1-2p)
            beta = theta[self.param_index]

            d_theta = theta.shape[0]
            jac = torch.zeros(d_theta, dtype=theta.dtype, device=theta.device)

            # ∂H/∂α = β * p(1-p)(1-2p)
            jac[0] = beta * pp_prime

            # ∂H/∂β = p(1-p) + β * t̃ * p(1-p)(1-2p)
            jac[self.param_index] = pp + beta * t_tilde * pp_prime

            return jac

        elif self.model_type == "linear":
            # Linear: Jacobian is just e_k
            d_theta = theta.shape[0]
            jac = torch.zeros(d_theta, dtype=theta.dtype, device=theta.device)
            jac[self.param_index] = 1.0
            return jac

        else:
            # Fall back to autodiff
            return None


# Convenience alias
AME = AverageMarginalEffect
