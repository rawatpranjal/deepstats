"""
Logistic regression model for structural estimation.

P(Y=1|T,X) = σ(α(X) + β(X) * T)

Loss: -y*log(p) - (1-y)*log(1-p) where p = σ(α + β*t)
Score: (p - y) * [1, t]
Hessian: p(1-p) * [[1, t], [t, t²]] (DEPENDS on θ through p!)

This is Regime A (randomized) or C (observational):
- Randomized: Λ can be COMPUTED via Monte Carlo (no Y dependence)
- Observational: Λ must be ESTIMATED with 3-way split
"""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseModel


class LogitModel(BaseModel):
    """
    Binary logistic regression model.

    Model: P(Y=1|T,X) = σ(α(X) + β(X) * T)

    Key property: Hessian depends on theta (through σ), NOT on Y.
    - Randomized: Regime A (compute Λ via ∫ p(1-p) tt' dF_T)
    - Observational: Regime C (estimate Λ with 3-way split)
    """

    theta_dim: int = 2
    hessian_depends_on_theta: bool = True  # Depends on p = σ(θ'T)
    hessian_depends_on_y: bool = False  # KEY: Enables Regime A under randomization

    def __init__(self, eps: float = 1e-7):
        """
        Initialize logit model.

        Args:
            eps: Numerical stability constant for log
        """
        self.eps = eps

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Binary cross-entropy loss (single observation).

        Args:
            y: Outcome (0 or 1)
            t: Treatment (scalar)
            theta: Parameters (2,) = [α, β]

        Returns:
            Scalar loss: -y*log(p) - (1-y)*log(1-p)
        """
        logits = theta[0] + theta[1] * t
        p = torch.sigmoid(logits)
        p = torch.clamp(p, self.eps, 1 - self.eps)
        return -y * torch.log(p) - (1 - y) * torch.log(1 - p)

    def score(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Closed-form score (gradient of loss).

        ∂ℓ/∂θ = (p - y) * [1, t]

        Args:
            y: Outcome (0 or 1)
            t: Treatment (scalar)
            theta: Parameters (2,)

        Returns:
            (2,) gradient vector
        """
        logits = theta[0] + theta[1] * t
        p = torch.sigmoid(logits)
        residual = p - y
        return torch.stack([residual, residual * t])

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Closed-form Hessian (depends on theta, NOT on y).

        ∂²ℓ/∂θ² = p(1-p) * [[1, t], [t, t²]]

        Args:
            y: Outcome - NOT USED (Hessian is Fisher information)
            t: Treatment (scalar)
            theta: Parameters (2,)

        Returns:
            (2, 2) Hessian matrix
        """
        logits = theta[0] + theta[1] * t
        p = torch.sigmoid(logits)
        w = p * (1 - p)  # Weight = variance of Bernoulli

        H = torch.zeros(2, 2, dtype=theta.dtype, device=theta.device)
        H[0, 0] = w
        H[0, 1] = w * t
        H[1, 0] = w * t
        H[1, 1] = w * t * t
        return H

    def compute_lambda_integral(
        self,
        theta: Tensor,
        t_samples: Tensor,
    ) -> Tensor:
        """
        Compute Λ(x) via Monte Carlo for randomized experiments.

        Λ(x) = ∫ p(1-p) tt' dF_T(t)
             ≈ (1/M) Σ_m p(θ,t_m)(1-p(θ,t_m)) t_m t_m'

        This is used in Regime A (randomized + hessian_depends_on_y=False).

        Args:
            theta: (d_theta,) parameter vector for this x
            t_samples: (M,) or (M, d_t) samples from F_T

        Returns:
            (d_theta, d_theta) Lambda matrix
        """
        M = t_samples.shape[0]
        d_theta = theta.shape[0]

        Lambda = torch.zeros(d_theta, d_theta, dtype=theta.dtype, device=theta.device)

        for m in range(M):
            t_m = t_samples[m]
            # Compute weight p(1-p) at this t
            logits = theta[0] + theta[1] * t_m
            p = torch.sigmoid(logits)
            w = p * (1 - p)

            # Add contribution: w * tt'
            t_vec = torch.stack([torch.ones_like(t_m), t_m])
            Lambda += w * torch.outer(t_vec, t_vec)

        return Lambda / M


# Convenience alias
Logit = LogitModel
