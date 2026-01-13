"""
Average parameter target: H = E[θ_k(X)]

This is the simplest target - just average a parameter across observations.
Common use case: Average Treatment Effect (β coefficient).

Jacobian is trivial: ∂H/∂θ = e_k (unit vector with 1 in position k)
"""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseTarget


class AverageParameter(BaseTarget):
    """
    Target: H = θ_k (the k-th parameter).

    For treatment effect estimation:
    - θ = (α, β) where α is intercept, β is treatment effect
    - H = β means k=1 (0-indexed)

    This target has a trivial Jacobian: [0, ..., 0, 1, 0, ..., 0]
    """

    output_dim: int = 1

    def __init__(self, param_index: int = 1, theta_dim: int = 2):
        """
        Initialize average parameter target.

        Args:
            param_index: Index of parameter to target (0-indexed)
            theta_dim: Total dimension of theta
        """
        self.param_index = param_index
        self.theta_dim = theta_dim

    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """
        Return the k-th parameter.

        Args:
            x: Covariates (not used)
            theta: Parameters (d_theta,)
            t_tilde: Evaluation point (not used)

        Returns:
            Scalar: θ_k
        """
        return theta[self.param_index]

    def jacobian(
        self, x: Tensor, theta: Tensor, t_tilde: Tensor
    ) -> Tensor:
        """
        Return unit vector e_k.

        ∂H/∂θ = e_k where e_k has 1 in position k, 0 elsewhere.

        Args:
            x: Covariates (not used)
            theta: Parameters (d_theta,)
            t_tilde: Evaluation point (not used)

        Returns:
            (d_theta,) unit vector
        """
        d_theta = theta.shape[0]
        jac = torch.zeros(d_theta, dtype=theta.dtype, device=theta.device)
        jac[self.param_index] = 1.0
        return jac


class AverageBeta(AverageParameter):
    """
    Convenience alias for average treatment effect target.

    For standard 2-parameter models (α, β), this targets β.
    """

    def __init__(self, theta_dim: int = 2):
        """
        Initialize average beta target.

        Args:
            theta_dim: Total dimension of theta (default 2)
        """
        super().__init__(param_index=1, theta_dim=theta_dim)


# Common alias
ATE = AverageBeta  # Average Treatment Effect
