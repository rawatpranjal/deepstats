"""Logit family implementation with closed-form optimizations."""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .base import BaseFamily


class LogitFamily(BaseFamily):
    """
    Logit (binary choice) structural model.

    Model: P[Y = 1 | T, X] = sigmoid(alpha(X) + beta(X) * T)

    Loss: Binary cross-entropy

    Parameters:
        theta = (alpha, beta) where
        - alpha(x): baseline log-odds
        - beta(x): treatment effect on log-odds

    Target: E[beta(X)] (average treatment effect on log-odds)

    Note: The Hessian depends on theta through the weight p(1-p),
    so three-way splitting is required.
    """

    theta_dim = 2

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Binary cross-entropy loss.

        Args:
            y: (n,) binary outcomes {0, 1}
            t: (n,) treatments
            theta: (n, 2) parameters [alpha, beta]

        Returns:
            (n,) per-observation losses
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]
        logits = alpha + beta * t

        # Use numerically stable BCE
        return F.binary_cross_entropy_with_logits(logits, y, reduction='none')

    def gradient(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Closed-form gradient.

        l_theta = -(y - p) * (1, t)'

        where p = sigmoid(alpha + beta*t)

        Args:
            y: (n,) outcomes
            t: (n,) treatments
            theta: (n, 2) parameters [alpha, beta]

        Returns:
            (n, 2) gradient tensor
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]
        p = torch.sigmoid(alpha + beta * t)

        # Gradient of BCE is (p - y) for logit, or -(y - p)
        residual = p - y

        grad_alpha = residual
        grad_beta = residual * t

        return torch.stack([grad_alpha, grad_beta], dim=1)

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Closed-form Hessian.

        l_theta_theta = p(1-p) * [[1, t], [t, t^2]]

        where p = sigmoid(alpha + beta*t)

        Note: The Hessian DEPENDS on theta through p(1-p).
        This means three-way splitting is required.

        Args:
            y: (n,) outcomes (unused in Hessian)
            t: (n,) treatments
            theta: (n, 2) parameters [alpha, beta]

        Returns:
            (n, 2, 2) Hessian tensor
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]
        p = torch.sigmoid(alpha + beta * t)

        # Numerical stability: clamp p away from 0 and 1
        p = torch.clamp(p, 1e-6, 1 - 1e-6)

        # Weight for Hessian
        w = p * (1 - p)

        n = len(y)
        H = torch.zeros(n, 2, 2, dtype=theta.dtype, device=theta.device)
        H[:, 0, 0] = w
        H[:, 0, 1] = w * t
        H[:, 1, 0] = w * t
        H[:, 1, 1] = w * t ** 2

        return H

    def hessian_depends_on_theta(self) -> bool:
        """
        Logit Hessian depends on theta through p(1-p).

        This means three-way splitting is required.
        """
        return True

    def residual(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Compute residual: y - p.

        Args:
            y: (n,) outcomes
            t: (n,) treatments
            theta: (n, 2) parameters [alpha, beta]

        Returns:
            (n,) residuals
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]
        p = torch.sigmoid(alpha + beta * t)
        return y - p

    def predicted_probability(self, t: Tensor, theta: Tensor) -> Tensor:
        """
        Compute predicted probability P[Y=1].

        Args:
            t: (n,) treatments
            theta: (n, 2) parameters [alpha, beta]

        Returns:
            (n,) predicted probabilities
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]
        return torch.sigmoid(alpha + beta * t)
