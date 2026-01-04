"""Family-specific influence function implementations.

Each family defines: loss(), residual(), weight(), influence_score()
"""

import math
from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.stats import norm


# =============================================================================
# Base Family
# =============================================================================

class BaseFamily(ABC):
    """Abstract base for influence function families."""
    name: str = "base"

    @abstractmethod
    def loss(self, y: torch.Tensor, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Per-sample loss L(Y, T, theta)."""
        pass

    @abstractmethod
    def residual(self, y: torch.Tensor, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Model residual r_i."""
        pass

    @abstractmethod
    def weight(self, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Hessian weight W_i."""
        pass

    def h_value(self, theta: torch.Tensor) -> torch.Tensor:
        """Target functional H(theta) = beta."""
        return theta[:, 1]

    def h_gradient(self) -> torch.Tensor:
        """Gradient of H w.r.t. theta = [0, 1]."""
        return torch.tensor([0.0, 1.0])

    def compute_hessian(self, theta: torch.Tensor, t: torch.Tensor, t_mean: torch.Tensor) -> torch.Tensor:
        """Compute Lambda = E[W * T_tilde @ T_tilde^T]."""
        n = len(t)
        t_tilde = t - t_mean
        T_design = torch.stack([torch.ones(n, device=t.device), t_tilde], dim=1)
        W = self.weight(t, theta)
        Lambda = (T_design.T @ (W.unsqueeze(1) * T_design)) / n
        return Lambda + 1e-4 * torch.eye(2, device=t.device)

    def influence_score(self, y, t, theta, t_mean, t_var, lambda_inv) -> torch.Tensor:
        """Default influence score: psi_i = beta_i - l_theta @ Lambda^{-1} @ H_grad."""
        n = len(y)
        beta_i = theta[:, 1]
        r_i = self.residual(y, t, theta)
        t_tilde = t - t_mean
        T_design = torch.stack([torch.ones(n, device=t.device), t_tilde], dim=1)
        H_grad = self.h_gradient().to(theta.device)
        l_theta = -r_i.unsqueeze(1) * T_design
        correction = l_theta @ (lambda_inv @ H_grad)
        return beta_i - correction


# =============================================================================
# Linear Family
# =============================================================================

class LinearFamily(BaseFamily):
    """Y = alpha + beta*T + epsilon. Loss: MSE. Weight: 1."""
    name = "linear"

    def loss(self, y, t, theta):
        mu = theta[:, 0] + theta[:, 1] * t
        return (y - mu) ** 2

    def residual(self, y, t, theta):
        mu = theta[:, 0] + theta[:, 1] * t
        return y - mu

    def weight(self, t, theta):
        return torch.ones_like(t)

    def influence_score(self, y, t, theta, t_mean, t_var, lambda_inv):
        """Simplified: psi_i = beta_i + r_i * (T - E[T]) / Var(T)."""
        beta_i = theta[:, 1]
        r_i = self.residual(y, t, theta)
        t_centered = t - t_mean
        global_var = torch.var(t) + 1e-4
        return beta_i + (r_i * t_centered) / global_var


# =============================================================================
# Gamma Family
# =============================================================================

class GammaFamily(BaseFamily):
    """Y ~ Gamma, mu = exp(alpha + beta*T). Loss: deviance. Weight: 1."""
    name = "gamma"

    def loss(self, y, t, theta):
        mu = torch.exp(torch.clamp(theta[:, 0] + theta[:, 1] * t, -10, 10))
        return y / mu + torch.log(mu)

    def residual(self, y, t, theta):
        mu = torch.exp(torch.clamp(theta[:, 0] + theta[:, 1] * t, -10, 10))
        return (y - mu) / torch.clamp(mu, min=1e-6)

    def weight(self, t, theta):
        return torch.ones_like(t)

    def influence_score(self, y, t, theta, t_mean, t_var, lambda_inv):
        beta_i = theta[:, 1]
        r_i = self.residual(y, t, theta)
        t_centered = t - t_mean
        global_var = torch.var(t) + 1e-4
        return beta_i + (r_i * t_centered) / global_var


# =============================================================================
# Gumbel Family
# =============================================================================

class GumbelFamily(BaseFamily):
    """Y ~ Gumbel(mu, scale), mu = alpha + beta*T. Weight: 1."""
    name = "gumbel"

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def loss(self, y, t, theta):
        mu = theta[:, 0] + theta[:, 1] * t
        z = (y - mu) / self.scale
        return z + torch.exp(-z)

    def residual(self, y, t, theta):
        mu = theta[:, 0] + theta[:, 1] * t
        z = (y - mu) / self.scale
        return 1 - torch.exp(-z)

    def weight(self, t, theta):
        return torch.ones_like(t)

    def influence_score(self, y, t, theta, t_mean, t_var, lambda_inv):
        beta_i = theta[:, 1]
        r_i = self.residual(y, t, theta)
        t_centered = t - t_mean
        global_var = torch.var(t) + 1e-4
        return beta_i + (r_i * t_centered) / global_var


# =============================================================================
# Poisson Family
# =============================================================================

class PoissonFamily(BaseFamily):
    """Y ~ Poisson(lambda), lambda = exp(alpha + beta*T). Weight: lambda."""
    name = "poisson"

    def loss(self, y, t, theta):
        lam = torch.exp(torch.clamp(theta[:, 0] + theta[:, 1] * t, -20, 20))
        return lam - y * torch.log(lam + 1e-10)

    def residual(self, y, t, theta):
        lam = torch.exp(torch.clamp(theta[:, 0] + theta[:, 1] * t, -20, 20))
        return y - lam

    def weight(self, t, theta):
        return torch.exp(torch.clamp(theta[:, 0] + theta[:, 1] * t, -20, 20))


# =============================================================================
# Logit Family
# =============================================================================

class LogitFamily(BaseFamily):
    """Y ~ Bernoulli(p), p = sigmoid(alpha + beta*T). Weight: p(1-p)."""
    name = "logit"

    def loss(self, y, t, theta):
        logits = theta[:, 0] + theta[:, 1] * t
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none')

    def residual(self, y, t, theta):
        p = torch.sigmoid(theta[:, 0] + theta[:, 1] * t)
        return y - p

    def weight(self, t, theta):
        p = torch.sigmoid(theta[:, 0] + theta[:, 1] * t)
        p = torch.clamp(p, 1e-6, 1 - 1e-6)
        return p * (1 - p)


# =============================================================================
# Tobit Family
# =============================================================================

class TobitFamily(BaseFamily):
    """Y = max(0, Y*), Y* = alpha + beta*T + epsilon. Censored model.

    Implementation Notes:
    - Hessian weight is simplified to P(uncensored) = 1 - Φ(-μ/σ)
    - The full censored likelihood has a block-structured Hessian, but this
      approximation works well when censoring is moderate (<50% censored)
    - Custom influence_score() accounts for censoring probability in the
      bias correction term by dividing by P(uncensored)
    - Residual uses Mills ratio for censored obs, standardized error for uncensored
    """
    name = "tobit"

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def loss(self, y, t, theta):
        mu = theta[:, 0] + theta[:, 1] * t
        censored = (y <= 0)
        z = -mu / self.sigma
        cdf_vals = norm.cdf(z.detach().numpy())
        nll_censored = -torch.log(torch.tensor(cdf_vals, dtype=torch.float32) + 1e-10)
        nll_uncensored = 0.5 * ((y - mu) / self.sigma) ** 2 + np.log(self.sigma)
        return torch.where(censored, nll_censored, nll_uncensored)

    def residual(self, y, t, theta):
        mu = theta[:, 0] + theta[:, 1] * t
        censored = (y <= 0)
        r_uncensored = (y - mu) / self.sigma
        z = (-mu / self.sigma).detach().numpy()
        mills = torch.tensor(-norm.pdf(z) / (norm.cdf(z) + 1e-10), dtype=torch.float32)
        return torch.where(censored, mills, r_uncensored)

    def weight(self, t, theta):
        mu = theta[:, 0] + theta[:, 1] * t
        z = (-mu / self.sigma).detach().numpy()
        return torch.tensor(1 - norm.cdf(z), dtype=torch.float32)

    def influence_score(self, y, t, theta, t_mean, t_var, lambda_inv):
        n = len(y)
        beta_i = theta[:, 1]
        r_i = self.residual(y, t, theta)
        t_tilde = t - t_mean
        T_design = torch.stack([torch.ones(n), t_tilde], dim=1)
        H_grad = self.h_gradient()
        l_theta = -r_i.unsqueeze(1) * T_design
        p_uncensored = self.weight(t, theta)
        correction = l_theta @ (lambda_inv @ H_grad)
        return beta_i - correction / torch.clamp(p_uncensored, min=0.1)


# =============================================================================
# Negative Binomial Family
# =============================================================================

class NegBinFamily(BaseFamily):
    """Y ~ NegBin, mu = exp(alpha + beta*T). Weight: mu/(1+alpha*mu)."""
    name = "negbin"

    def __init__(self, overdispersion: float = 0.5):
        self.overdispersion = overdispersion

    def loss(self, y, t, theta):
        mu = torch.exp(torch.clamp(theta[:, 0] + theta[:, 1] * t, -20, 20))
        return mu - y * torch.log(mu + 1e-10)

    def residual(self, y, t, theta):
        """Score residual: (Y - μ) / (1 + αμ).

        Note: This is the score residual derived from the NegBin log-likelihood
        gradient, NOT the Pearson residual (Y - μ)/√var. The influence function
        requires the score residual for correct bias correction.
        """
        mu = torch.exp(torch.clamp(theta[:, 0] + theta[:, 1] * t, -20, 20))
        return (y - mu) / (1.0 + self.overdispersion * mu)

    def weight(self, t, theta):
        mu = torch.exp(torch.clamp(theta[:, 0] + theta[:, 1] * t, -20, 20))
        return mu / (1.0 + self.overdispersion * mu)


# =============================================================================
# Weibull Family
# =============================================================================

class WeibullFamily(BaseFamily):
    """Y ~ Weibull(k, lambda), lambda = exp(alpha + beta*T). Weight: k^2."""
    name = "weibull"

    def __init__(self, shape: float = 2.0):
        self.shape = shape

    def loss(self, y, t, theta):
        lam = torch.exp(torch.clamp(theta[:, 0] + theta[:, 1] * t, -20, 20))
        k = self.shape
        z = (y / lam) ** k
        return -math.log(k) + k * torch.log(lam) - (k - 1) * torch.log(y + 1e-10) + z

    def residual(self, y, t, theta):
        lam = torch.exp(torch.clamp(theta[:, 0] + theta[:, 1] * t, -20, 20))
        return (y / lam) ** self.shape - 1

    def weight(self, t, theta):
        return torch.ones_like(t) * (self.shape ** 2)


# =============================================================================
# Factory
# =============================================================================

FAMILIES = {
    "linear": LinearFamily,
    "gamma": GammaFamily,
    "gumbel": GumbelFamily,
    "poisson": PoissonFamily,
    "logit": LogitFamily,
    "tobit": TobitFamily,
    "negbin": NegBinFamily,
    "weibull": WeibullFamily,
}


def get_family(name: str, **kwargs) -> BaseFamily:
    """Factory function for families."""
    if name not in FAMILIES:
        raise ValueError(f"Unknown family: {name}. Available: {list(FAMILIES.keys())}")
    return FAMILIES[name](**kwargs) if kwargs else FAMILIES[name]()
