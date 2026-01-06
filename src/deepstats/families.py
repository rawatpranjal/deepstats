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
    n_params: int = 2  # Number of structural parameters [α, β] by default

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

    def h_value(self, theta: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """Target functional H(theta). Override for T-dependent targets like AME."""
        return theta[:, 1]

    def h_gradient(self, theta: torch.Tensor = None, t: torch.Tensor = None) -> torch.Tensor:
        """Gradient of H w.r.t. theta = [0, 1]. Override for T-dependent targets."""
        return torch.tensor([0.0, 1.0])

    def compute_hessian(self, theta: torch.Tensor, t: torch.Tensor, t_mean: torch.Tensor) -> torch.Tensor:
        """Compute Lambda = E[W * T_tilde @ T_tilde^T].

        Uses ridge regularization for numerical stability.
        """
        n = len(t)
        t_tilde = t - t_mean
        T_design = torch.stack([torch.ones(n, device=t.device), t_tilde], dim=1)
        W = self.weight(t, theta)
        Lambda = (T_design.T @ (W.unsqueeze(1) * T_design)) / n
        # Small ridge for numerical stability
        return Lambda + 1e-4 * torch.eye(2, device=t.device)

    def compute_hessian_with_diagnostics(
        self, theta: torch.Tensor, t: torch.Tensor, t_mean: torch.Tensor
    ) -> tuple:
        """Compute Hessian with stability diagnostics (Phase 3).

        Returns:
            Lambda: The Hessian matrix
            min_eig: Minimum eigenvalue (should be > 1e-4 for stability)
            condition: Condition number (lower is better)
        """
        Lambda = self.compute_hessian(theta, t, t_mean)
        eigenvalues = torch.linalg.eigvalsh(Lambda)
        min_eig = eigenvalues.min().item()
        max_eig = eigenvalues.max().item()
        condition = max_eig / (min_eig + 1e-10)
        return Lambda, min_eig, condition

    def influence_score(
        self, y, t, theta, t_mean, t_var, lambda_inv, return_correction: bool = False
    ):
        """Default influence score: psi_i = H(theta_i) - l_theta @ Lambda^{-1} @ H_grad.

        Args:
            return_correction: If True, return (psi, correction) tuple for Phase 3 diagnostics
        """
        n = len(y)
        h_i = self.h_value(theta, t)  # Target value (beta for default, p(1-p)beta for AME)
        r_i = self.residual(y, t, theta)
        t_tilde = t - t_mean
        T_design = torch.stack([torch.ones(n, device=t.device), t_tilde], dim=1)
        H_grad = self.h_gradient(theta, t).to(theta.device)
        l_theta = -r_i.unsqueeze(1) * T_design
        correction = l_theta @ (lambda_inv @ H_grad)
        psi = h_i - correction
        if return_correction:
            return psi, correction
        return psi


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

    def influence_score(
        self, y, t, theta, t_mean, t_var, lambda_inv, return_correction: bool = False
    ):
        """Full Hessian-based influence score.

        psi_i = H(theta_i) + nabla_H' @ Lambda^{-1} @ nabla_ell_i

        Where:
        - H(theta) = beta (target functional)
        - nabla_H = [0, 1] (gradient of H w.r.t. theta)
        - nabla_ell = -r_i * [1, T_tilde] (score/gradient of loss)
        - Lambda = E[W * T_design @ T_design'] (Hessian)

        Args:
            return_correction: If True, return (psi, correction) tuple for Phase 3 diagnostics
        """
        n = len(y)
        beta_i = theta[:, 1]
        r_i = self.residual(y, t, theta)
        t_tilde = t - t_mean

        # Design matrix [1, T-E[T]]
        T_design = torch.stack([torch.ones(n, device=t.device), t_tilde], dim=1)

        # Gradient of H w.r.t. theta = [0, 1]
        H_grad = self.h_gradient(theta, t).to(theta.device)

        # Score (gradient of loss): l_theta = -r_i * [1, T_tilde]
        l_theta = -r_i.unsqueeze(1) * T_design

        # Correction term: l_theta @ Lambda^{-1} @ H_grad
        correction = l_theta @ (lambda_inv @ H_grad)

        # Target value: H(theta) = beta for linear
        h_i = self.h_value(theta, t)
        psi = h_i - correction
        if return_correction:
            return psi, correction
        return psi


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

    def influence_score(
        self, y, t, theta, t_mean, t_var, lambda_inv, return_correction: bool = False
    ):
        beta_i = theta[:, 1]
        r_i = self.residual(y, t, theta)
        t_centered = t - t_mean
        global_var = torch.var(t) + 1e-4
        correction = (r_i * t_centered) / global_var
        psi = beta_i + correction  # Note: simplified formula uses + not -
        if return_correction:
            return psi, -correction  # Return as correction term (negated for consistency)
        return psi


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

    def influence_score(
        self, y, t, theta, t_mean, t_var, lambda_inv, return_correction: bool = False
    ):
        beta_i = theta[:, 1]
        r_i = self.residual(y, t, theta)
        t_centered = t - t_mean
        global_var = torch.var(t) + 1e-4
        correction = (r_i * t_centered) / global_var
        psi = beta_i + correction
        if return_correction:
            return psi, -correction
        return psi


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
    """Y ~ Bernoulli(p), p = sigmoid(alpha + beta*T). Weight: p(1-p).

    Supports two targets:
    - 'beta': E[β(X)] - average log-odds ratio (default)
    - 'ame': E[p(1-p)β(X)] - average marginal effect on probability
    """
    name = "logit"

    def __init__(self, target: str = "beta"):
        """Initialize LogitFamily with target selection.

        Args:
            target: 'beta' (log-odds ratio) or 'ame' (average marginal effect)
        """
        if target not in ("beta", "ame"):
            raise ValueError(f"target must be 'beta' or 'ame', got '{target}'")
        self.target = target

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

    def h_value(self, theta: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """Target functional H(theta).

        For beta target: H(θ) = β
        For AME target: H(θ) = p(1-p)β where p = σ(α + βT)
        """
        if self.target == "ame" and t is not None:
            p = torch.sigmoid(theta[:, 0] + theta[:, 1] * t)
            p = torch.clamp(p, 1e-6, 1 - 1e-6)
            return p * (1 - p) * theta[:, 1]
        return theta[:, 1]

    def h_gradient(self, theta: torch.Tensor = None, t: torch.Tensor = None) -> torch.Tensor:
        """Gradient of H w.r.t. theta.

        For beta target: ∇H = [0, 1]
        For AME target: ∇H = [β·w·(1-2p), w + β·w·(1-2p)·T] where w = p(1-p)
        """
        if self.target == "ame" and theta is not None and t is not None:
            p = torch.sigmoid(theta[:, 0] + theta[:, 1] * t)
            p = torch.clamp(p, 1e-6, 1 - 1e-6)
            w = p * (1 - p)
            beta = theta[:, 1]
            dw_dz = w * (1 - 2 * p)  # derivative of w w.r.t. z = α + βT
            # ∂(wβ)/∂α = β · ∂w/∂z · 1 = β · w · (1-2p)
            # ∂(wβ)/∂β = w + β · ∂w/∂z · T = w + β · w · (1-2p) · T
            grad_alpha = (beta * dw_dz).mean()
            grad_beta = (w + beta * dw_dz * t).mean()
            return torch.tensor([grad_alpha.item(), grad_beta.item()])
        return torch.tensor([0.0, 1.0])


# =============================================================================
# Tobit Family
# =============================================================================

class TobitFamily(BaseFamily):
    """Tobit model: Y = max(0, α + βT + σε), ε ~ N(0,1).

    Network outputs θ = [α, β, γ] where σ = exp(γ).

    Targets:
    - 'latent': E[β] - effect on latent Y*
    - 'observed': E[β·Φ(z)] - average effect on observed outcome

    Mathematical Foundation:
    - z = μ/σ = (α + βT)/σ
    - For latent: H = β, ∇H = [0, 1, 0]
    - For observed: H = β·Φ(z), ∇H = [β·φ(z)/σ, Φ(z)+βT·φ(z)/σ, -βz·φ(z)]

    Score Vector (NLL minimization):
    Uncensored (y > 0):
        ∂ℓ/∂α = (μ-y)/σ², ∂ℓ/∂β = T·(μ-y)/σ², ∂ℓ/∂γ = 1 - e²
    Censored (y = 0):
        ∂ℓ/∂α = λ/σ, ∂ℓ/∂β = T·λ/σ, ∂ℓ/∂γ = -z·λ
    where λ = φ(z)/(1-Φ(z)) is the Mills ratio.
    """
    name = "tobit"
    n_params = 3  # [α, β, γ=log(σ)]

    def __init__(self, target: str = "latent"):
        """Initialize TobitFamily with target selection.

        Args:
            target: 'latent' (E[β]) or 'observed' (E[β·Φ(z)])
        """
        if target not in ("latent", "observed"):
            raise ValueError(f"target must be 'latent' or 'observed', got '{target}'")
        self.target = target

    def loss(self, y, t, theta):
        """Tobit NLL loss (autograd-compatible via torch.distributions)."""
        alpha, beta, gamma = theta[:, 0], theta[:, 1], theta[:, 2]
        gamma = torch.clamp(gamma, -10, 10)
        sigma = torch.exp(gamma)
        mu = alpha + beta * t
        z = mu / sigma

        censored = (y <= 0)
        dist = torch.distributions.Normal(0, 1)

        # Uncensored: log(σ) + ½((y-μ)/σ)²
        nll_uncensored = gamma + 0.5 * ((y - mu) / sigma) ** 2

        # Censored: -log(Φ(-z)) = -log(1 - Φ(z))
        log_Phi_neg_z = dist.cdf(-z).clamp(min=1e-10).log()
        nll_censored = -log_Phi_neg_z

        return torch.where(censored, nll_censored, nll_uncensored)

    def residual(self, y, t, theta):
        """Residual: standardized error (uncensored) or negative Mills ratio (censored)."""
        alpha, beta, gamma = theta[:, 0], theta[:, 1], theta[:, 2]
        sigma = torch.exp(torch.clamp(gamma, -10, 10))
        mu = alpha + beta * t
        z = mu / sigma
        censored = (y <= 0)

        dist = torch.distributions.Normal(0, 1)
        phi_neg_z = torch.exp(dist.log_prob(-z))
        Phi_neg_z = dist.cdf(-z).clamp(min=1e-10)
        mills = -phi_neg_z / Phi_neg_z  # Negative Mills ratio for censored

        r_uncensored = (y - mu) / sigma
        return torch.where(censored, mills, r_uncensored)

    def weight(self, t, theta):
        """Hessian weight: P(uncensored) = Φ(z) = Φ(μ/σ)."""
        alpha, beta, gamma = theta[:, 0], theta[:, 1], theta[:, 2]
        sigma = torch.exp(torch.clamp(gamma, -10, 10))
        mu = alpha + beta * t
        z = mu / sigma
        return torch.distributions.Normal(0, 1).cdf(z).clamp(min=0.01, max=0.99)

    def h_value(self, theta: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """Target functional H(θ).

        For latent: H = β
        For observed: H = β·Φ(z) where z = (α + βT)/σ
        """
        if self.target == "observed" and t is not None:
            alpha, beta, gamma = theta[:, 0], theta[:, 1], theta[:, 2]
            sigma = torch.exp(torch.clamp(gamma, -10, 10))
            z = (alpha + beta * t) / sigma
            Phi_z = torch.distributions.Normal(0, 1).cdf(z)
            return beta * Phi_z
        return theta[:, 1]  # latent: just β

    def h_gradient(self, theta: torch.Tensor = None, t: torch.Tensor = None) -> torch.Tensor:
        """Gradient of H w.r.t. θ = [α, β, γ].

        For latent: ∇H = [0, 1, 0]
        For observed: ∇H = [β·φ(z)/σ, Φ(z) + βT·φ(z)/σ, -βz·φ(z)]
        """
        if self.target == "observed" and theta is not None and t is not None:
            alpha, beta, gamma = theta[:, 0], theta[:, 1], theta[:, 2]
            sigma = torch.exp(torch.clamp(gamma, -10, 10))
            z = (alpha + beta * t) / sigma
            dist = torch.distributions.Normal(0, 1)
            phi_z = torch.exp(dist.log_prob(z))
            Phi_z = dist.cdf(z)

            # ∂H/∂α = β·φ(z)/σ
            grad_alpha = (beta * phi_z / sigma).mean()
            # ∂H/∂β = Φ(z) + β·T·φ(z)/σ
            grad_beta = (Phi_z + beta * t * phi_z / sigma).mean()
            # ∂H/∂γ = -β·z·φ(z) (since ∂σ/∂γ = σ)
            grad_gamma = (-beta * z * phi_z).mean()

            return torch.tensor([grad_alpha.item(), grad_beta.item(), grad_gamma.item()])
        return torch.tensor([0.0, 1.0, 0.0])  # latent: ∇H = [0, 1, 0]

    def compute_hessian(self, theta: torch.Tensor, t: torch.Tensor, t_mean: torch.Tensor) -> torch.Tensor:
        """Compute 3×3 Hessian Λ for Tobit model.

        Uses Fisher information approximation with proper scaling.
        """
        n = len(t)
        sigma = torch.exp(torch.clamp(theta[:, 2], -10, 10))
        t_tilde = t - t_mean

        # Weight: P(uncensored)
        W = self.weight(t, theta)

        # Design matrix for [α, β, γ]: scale by 1/σ for mean params
        ones = torch.ones(n, device=t.device)
        T_design = torch.stack([ones / sigma, t_tilde / sigma, ones], dim=1)

        Lambda = (T_design.T @ (W.unsqueeze(1) * T_design)) / n
        return Lambda + 1e-4 * torch.eye(3, device=t.device)

    def compute_hessian_with_diagnostics(
        self, theta: torch.Tensor, t: torch.Tensor, t_mean: torch.Tensor
    ) -> tuple:
        """Compute 3×3 Hessian with stability diagnostics."""
        Lambda = self.compute_hessian(theta, t, t_mean)
        eigenvalues = torch.linalg.eigvalsh(Lambda)
        min_eig = eigenvalues.min().item()
        max_eig = eigenvalues.max().item()
        condition = max_eig / (min_eig + 1e-10)
        return Lambda, min_eig, condition

    def influence_score(
        self, y, t, theta, t_mean, t_var, lambda_inv, return_correction: bool = False
    ):
        """Influence score for 3-parameter Tobit model.

        ψᵢ = H(θᵢ) - ∇ℓᵢ @ Λ⁻¹ @ ∇H

        Score vector (for NLL minimization):
        - Uncensored: [-(y-μ)/σ², -T(y-μ)/σ², 1-e²]
        - Censored: [λ/σ, Tλ/σ, -zλ] where λ = φ/(1-Φ)
        """
        n = len(y)
        h_i = self.h_value(theta, t)

        # Extract parameters
        alpha, beta, gamma = theta[:, 0], theta[:, 1], theta[:, 2]
        sigma = torch.exp(torch.clamp(gamma, -10, 10))
        mu = alpha + beta * t
        z = mu / sigma
        censored = (y <= 0)

        # Mills ratio computation
        dist = torch.distributions.Normal(0, 1)
        phi_z = torch.exp(dist.log_prob(z))
        Phi_neg_z = dist.cdf(-z).clamp(min=1e-10)
        mills = phi_z / Phi_neg_z

        e = (y - mu) / sigma

        # Score components (gradient of NLL)
        # Uncensored: ∂ℓ/∂μ = (μ-y)/σ² = -e/σ
        score_alpha_unc = -e / sigma
        score_beta_unc = -e * t / sigma
        score_gamma_unc = 1 - e ** 2

        # Censored: ∂ℓ/∂μ = λ/σ (positive for NLL!)
        score_alpha_cen = mills / sigma
        score_beta_cen = mills * t / sigma
        score_gamma_cen = -z * mills

        score_alpha = torch.where(censored, score_alpha_cen, score_alpha_unc)
        score_beta = torch.where(censored, score_beta_cen, score_beta_unc)
        score_gamma = torch.where(censored, score_gamma_cen, score_gamma_unc)

        # Stack scores into l_theta
        l_theta = torch.stack([score_alpha, score_beta, score_gamma], dim=1)

        # Get gradient of H
        H_grad = self.h_gradient(theta, t).to(theta.device)

        # Influence correction: l_theta @ Λ⁻¹ @ ∇H
        correction = l_theta @ (lambda_inv @ H_grad)
        psi = h_i - correction

        if return_correction:
            return psi, correction
        return psi


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
# Heteroskedastic Linear Family
# =============================================================================

class HeteroLinearFamily(BaseFamily):
    """Y ~ N(α + βT, σ²(X)) where σ² = exp(γ). Target: E[σ²(X)].

    Model:
        Y = α(X) + β(X)T + σ(X)ε, where σ² = exp(γ)
        θ = [α, β, γ]

    Loss (Gaussian NLL):
        L = ½γ + (Y - μ)² / (2σ²)

    Target:
        μ* = E[σ²(X)] = E[exp(γ(X))]

    Influence Score (from Farrell et al. derivation):
        ψᵢ = 2σ̂² - ε̂²

    This elegant formula corrects for regularization bias when estimating
    the average variance.
    """
    name = "heterolinear"
    n_params = 3  # [α, β, γ]

    def loss(self, y, t, theta):
        alpha, beta, gamma = theta[:, 0], theta[:, 1], theta[:, 2]
        mu = alpha + beta * t
        gamma_clamp = torch.clamp(gamma, -10, 10)
        sigma2 = torch.exp(gamma_clamp)
        # Gaussian NLL: ½log(σ²) + (y-μ)²/(2σ²)
        return 0.5 * gamma_clamp + 0.5 * (y - mu) ** 2 / sigma2

    def residual(self, y, t, theta):
        mu = theta[:, 0] + theta[:, 1] * t
        return y - mu

    def weight(self, t, theta):
        sigma2 = torch.exp(torch.clamp(theta[:, 2], -10, 10))
        return 1.0 / sigma2

    def h_value(self, theta: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """Target: E[σ²(X)] = E[exp(γ)]."""
        return torch.exp(torch.clamp(theta[:, 2], -10, 10))

    def influence_score(
        self, y, t, theta, t_mean, t_var, lambda_inv, return_correction: bool = False
    ):
        """Influence score for variance target: ψᵢ = 2σ̂² - ε̂².

        Note: This is a special case where the influence score has a closed form.
        The "correction" is the difference between ψ and the plug-in estimate.
        """
        sigma2 = torch.exp(torch.clamp(theta[:, 2], -10, 10))
        eps = self.residual(y, t, theta)
        psi = 2 * sigma2 - eps ** 2
        if return_correction:
            correction = sigma2 - eps ** 2  # Correction = psi - sigma2
            return psi, correction
        return psi


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
    "heterolinear": HeteroLinearFamily,
}


def get_family(name: str, **kwargs) -> BaseFamily:
    """Factory function for families."""
    if name not in FAMILIES:
        raise ValueError(f"Unknown family: {name}. Available: {list(FAMILIES.keys())}")
    return FAMILIES[name](**kwargs) if kwargs else FAMILIES[name]()
