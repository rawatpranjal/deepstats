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
# Multinomial Logit (Conditional Logit) Family
# =============================================================================

class MultinomialLogitFamily(BaseFamily):
    """Conditional Logit (McFadden's Choice Model) with alternative-specific attributes.

    Choice probability:
        P(Y_ij=1 | x_i, w_i) = exp(V_ij) / Σ_m exp(V_im)
        V_ij = α_j(W_i) + x'_{i,j} β(W_i)

    Parameters:
        - α_j(W): J-1 alternative-specific intercepts (j=0 normalized to 0)
        - β(W): K generic coefficients for continuous attributes
        - Total: (J-1) + K parameters

    Data structure:
        - t: packed X_alt tensor of shape (N, J*K), unpacked to (N, J, K)
        - y: chosen alternative (integer 0...J-1)
        - theta: (N, (J-1)+K) structural parameters

    Targets:
        - 'beta': E[β_k(W)] for k-th coefficient
        - 'choice_prob': E[P(Y=j|W,X)] average choice probability

    Following FLM paper formulas:
        - Gradient: ℓ_δ = [c_{i,1}..c_{i,J-1}, c̃_{i,1}..c̃_{i,K}]
        - Hessian: Λ = E[X̃' Ġ X̃] where Ġ has diag p(1-p), off-diag -p_j*p_m
    """
    name = "multinomial_logit"

    def __init__(self, J: int, K: int, target: str = "beta", target_idx: int = 0):
        """Initialize MultinomialLogitFamily.

        Args:
            J: Number of alternatives
            K: Number of continuous attributes per alternative
            target: 'beta' (average coefficient) or 'choice_prob' (average choice probability)
            target_idx: Which coefficient (0..K-1) or alternative (0..J-1) to target
        """
        if J < 2:
            raise ValueError(f"J must be >= 2, got {J}")
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if target not in ("beta", "choice_prob"):
            raise ValueError(f"target must be 'beta' or 'choice_prob', got '{target}'")

        self.J = J
        self.K = K
        self.n_params = (J - 1) + K
        self.target = target
        self.target_idx = target_idx

    def _unpack_and_compute_probs(self, t: torch.Tensor, theta: torch.Tensor):
        """Unpack data and compute choice probabilities.

        Args:
            t: Packed X_alt (N, J*K)
            theta: Structural parameters (N, (J-1)+K)

        Returns:
            X_alt: Unpacked alternative attributes (N, J, K)
            probs: Choice probabilities (N, J)
            V: Utilities (N, J)
        """
        N = t.shape[0]
        X_alt = t.reshape(N, self.J, self.K)

        # Extract parameters: first J-1 are intercepts, next K are coefficients
        alpha = theta[:, :self.J - 1]  # (N, J-1)
        beta = theta[:, self.J - 1:]   # (N, K)

        # Build full alpha with α_0 = 0 (normalized)
        alpha_full = torch.cat([
            torch.zeros(N, 1, device=theta.device),
            alpha
        ], dim=1)  # (N, J)

        # Compute utilities: V_ij = α_j + Σ_k x_{ijk} β_k
        # einsum: (N, J, K) @ (N, K) -> (N, J)
        V = alpha_full + torch.einsum('njk,nk->nj', X_alt, beta)

        # Clamp for numerical stability before softmax
        V = torch.clamp(V, -20, 20)

        # Softmax for probabilities
        probs = torch.softmax(V, dim=1)
        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)

        return X_alt, probs, V

    def loss(self, y: torch.Tensor, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Multinomial logit NLL (cross-entropy loss).

        Args:
            y: Chosen alternative (N,) integer or (N, J) one-hot
            t: Packed X_alt (N, J*K)
            theta: Structural parameters (N, (J-1)+K)

        Returns:
            Per-sample NLL (N,)
        """
        N = t.shape[0]
        X_alt, probs, V = self._unpack_and_compute_probs(t, theta)

        # Use log_softmax for numerical stability
        log_probs = torch.log_softmax(V, dim=1)

        # Handle both integer and one-hot y
        if y.dim() == 1:
            # Integer labels: gather the log prob of chosen alternative
            nll = -log_probs[torch.arange(N, device=y.device), y.long()]
        else:
            # One-hot: dot product
            nll = -torch.sum(y * log_probs, dim=1)

        return nll

    def residual(self, y: torch.Tensor, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Per-alternative residuals: c_{ij} = y_{ij} - P(Y_ij=1).

        Args:
            y: Chosen alternative (N,) integer or (N, J) one-hot
            t: Packed X_alt (N, J*K)
            theta: Structural parameters (N, (J-1)+K)

        Returns:
            Residuals (N, J) for each alternative
        """
        N = t.shape[0]
        X_alt, probs, V = self._unpack_and_compute_probs(t, theta)

        # Convert y to one-hot if integer
        if y.dim() == 1:
            y_onehot = torch.zeros(N, self.J, device=y.device)
            y_onehot.scatter_(1, y.long().unsqueeze(1), 1)
        else:
            y_onehot = y

        return y_onehot - probs  # (N, J)

    def weight(self, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Fisher information weight matrix Ġ (N, J, J).

        Ġ_{jm} = p_j(1-p_j) if j=m, else -p_j*p_m
        Equivalently: Ġ = diag(p) - p @ p^T

        Args:
            t: Packed X_alt (N, J*K)
            theta: Structural parameters (N, (J-1)+K)

        Returns:
            Weight matrices (N, J, J)
        """
        X_alt, probs, V = self._unpack_and_compute_probs(t, theta)

        # G = diag(p) - p @ p^T
        G_diag = torch.diag_embed(probs)  # (N, J, J)
        G_outer = torch.einsum('ni,nj->nij', probs, probs)  # (N, J, J)
        G = G_diag - G_outer

        return G

    def h_value(self, theta: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """Target functional H(θ).

        For 'beta' target: H = β_{target_idx}
        For 'choice_prob' target: H = P(Y=target_idx | θ, x)
        """
        if self.target == "beta":
            return theta[:, self.J - 1 + self.target_idx]
        else:  # choice_prob
            if t is None:
                raise ValueError("t required for choice_prob target")
            X_alt, probs, V = self._unpack_and_compute_probs(t, theta)
            return probs[:, self.target_idx]

    def h_gradient(self, theta: torch.Tensor = None, t: torch.Tensor = None) -> torch.Tensor:
        """Gradient of H w.r.t. θ = [(J-1)+K] parameters.

        For 'beta' target: ∇H = e_{(J-1)+target_idx} (unit vector)
        For 'choice_prob' target: chain rule through softmax
        """
        n_params = self.n_params

        if self.target == "beta":
            grad = torch.zeros(n_params)
            grad[self.J - 1 + self.target_idx] = 1.0
            return grad

        # choice_prob target: ∇H = ∇P(Y=j)
        if theta is None or t is None:
            # Return beta gradient as fallback
            grad = torch.zeros(n_params)
            grad[self.J - 1] = 1.0
            return grad

        X_alt, probs, V = self._unpack_and_compute_probs(t, theta)
        j = self.target_idx
        p_j = probs[:, j]  # (N,)

        # ∂P_j/∂α_m = P_j * (I(j=m) - P_m) for m=1..J-1
        # Note: α_0 is normalized, so we only have gradients for α_1..α_{J-1}
        grad_alpha = torch.zeros(theta.shape[0], self.J - 1, device=theta.device)
        for m in range(1, self.J):
            indicator = 1.0 if j == m else 0.0
            grad_alpha[:, m - 1] = p_j * (indicator - probs[:, m])

        # ∂P_j/∂β_k = P_j * Σ_m (I(j=m) - P_m) * x_{mk}
        #           = P_j * (x_jk - Σ_m P_m * x_{mk})
        x_j = X_alt[:, j, :]  # (N, K)
        x_weighted = torch.einsum('nj,njk->nk', probs, X_alt)  # (N, K)
        grad_beta = p_j.unsqueeze(1) * (x_j - x_weighted)  # (N, K)

        # Average gradients across observations
        grad = torch.cat([grad_alpha.mean(0), grad_beta.mean(0)])
        return grad

    def compute_hessian(self, theta: torch.Tensor, t: torch.Tensor, t_mean: torch.Tensor) -> torch.Tensor:
        """Compute ((J-1)+K) × ((J-1)+K) Hessian Λ.

        Λ = (1/N) Σ_i X̃_i' Ġ_i X̃_i

        where X̃_i is the extended design matrix (J, n_params):
        - Columns 0..J-2: indicator for alternatives 1..J-1
        - Columns J-1..n_params-1: x_{ijk} for each attribute k

        Args:
            theta: (N, n_params)
            t: Packed X_alt (N, J*K)
            t_mean: Not used (for API compatibility)

        Returns:
            Lambda: (n_params, n_params) Hessian matrix
        """
        N = t.shape[0]
        n_params = self.n_params
        X_alt = t.reshape(N, self.J, self.K)

        # Get weight matrices G (N, J, J)
        G = self.weight(t, theta)

        # Build extended design matrix X̃ (N, J, n_params)
        X_tilde = torch.zeros(N, self.J, n_params, device=t.device)

        # Alpha part: indicator for alternatives 1..J-1
        # X̃[n, j, j-1] = 1 for j=1..J-1
        for j in range(1, self.J):
            X_tilde[:, j, j - 1] = 1.0

        # Beta part: x_{ijk} for each attribute k
        X_tilde[:, :, self.J - 1:] = X_alt  # (N, J, K)

        # Λ = (1/N) Σ_i X̃_i' Ġ_i X̃_i
        # Using einsum: (N,J,p)' @ (N,J,J) @ (N,J,q) -> (p,q)
        Lambda = torch.einsum('njp,njm,nmq->pq', X_tilde, G, X_tilde) / N

        # Ridge regularization for numerical stability
        Lambda = Lambda + 1e-4 * torch.eye(n_params, device=t.device)

        return Lambda

    def compute_hessian_with_diagnostics(
        self, theta: torch.Tensor, t: torch.Tensor, t_mean: torch.Tensor
    ) -> tuple:
        """Compute Hessian with stability diagnostics."""
        Lambda = self.compute_hessian(theta, t, t_mean)
        eigenvalues = torch.linalg.eigvalsh(Lambda)
        min_eig = eigenvalues.min().item()
        max_eig = eigenvalues.max().item()
        condition = max_eig / (min_eig + 1e-10)
        return Lambda, min_eig, condition

    def influence_score(
        self, y, t, theta, t_mean, t_var, lambda_inv, return_correction: bool = False
    ):
        """Influence score for multinomial logit.

        ψ_i = H(θ_i) - l_θ_i @ Λ⁻¹ @ ∇H

        Score vector l_θ (gradient of NLL):
        - l_α_j = -(y_j - p_j) for j=1..J-1
        - l_β_k = -Σ_j (y_j - p_j) x_{jk} for k=0..K-1

        Args:
            y: Chosen alternative (N,) or (N, J) one-hot
            t: Packed X_alt (N, J*K)
            theta: (N, n_params)
            t_mean, t_var: Not used (API compatibility)
            lambda_inv: Inverted Hessian (n_params, n_params)
            return_correction: If True, return (psi, correction) tuple

        Returns:
            psi: Influence scores (N,)
            correction: (optional) Correction terms (N,)
        """
        N = y.shape[0]
        n_params = self.n_params
        X_alt = t.reshape(N, self.J, self.K)

        # Target value
        h_i = self.h_value(theta, t)

        # Get residuals (N, J): c_{ij} = y_{ij} - p_{ij}
        residuals = self.residual(y, t, theta)

        # Build score vector l_θ (gradient of NLL = negative of score)
        # l_α_j = -(y_j - p_j) = p_j - y_j for j=1..J-1
        l_alpha = -residuals[:, 1:]  # (N, J-1) - exclude j=0

        # l_β_k = -Σ_j (y_j - p_j) x_{jk}
        l_beta = -torch.einsum('nj,njk->nk', residuals, X_alt)  # (N, K)

        l_theta = torch.cat([l_alpha, l_beta], dim=1)  # (N, n_params)

        # Gradient of H
        H_grad = self.h_gradient(theta, t).to(theta.device)

        # Correction term: l_θ @ Λ⁻¹ @ ∇H
        correction = l_theta @ (lambda_inv @ H_grad)

        psi = h_i - correction

        if return_correction:
            return psi, correction
        return psi


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
    "multinomial_logit": MultinomialLogitFamily,
}


def get_family(name: str, **kwargs) -> BaseFamily:
    """Factory function for families."""
    if name not in FAMILIES:
        raise ValueError(f"Unknown family: {name}. Available: {list(FAMILIES.keys())}")
    return FAMILIES[name](**kwargs) if kwargs else FAMILIES[name]()
