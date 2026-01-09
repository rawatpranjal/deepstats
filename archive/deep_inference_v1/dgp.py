"""Data Generating Processes for influence function validation.

All 8 DGPs share the same covariate distribution and treatment assignment:
- X ~ Uniform(-1, 1)^d
- alpha(X) = sin(pi*X_1) + X_2^2 + exp(X_3/2)
- beta(X) = cos(pi*X_1) * I(X_4 > 0) + 0.5*X_5
- T = 0.5*beta(X) + 0.2*sum(X_6...X_10) + nu
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


# =============================================================================
# True Parameter Functions (Complex - forces regularization bias)
# =============================================================================

def alpha_star(X: np.ndarray) -> np.ndarray:
    """True baseline: complex nonlinear function.

    α*(X) = sin(2πX₁) + X₂³ - 2cos(πX₃) + exp(X₄/3)·I(X₄>0) + 0.5·X₅·X₆

    Features:
    - Higher frequency sine (2π instead of π)
    - Cubic term (X₂³)
    - Cosine interaction
    - Indicator-weighted exponential
    - Covariate interaction (X₅·X₆)
    """
    return (np.sin(2 * np.pi * X[:, 0])
            + X[:, 1]**3
            - 2 * np.cos(np.pi * X[:, 2])
            + np.exp(X[:, 3] / 3) * (X[:, 3] > 0).astype(float)
            + 0.5 * X[:, 4] * X[:, 5])


def beta_star(X: np.ndarray) -> np.ndarray:
    """True CATE: complex heterogeneous treatment effect.

    β*(X) = cos(2πX₁)·sin(πX₂) + 0.8·tanh(3X₃) - 0.5·X₄² + 0.3·X₅·I(X₆>0)

    Features:
    - Product of trig functions (highly nonlinear)
    - Tanh saturation effect
    - Quadratic term
    - Indicator interaction
    """
    return (np.cos(2 * np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
            + 0.8 * np.tanh(3 * X[:, 2])
            - 0.5 * X[:, 3]**2
            + 0.3 * X[:, 4] * (X[:, 5] > 0).astype(float))


def generate_treatment(X: np.ndarray, beta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate confounded treatment: T = 0.5*beta + 0.2*sum(X_6..X_10) + nu."""
    n = len(X)
    confounding = 0.2 * np.sum(X[:, 5:10], axis=1)
    nu = rng.normal(0, 0.5, n)
    return 0.5 * beta + confounding + nu


def compute_true_mu(n_mc: int = 100000, seed: int = 12345) -> float:
    """Compute true E[beta(X)] via Monte Carlo."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n_mc, 10))
    return float(np.mean(beta_star(X)))


# =============================================================================
# Base Classes
# =============================================================================

@dataclass
class DGPResult:
    """Container for generated data."""
    X: np.ndarray
    T: np.ndarray
    Y: np.ndarray
    alpha_true: np.ndarray
    beta_true: np.ndarray
    mu_true: float


@dataclass
class MultinomialLogitResult:
    """Extended result container for conditional logit DGP.

    Stores the full conditional logit data structure with compatibility
    properties for the existing inference pipeline.

    Attributes:
        W: Individual characteristics (N, d_w) - drives heterogeneous params
        X_alt: Alternative-specific attributes (N, J, K)
        Y: Chosen alternative (N,) as integer 0..J-1
        alpha_true: True intercepts (N, J-1)
        beta_true: True coefficients (N, K)
        mu_true: Ground truth E[β_k(W)] for target coefficient
    """
    W: np.ndarray           # (N, d_w)
    X_alt: np.ndarray       # (N, J, K)
    Y: np.ndarray           # (N,) integer choices
    alpha_true: np.ndarray  # (N, J-1)
    beta_true: np.ndarray   # (N, K)
    mu_true: float          # E[β_target_idx(W)]

    @property
    def X(self) -> np.ndarray:
        """Compatibility: individual characteristics as X."""
        return self.W

    @property
    def T(self) -> np.ndarray:
        """Compatibility: packed X_alt as T (N, J*K)."""
        return self.X_alt.reshape(len(self.W), -1)


class BaseDGP(ABC):
    """Abstract base for DGPs."""
    name: str = "base"

    def __init__(self, d: int = 10, n_noise: int = 10, seed: int = 42):
        if d < 10:
            raise ValueError("d must be at least 10")
        self.d = d
        self.n_noise = n_noise
        self.d_total = d + n_noise  # Total features (signal + noise)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._mu_true: Optional[float] = None

    def reset_rng(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def generate(self, n: int) -> DGPResult:
        pass

    def compute_true_mu(self, n_mc: int = 100000) -> float:
        if self._mu_true is None:
            self._mu_true = compute_true_mu(n_mc, seed=self.seed + 999999)
        return self._mu_true

    def _generate_base_data(self, n: int) -> tuple:
        """Generate (X, T, alpha, beta) common to all DGPs.

        X includes both signal features (d) and noise features (n_noise).
        Only signal features are used for alpha, beta, and treatment.
        """
        # Signal features (used for alpha, beta, T)
        X_signal = self.rng.uniform(-1, 1, (n, self.d))
        alpha = alpha_star(X_signal)
        beta = beta_star(X_signal)
        T = generate_treatment(X_signal, beta, self.rng)

        # Add noise features if n_noise > 0
        if self.n_noise > 0:
            X_noise = self.rng.uniform(-1, 1, (n, self.n_noise))
            X = np.concatenate([X_signal, X_noise], axis=1)
        else:
            X = X_signal

        return X, T, alpha, beta


# =============================================================================
# DGP Implementations
# =============================================================================

class LinearDGP(BaseDGP):
    """Y = alpha + beta*T + epsilon, epsilon ~ N(0, sigma^2)."""
    name = "linear"

    def __init__(self, d: int = 10, n_noise: int = 10, sigma: float = 1.0, seed: int = 42):
        super().__init__(d, n_noise, seed)
        self.sigma = sigma

    def generate(self, n: int) -> DGPResult:
        X, T, alpha, beta = self._generate_base_data(n)
        mu = alpha + beta * T
        Y = mu + self.rng.normal(0, self.sigma, n)
        return DGPResult(X, T, Y, alpha, beta, self.compute_true_mu())


class GammaDGP(BaseDGP):
    """Y ~ Gamma(shape, mu/shape) where mu = exp(alpha + beta*T)."""
    name = "gamma"

    def __init__(self, d: int = 10, n_noise: int = 10, shape: float = 2.0, seed: int = 42):
        super().__init__(d, n_noise, seed)
        self.shape = shape

    def generate(self, n: int) -> DGPResult:
        X, T, alpha, beta = self._generate_base_data(n)
        alpha_s, beta_s = alpha * 0.5, beta * 0.3
        mu = np.exp(np.clip(alpha_s + beta_s * T, -5, 5))
        Y = self.rng.gamma(self.shape, mu / self.shape)
        return DGPResult(X, T, Y, alpha_s, beta_s, self._scaled_mu(0.3))

    def _scaled_mu(self, scale: float) -> float:
        if self._mu_true is None:
            rng = np.random.default_rng(self.seed + 999999)
            X = rng.uniform(-1, 1, (100000, self.d))
            self._mu_true = float(np.mean(beta_star(X) * scale))
        return self._mu_true


class GumbelDGP(BaseDGP):
    """Y ~ Gumbel(mu, scale) where mu = alpha + beta*T."""
    name = "gumbel"

    def __init__(self, d: int = 10, n_noise: int = 10, scale: float = 1.0, seed: int = 42):
        super().__init__(d, n_noise, seed)
        self.scale = scale

    def generate(self, n: int) -> DGPResult:
        X, T, alpha, beta = self._generate_base_data(n)
        mu = alpha + beta * T
        U = self.rng.uniform(0.0001, 0.9999, n)
        Y = mu - self.scale * np.log(-np.log(U))
        return DGPResult(X, T, Y, alpha, beta, self.compute_true_mu())


class PoissonDGP(BaseDGP):
    """Y ~ Poisson(lambda) where lambda = exp(alpha + beta*T)."""
    name = "poisson"

    def generate(self, n: int) -> DGPResult:
        X, T, alpha, beta = self._generate_base_data(n)
        alpha_s, beta_s = alpha * 0.3, beta * 0.3
        lam = np.exp(np.clip(alpha_s + beta_s * T, -5, 5))
        Y = self.rng.poisson(lam).astype(float)
        return DGPResult(X, T, Y, alpha_s, beta_s, self._scaled_mu(0.3))

    def _scaled_mu(self, scale: float) -> float:
        if self._mu_true is None:
            rng = np.random.default_rng(self.seed + 999999)
            X = rng.uniform(-1, 1, (100000, self.d))
            self._mu_true = float(np.mean(beta_star(X) * scale))
        return self._mu_true


class LogitDGP(BaseDGP):
    """Y ~ Bernoulli(p) where p = sigmoid(alpha + beta*T).

    Supports two ground truth estimands:
    - 'beta': E[β(X)] - latent log-odds effect
    - 'ame': E[p(1-p)β(X)] - average marginal effect on probability
    """
    name = "logit"

    def __init__(self, d: int = 10, n_noise: int = 10, seed: int = 42):
        super().__init__(d, n_noise, seed)
        self._mu_true_beta = None
        self._mu_true_ame = None

    def generate(self, n: int) -> DGPResult:
        X, T, alpha, beta = self._generate_base_data(n)
        alpha_s, beta_s = alpha * 0.5, beta * 0.5
        eta = alpha_s + beta_s * T
        p = 1 / (1 + np.exp(-np.clip(eta, -20, 20)))
        Y = self.rng.binomial(1, p).astype(float)
        return DGPResult(X, T, Y, alpha_s, beta_s, self.compute_true_mu(target="beta"))

    def compute_true_mu(self, target: str = "beta", n_mc: int = 100000) -> float:
        """Compute true E[β] or E[p(1-p)β] via Monte Carlo.

        Args:
            target: 'beta' for E[β], 'ame' for E[p(1-p)β]
            n_mc: Monte Carlo samples

        Returns:
            True population parameter
        """
        if target == "beta":
            if self._mu_true_beta is None:
                rng = np.random.default_rng(self.seed + 999999)
                X = rng.uniform(-1, 1, (n_mc, self.d))
                self._mu_true_beta = float(np.mean(beta_star(X) * 0.5))
            return self._mu_true_beta
        elif target == "ame":
            if self._mu_true_ame is None:
                rng = np.random.default_rng(self.seed + 999999)
                X = rng.uniform(-1, 1, (n_mc, self.d))
                beta_s = beta_star(X) * 0.5
                T = generate_treatment(X, beta_s * 2, rng)  # Use unscaled beta for treatment
                alpha_s = alpha_star(X) * 0.5
                eta = alpha_s + beta_s * T
                p = 1 / (1 + np.exp(-np.clip(eta, -20, 20)))
                # AME = E[p(1-p)β]
                self._mu_true_ame = float(np.mean(p * (1 - p) * beta_s))
            return self._mu_true_ame
        else:
            raise ValueError(f"target must be 'beta' or 'ame', got '{target}'")


class TobitDGP(BaseDGP):
    """Y = max(0, alpha + beta*T + epsilon) - censored at 0.

    Supports two ground truth estimands:
    - 'latent': E[β(X)] - effect on latent Y*
    - 'observed': E[β(X)·Φ(z)] - effect on observed E[Y|X,T]

    where z = (α + βT)/σ.
    """
    name = "tobit"

    def __init__(self, d: int = 10, n_noise: int = 10, sigma: float = 1.0, seed: int = 42):
        super().__init__(d, n_noise, seed)
        self.sigma = sigma
        self._mu_true_latent = None
        self._mu_true_observed = None

    def generate(self, n: int) -> DGPResult:
        X, T, alpha, beta = self._generate_base_data(n)
        Y_star = alpha + beta * T + self.rng.normal(0, self.sigma, n)
        Y = np.maximum(0, Y_star)
        return DGPResult(X, T, Y, alpha, beta, self.compute_true_mu())

    def compute_true_mu(self, target: str = "latent", n_mc: int = 100000) -> float:
        """Compute true E[β] or E[β·Φ(z)] via Monte Carlo.

        Args:
            target: 'latent' for E[β], 'observed' for E[β·Φ(z)]
            n_mc: Number of Monte Carlo samples

        Returns:
            Ground truth value
        """
        from scipy.stats import norm as scipy_norm

        if target == "latent":
            if self._mu_true_latent is None:
                self._mu_true_latent = compute_true_mu(n_mc, seed=self.seed + 999999)
            return self._mu_true_latent
        else:  # observed
            if self._mu_true_observed is None:
                rng = np.random.default_rng(self.seed + 999998)
                X = rng.uniform(-1, 1, (n_mc, self.d))
                alpha = alpha_star(X)
                beta = beta_star(X)
                T = generate_treatment(X, beta, rng)
                mu = alpha + beta * T
                z = mu / self.sigma
                Phi_z = scipy_norm.cdf(z)
                self._mu_true_observed = float(np.mean(beta * Phi_z))
            return self._mu_true_observed


class NegBinDGP(BaseDGP):
    """Y ~ NegBin with mu = exp(alpha + beta*T), Var = mu + alpha*mu^2."""
    name = "negbin"

    def __init__(self, d: int = 10, n_noise: int = 10, overdispersion: float = 0.5, seed: int = 42):
        super().__init__(d, n_noise, seed)
        self.overdispersion = overdispersion

    def generate(self, n: int) -> DGPResult:
        X, T, alpha, beta = self._generate_base_data(n)
        alpha_s, beta_s = alpha * 0.3, beta * 0.3
        mu = np.exp(np.clip(alpha_s + beta_s * T, -5, 5))
        r = 1.0 / self.overdispersion
        p = r / (r + mu)
        Y = self.rng.negative_binomial(r, p).astype(float)
        return DGPResult(X, T, Y, alpha_s, beta_s, self._scaled_mu(0.3))

    def _scaled_mu(self, scale: float) -> float:
        if self._mu_true is None:
            rng = np.random.default_rng(self.seed + 999999)
            X = rng.uniform(-1, 1, (100000, self.d))
            self._mu_true = float(np.mean(beta_star(X) * scale))
        return self._mu_true


class WeibullDGP(BaseDGP):
    """Y ~ Weibull(k, lambda) where lambda = exp(alpha + beta*T)."""
    name = "weibull"

    def __init__(self, d: int = 10, n_noise: int = 10, shape: float = 2.0, seed: int = 42):
        super().__init__(d, n_noise, seed)
        self.shape = shape

    def generate(self, n: int) -> DGPResult:
        X, T, alpha, beta = self._generate_base_data(n)
        alpha_s, beta_s = alpha * 0.3, beta * 0.3
        lam = np.exp(np.clip(alpha_s + beta_s * T, -5, 5))
        U = self.rng.uniform(0, 1, n)
        Y = lam * (-np.log(U + 1e-10)) ** (1.0 / self.shape)
        return DGPResult(X, T, Y, alpha_s, beta_s, self._scaled_mu(0.3))

    def _scaled_mu(self, scale: float) -> float:
        if self._mu_true is None:
            rng = np.random.default_rng(self.seed + 999999)
            X = rng.uniform(-1, 1, (100000, self.d))
            self._mu_true = float(np.mean(beta_star(X) * scale))
        return self._mu_true


# =============================================================================
# Heteroskedastic Linear DGP
# =============================================================================

class HeteroLinearDGP(BaseDGP):
    """Y ~ N(α + βT, σ²(X)) where σ² = exp(X₁). Target: E[σ²(X)].

    This DGP generates data with heteroskedastic errors:
    - Mean: μ = α(X) + β(X)T
    - Variance: σ²(X) = exp(X₁) (depends on first covariate)
    - Y = μ + σ(X) * ε, ε ~ N(0, 1)

    The target is the average variance E[σ²(X)] = E[exp(X₁)].
    For X₁ ~ Uniform(-1, 1): E[exp(X₁)] = sinh(1) ≈ 1.1752
    """
    name = "heterolinear"

    def __init__(self, d: int = 10, n_noise: int = 10, seed: int = 42):
        super().__init__(d, n_noise, seed)
        # Cache the true variance mean
        self._mu_true = (np.exp(1) - np.exp(-1)) / 2  # sinh(1) ≈ 1.1752

    def generate(self, n: int) -> DGPResult:
        X, T, alpha, beta = self._generate_base_data(n)
        mu = alpha + beta * T
        # Variance depends on X₁ (signal feature): σ² = exp(X₁)
        # Note: X[:, 1] is from signal features, not noise
        sigma = np.exp(X[:, 1] / 2)  # σ = exp(X₁/2), so σ² = exp(X₁)
        Y = mu + sigma * self.rng.normal(0, 1, n)
        # mu_true for this DGP is E[σ²] = E[exp(X₁)]
        return DGPResult(X, T, Y, alpha, beta, self._mu_true)

    def compute_true_mu(self, n_mc: int = 100000) -> float:
        """Override: E[σ²(X)] = E[exp(X₁)] for X₁ ~ U(-1,1) = sinh(1)."""
        # Analytical solution, no MC needed
        return self._mu_true


# =============================================================================
# Multinomial Logit (Conditional Logit) DGP
# =============================================================================

class MultinomialLogitDGP(BaseDGP):
    """Conditional Logit DGP with heterogeneous coefficients.

    Generates choice data where:
    - W ~ Uniform(-1, 1)^d (individual characteristics)
    - X_alt ~ Uniform(-1, 1)^{J×K} (alternative-specific attributes)
    - α_j(W): alternative-specific intercepts (J-1 free, α_0=0)
    - β(W): generic coefficients for attributes (K coefficients)
    - V_ij = α_j(W) + X_alt[j] @ β(W) (utility)
    - Y = argmax_j(V_j + ε_j) where ε ~ Gumbel(0, 1)

    Target: E[β_k(W)] for specified coefficient k.
    """
    name = "multinomial_logit"

    def __init__(
        self,
        J: int = 4,
        K: int = 3,
        d: int = 10,
        n_noise: int = 10,
        target_idx: int = 0,
        seed: int = 42
    ):
        """Initialize MultinomialLogitDGP.

        Args:
            J: Number of alternatives
            K: Number of continuous attributes per alternative
            d: Dimension of individual characteristics W
            n_noise: Number of noise features to add
            target_idx: Which β_k to use for ground truth (0..K-1)
            seed: Random seed
        """
        super().__init__(d, n_noise, seed)
        self.J = J
        self.K = K
        self.target_idx = target_idx
        self._mu_true_cache = None

    def alpha_star_j(self, W: np.ndarray, j: int) -> np.ndarray:
        """True alternative-specific intercept α_j(W).

        α_j(W) = sin(2πW_0) * j/J + W_1² * (-1)^j * 0.5 + 0.3*W_2*I(j>J/2)

        Args:
            W: Individual characteristics (N, d)
            j: Alternative index (0..J-1)

        Returns:
            α_j values (N,)
        """
        if j == 0:
            return np.zeros(len(W))  # Normalized to 0

        return (np.sin(2 * np.pi * W[:, 0]) * j / self.J
                + W[:, 1]**2 * ((-1)**j) * 0.5
                + 0.3 * W[:, 2] * (j > self.J // 2))

    def beta_star(self, W: np.ndarray) -> np.ndarray:
        """True generic coefficients β(W).

        β_k(W) = cos(2πW_0) * (k+1)/(K+1) + 0.8*tanh(2*W_3) - 0.5*W_4²
               + 0.3*W_5*I(W_6>0)

        Returns:
            β values (N, K)
        """
        N = len(W)
        beta = np.zeros((N, self.K))

        for k in range(self.K):
            # Base component varying by k
            beta[:, k] = (
                np.cos(2 * np.pi * W[:, 0]) * (k + 1) / (self.K + 1)
                + 0.8 * np.tanh(2 * W[:, 3])
                - 0.5 * W[:, 4]**2
                + 0.3 * W[:, min(5, self.d - 1)] * (W[:, min(6, self.d - 1)] > 0)
            )

        return beta

    def generate(self, n: int) -> MultinomialLogitResult:
        """Generate conditional logit data.

        Args:
            n: Number of observations

        Returns:
            MultinomialLogitResult with W, X_alt, Y, true parameters
        """
        # Individual characteristics (signal features)
        W_signal = self.rng.uniform(-1, 1, (n, self.d))

        # Add noise features
        if self.n_noise > 0:
            W_noise = self.rng.uniform(-1, 1, (n, self.n_noise))
            W = np.concatenate([W_signal, W_noise], axis=1)
        else:
            W = W_signal

        # Alternative-specific attributes
        X_alt = self.rng.uniform(-1, 1, (n, self.J, self.K))

        # True parameters (computed from signal features only)
        alpha_true = np.zeros((n, self.J - 1))
        for j in range(1, self.J):
            alpha_true[:, j - 1] = self.alpha_star_j(W_signal, j)

        beta_true = self.beta_star(W_signal)  # (N, K)

        # Compute utilities: V_ij = α_j + X_alt[j] @ β
        V = np.zeros((n, self.J))
        for j in range(self.J):
            V[:, j] = self.alpha_star_j(W_signal, j) + np.einsum('nk,nk->n', X_alt[:, j, :], beta_true)

        # Add Gumbel noise and choose maximum utility alternative
        epsilon = self.rng.gumbel(0, 1, (n, self.J))
        U = V + epsilon
        Y = np.argmax(U, axis=1).astype(float)

        # Ground truth: E[β_target_idx(W)]
        mu_true = self.compute_true_mu()

        return MultinomialLogitResult(
            W=W,
            X_alt=X_alt,
            Y=Y,
            alpha_true=alpha_true,
            beta_true=beta_true,
            mu_true=mu_true,
        )

    def compute_true_mu(self, n_mc: int = 100000) -> float:
        """Compute E[β_k(W)] for target coefficient via Monte Carlo.

        Args:
            n_mc: Number of Monte Carlo samples

        Returns:
            Ground truth E[β_target_idx(W)]
        """
        if self._mu_true_cache is None:
            rng = np.random.default_rng(self.seed + 999999)
            W = rng.uniform(-1, 1, (n_mc, self.d))
            beta = self.beta_star(W)
            self._mu_true_cache = float(np.mean(beta[:, self.target_idx]))
        return self._mu_true_cache


# =============================================================================
# Ground Truth Verification
# =============================================================================

def verify_ground_truth(n_mc: int = 1_000_000, seed: int = 42) -> dict:
    """Verify ground truth μ* for all DGP types.

    Uses 1M Monte Carlo samples for high precision.
    Returns dict mapping model name to computed μ*.

    The ground truth is:
    - Linear/Gumbel: μ* = E[β*(X)] (no scaling)
    - Tobit (latent): μ* = E[β*(X)]
    - Tobit (observed): μ* = E[β*(X)·Φ(z)] where z = (α + βT)/σ
    - Gamma/Poisson/NegBin/Weibull: μ* = E[0.3 × β*(X)] (log-link scaling)
    - Logit: μ* = E[0.5 × β*(X)] (logit scaling)
    """
    from scipy.stats import norm as scipy_norm

    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n_mc, 10))
    alpha = alpha_star(X)
    beta = beta_star(X)

    # For tobit observed, need T as well
    T = generate_treatment(X, beta, rng)
    mu_tobit = alpha + beta * T
    sigma = 1.0  # Default DGP sigma
    z = mu_tobit / sigma
    Phi_z = scipy_norm.cdf(z)

    results = {
        # Linear models (no scaling)
        "linear": float(np.mean(beta)),
        "gumbel": float(np.mean(beta)),
        "tobit_latent": float(np.mean(beta)),
        "tobit_observed": float(np.mean(beta * Phi_z)),

        # Log-link models (0.3 scaling)
        "gamma": float(np.mean(beta * 0.3)),
        "poisson": float(np.mean(beta * 0.3)),
        "negbin": float(np.mean(beta * 0.3)),
        "weibull": float(np.mean(beta * 0.3)),

        # Logit (0.5 scaling)
        "logit": float(np.mean(beta * 0.5)),
    }

    print("Ground Truth Verification (N=1,000,000)")
    print("=" * 50)
    for model, mu in results.items():
        print(f"{model:16s}: μ* = {mu:.6f}")
    print("=" * 50)

    return results


# =============================================================================
# Factory
# =============================================================================

DGPS = {
    "linear": LinearDGP,
    "gamma": GammaDGP,
    "gumbel": GumbelDGP,
    "poisson": PoissonDGP,
    "logit": LogitDGP,
    "tobit": TobitDGP,
    "negbin": NegBinDGP,
    "weibull": WeibullDGP,
    "heterolinear": HeteroLinearDGP,
    "multinomial_logit": MultinomialLogitDGP,
}


def get_dgp(name: str, **kwargs) -> BaseDGP:
    """Factory function for DGPs."""
    if name not in DGPS:
        raise ValueError(f"Unknown DGP: {name}. Available: {list(DGPS.keys())}")
    return DGPS[name](**kwargs)
