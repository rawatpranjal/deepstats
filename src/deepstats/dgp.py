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
    """True baseline: sin(pi*X_1) + X_2^2 + exp(X_3/2)."""
    return np.sin(np.pi * X[:, 0]) + X[:, 1]**2 + np.exp(X[:, 2] / 2)


def beta_star(X: np.ndarray) -> np.ndarray:
    """True CATE: cos(pi*X_1) * I(X_4 > 0) + 0.5*X_5."""
    return np.cos(np.pi * X[:, 0]) * (X[:, 3] > 0).astype(float) + 0.5 * X[:, 4]


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


class BaseDGP(ABC):
    """Abstract base for DGPs."""
    name: str = "base"

    def __init__(self, d: int = 10, seed: int = 42):
        if d < 10:
            raise ValueError("d must be at least 10")
        self.d = d
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
        """Generate (X, T, alpha, beta) common to all DGPs."""
        X = self.rng.uniform(-1, 1, (n, self.d))
        alpha = alpha_star(X)
        beta = beta_star(X)
        T = generate_treatment(X, beta, self.rng)
        return X, T, alpha, beta


# =============================================================================
# DGP Implementations
# =============================================================================

class LinearDGP(BaseDGP):
    """Y = alpha + beta*T + epsilon, epsilon ~ N(0, sigma^2)."""
    name = "linear"

    def __init__(self, d: int = 10, sigma: float = 1.0, seed: int = 42):
        super().__init__(d, seed)
        self.sigma = sigma

    def generate(self, n: int) -> DGPResult:
        X, T, alpha, beta = self._generate_base_data(n)
        mu = alpha + beta * T
        Y = mu + self.rng.normal(0, self.sigma, n)
        return DGPResult(X, T, Y, alpha, beta, self.compute_true_mu())


class GammaDGP(BaseDGP):
    """Y ~ Gamma(shape, mu/shape) where mu = exp(alpha + beta*T)."""
    name = "gamma"

    def __init__(self, d: int = 10, shape: float = 2.0, seed: int = 42):
        super().__init__(d, seed)
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

    def __init__(self, d: int = 10, scale: float = 1.0, seed: int = 42):
        super().__init__(d, seed)
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
    """Y ~ Bernoulli(p) where p = sigmoid(alpha + beta*T)."""
    name = "logit"

    def generate(self, n: int) -> DGPResult:
        X, T, alpha, beta = self._generate_base_data(n)
        alpha_s, beta_s = alpha * 0.5, beta * 0.5
        eta = alpha_s + beta_s * T
        p = 1 / (1 + np.exp(-np.clip(eta, -20, 20)))
        Y = self.rng.binomial(1, p).astype(float)
        return DGPResult(X, T, Y, alpha_s, beta_s, self._scaled_mu(0.5))

    def _scaled_mu(self, scale: float) -> float:
        if self._mu_true is None:
            rng = np.random.default_rng(self.seed + 999999)
            X = rng.uniform(-1, 1, (100000, self.d))
            self._mu_true = float(np.mean(beta_star(X) * scale))
        return self._mu_true


class TobitDGP(BaseDGP):
    """Y = max(0, alpha + beta*T + epsilon) - censored at 0."""
    name = "tobit"

    def __init__(self, d: int = 10, sigma: float = 1.0, seed: int = 42):
        super().__init__(d, seed)
        self.sigma = sigma

    def generate(self, n: int) -> DGPResult:
        X, T, alpha, beta = self._generate_base_data(n)
        Y_star = alpha + beta * T + self.rng.normal(0, self.sigma, n)
        Y = np.maximum(0, Y_star)
        return DGPResult(X, T, Y, alpha, beta, self.compute_true_mu())


class NegBinDGP(BaseDGP):
    """Y ~ NegBin with mu = exp(alpha + beta*T), Var = mu + alpha*mu^2."""
    name = "negbin"

    def __init__(self, d: int = 10, overdispersion: float = 0.5, seed: int = 42):
        super().__init__(d, seed)
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

    def __init__(self, d: int = 10, shape: float = 2.0, seed: int = 42):
        super().__init__(d, seed)
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
# Ground Truth Verification
# =============================================================================

def verify_ground_truth(n_mc: int = 1_000_000, seed: int = 42) -> dict:
    """Verify ground truth μ* for all DGP types.

    Uses 1M Monte Carlo samples for high precision.
    Returns dict mapping model name to computed μ*.

    The ground truth is:
    - Linear/Gumbel/Tobit: μ* = E[β*(X)] (no scaling)
    - Gamma/Poisson/NegBin/Weibull: μ* = E[0.3 × β*(X)] (log-link scaling)
    - Logit: μ* = E[0.5 × β*(X)] (logit scaling)
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n_mc, 10))
    beta = beta_star(X)  # cos(πX₁)·I(X₄>0) + 0.5·X₅

    results = {
        # Linear models (no scaling)
        "linear": float(np.mean(beta)),
        "gumbel": float(np.mean(beta)),
        "tobit": float(np.mean(beta)),

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
        print(f"{model:12s}: μ* = {mu:.6f}")
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
}


def get_dgp(name: str, **kwargs) -> BaseDGP:
    """Factory function for DGPs."""
    if name not in DGPS:
        raise ValueError(f"Unknown DGP: {name}. Available: {list(DGPS.keys())}")
    return DGPS[name](**kwargs)
