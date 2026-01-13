"""
Canonical DGP: Heterogeneous Logistic Demand (Regime C)

Most stressful case for the framework:
- Nonlinear Λ (Hessian depends on θ)
- 3-way cross-fitting required
- Confounded treatment (T depends on X)

DGP Specification:
    X ~ Uniform(-2, 2)
    α*(x) = 0.5 * sin(x)
    β*(x) = 1.0 + 0.5 * x
    θ*(x) = [α*(x), β*(x)]

    T = β*(x) + ξ,  where ξ ~ N(0, 0.5²)  [CONFOUNDED]
    Y ~ Bernoulli(σ(α*(x) + β*(x)·T))

Target: AME at t̃=0
    H(x, θ) = σ(α)·(1-σ(α))·β

True μ*:
    E[H(X, θ*(X))] = E[σ(0.5·sin(X))·(1-σ(0.5·sin(X)))·(1+0.5·X)]

    Since X ~ Uniform(-2, 2), we compute numerically:
    μ* ≈ 0.253 (via numerical integration)
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import torch
from torch import Tensor
from scipy import integrate
from scipy.special import expit  # sigmoid


@dataclass
class CanonicalDGP:
    """Configuration for the canonical DGP."""

    # True parameter functions
    A0: float = 0.0   # Intercept for alpha
    A1: float = 0.5   # sin coefficient for alpha
    B0: float = 1.0   # Intercept for beta
    B1: float = 0.5   # Linear coefficient for beta

    # Covariate distribution
    X_low: float = -2.0
    X_high: float = 2.0

    # Treatment noise
    T_noise_std: float = 0.5

    # Evaluation point
    t_tilde: float = 0.0

    def alpha_star(self, x: np.ndarray) -> np.ndarray:
        """True alpha(x) = 0.5 * sin(x)."""
        return self.A0 + self.A1 * np.sin(x)

    def beta_star(self, x: np.ndarray) -> np.ndarray:
        """True beta(x) = 1.0 + 0.5 * x."""
        return self.B0 + self.B1 * x

    def theta_star(self, x: np.ndarray) -> np.ndarray:
        """True θ*(x) = [α*(x), β*(x)]."""
        return np.column_stack([self.alpha_star(x), self.beta_star(x)])

    def prob(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """P(Y=1 | X=x, T=t) = σ(α*(x) + β*(x)·t)."""
        alpha = self.alpha_star(x)
        beta = self.beta_star(x)
        return expit(alpha + beta * t)

    def target_h(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Target functional: AME at t_tilde=0.

        H(x, θ) = σ(α)·(1-σ(α))·β

        At t_tilde=0, the AME is:
            ∂P/∂t |_{t=0} = σ'(α + β·0)·β = σ(α)·(1-σ(α))·β
        """
        alpha = theta[:, 0]
        beta = theta[:, 1]
        p = expit(alpha)  # σ(α) when t=0
        return p * (1 - p) * beta

    def mu_true(self) -> float:
        """
        True target: E[H(X, θ*(X))].

        Computed via numerical integration over X ~ Uniform(-2, 2).
        """
        def integrand(x):
            alpha = self.A0 + self.A1 * np.sin(x)
            beta = self.B0 + self.B1 * x
            p = expit(alpha)
            h = p * (1 - p) * beta
            # PDF of Uniform(-2, 2) is 1/4
            return h * 0.25

        result, _ = integrate.quad(integrand, self.X_low, self.X_high)
        return result


def generate_canonical_dgp(
    n: int,
    seed: Optional[int] = None,
    dgp: Optional[CanonicalDGP] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, float]:
    """
    Generate data from the canonical DGP.

    Args:
        n: Number of observations
        seed: Random seed
        dgp: DGP configuration (default: CanonicalDGP())

    Returns:
        Y: (n,) binary outcomes
        T: (n,) treatments (confounded!)
        X: (n, 1) covariates
        theta_true: (n, 2) true parameters [α*(x), β*(x)]
        mu_true: True target μ*
    """
    if dgp is None:
        dgp = CanonicalDGP()

    if seed is not None:
        np.random.seed(seed)

    # Generate X ~ Uniform(-2, 2)
    X = np.random.uniform(dgp.X_low, dgp.X_high, n)

    # True parameters
    alpha_true = dgp.alpha_star(X)
    beta_true = dgp.beta_star(X)
    theta_true = np.column_stack([alpha_true, beta_true])

    # Generate T = β*(X) + ξ  [CONFOUNDED!]
    xi = np.random.normal(0, dgp.T_noise_std, n)
    T = beta_true + xi

    # Generate Y ~ Bernoulli(σ(α*(X) + β*(X)·T))
    prob = dgp.prob(X, T)
    Y = np.random.binomial(1, prob).astype(float)

    # Compute true μ*
    mu_true = dgp.mu_true()

    # Convert to tensors
    Y_t = torch.tensor(Y, dtype=torch.float32)
    T_t = torch.tensor(T, dtype=torch.float32)
    X_t = torch.tensor(X.reshape(-1, 1), dtype=torch.float32)
    theta_true_t = torch.tensor(theta_true, dtype=torch.float32)

    return Y_t, T_t, X_t, theta_true_t, mu_true


# === ORACLE FORMULAS (for validation) ===

def oracle_score(y: float, t: float, theta: np.ndarray) -> np.ndarray:
    """
    Oracle score: ∂ℓ/∂θ for logistic loss.

    ℓ(y, t, θ) = -[y·log(p) + (1-y)·log(1-p)]
    where p = σ(α + β·t)

    ∂ℓ/∂α = p - y
    ∂ℓ/∂β = t·(p - y)

    Args:
        y: Outcome (0 or 1)
        t: Treatment
        theta: [alpha, beta]

    Returns:
        [∂ℓ/∂α, ∂ℓ/∂β]
    """
    alpha, beta = theta[0], theta[1]
    p = expit(alpha + beta * t)
    return np.array([p - y, t * (p - y)])


def oracle_hessian(y: float, t: float, theta: np.ndarray) -> np.ndarray:
    """
    Oracle Hessian: ∂²ℓ/∂θ² for logistic loss.

    ∂²ℓ/∂α² = p(1-p)
    ∂²ℓ/∂α∂β = t·p(1-p)
    ∂²ℓ/∂β² = t²·p(1-p)

    Args:
        y: Outcome (0 or 1) - NOTE: Hessian doesn't depend on y!
        t: Treatment
        theta: [alpha, beta]

    Returns:
        [[∂²ℓ/∂α², ∂²ℓ/∂α∂β],
         [∂²ℓ/∂β∂α, ∂²ℓ/∂β²]]
    """
    alpha, beta = theta[0], theta[1]
    p = expit(alpha + beta * t)
    v = p * (1 - p)
    return np.array([
        [v, t * v],
        [t * v, t * t * v]
    ])


def oracle_target_jacobian(theta: np.ndarray, t_tilde: float = 0.0) -> np.ndarray:
    """
    Oracle target Jacobian: ∂H/∂θ for AME at t_tilde.

    H(θ) = σ(α + β·t̃)·(1 - σ(α + β·t̃))·β

    Let s = σ(α + β·t̃), then H = s(1-s)β

    ∂H/∂α = β·s(1-s)(1-2s) = β·s·(1-s)·(1-2s)
    ∂H/∂β = s(1-s) + β·t̃·s(1-s)(1-2s)

    At t̃=0:
        ∂H/∂α = β·σ(α)·(1-σ(α))·(1-2σ(α))
        ∂H/∂β = σ(α)·(1-σ(α))

    Args:
        theta: [alpha, beta]
        t_tilde: Evaluation point (default: 0)

    Returns:
        [∂H/∂α, ∂H/∂β]
    """
    alpha, beta = theta[0], theta[1]
    s = expit(alpha + beta * t_tilde)
    dsdt = s * (1 - s)  # σ'(z)
    d2sdt2 = dsdt * (1 - 2 * s)  # σ''(z)

    # H = s(1-s)β = dsdt·β
    # ∂H/∂α = β·d(dsdt)/dα = β·d2sdt2
    # ∂H/∂β = dsdt + β·t̃·d2sdt2

    dH_dalpha = beta * d2sdt2
    dH_dbeta = dsdt + beta * t_tilde * d2sdt2

    return np.array([dH_dalpha, dH_dbeta])


def oracle_lambda_conditional(x: float, dgp: CanonicalDGP, n_samples: int = 10000) -> np.ndarray:
    """
    Oracle Λ(x) = E[ℓ_θθ | X=x] via Monte Carlo integration.

    Since T | X ~ N(β*(x), 0.5²) in our DGP,
    Λ(x) = E_T[p(1-p)·[[1, T], [T, T²]] | X=x]

    where p = σ(α*(x) + β*(x)·T)

    Args:
        x: Covariate value
        dgp: DGP configuration
        n_samples: MC samples

    Returns:
        (2, 2) conditional Hessian matrix
    """
    # True parameters at x
    alpha_x = dgp.alpha_star(np.array([x]))[0]
    beta_x = dgp.beta_star(np.array([x]))[0]

    # Sample T | X ~ N(β*(x), 0.5²)
    T_samples = np.random.normal(beta_x, dgp.T_noise_std, n_samples)

    # Compute Hessians and average
    Lambda = np.zeros((2, 2))
    for t in T_samples:
        H = oracle_hessian(0, t, np.array([alpha_x, beta_x]))  # y doesn't matter
        Lambda += H
    Lambda /= n_samples

    return Lambda


if __name__ == "__main__":
    # Test DGP generation
    dgp = CanonicalDGP()
    print("=== Canonical DGP Configuration ===")
    print(f"α*(x) = {dgp.A0} + {dgp.A1}·sin(x)")
    print(f"β*(x) = {dgp.B0} + {dgp.B1}·x")
    print(f"X ~ Uniform({dgp.X_low}, {dgp.X_high})")
    print(f"T = β*(x) + N(0, {dgp.T_noise_std}²)")
    print(f"t̃ = {dgp.t_tilde}")
    print(f"\nTrue μ* = {dgp.mu_true():.6f}")

    # Generate sample
    Y, T, X, theta_true, mu_true = generate_canonical_dgp(n=1000, seed=42)
    print(f"\n=== Sample Statistics (n=1000) ===")
    print(f"Y: mean={Y.mean():.4f}, std={Y.std():.4f}")
    print(f"T: mean={T.mean():.4f}, std={T.std():.4f}")
    print(f"X: mean={X.mean():.4f}, std={X.std():.4f}")
    print(f"α*: mean={theta_true[:, 0].mean():.4f}, std={theta_true[:, 0].std():.4f}")
    print(f"β*: mean={theta_true[:, 1].mean():.4f}, std={theta_true[:, 1].std():.4f}")

    # Test oracle functions
    print(f"\n=== Oracle Functions (single point) ===")
    y_test, t_test = 1.0, 0.5
    theta_test = np.array([0.1, 2.0])

    print(f"Test: y={y_test}, t={t_test}, θ={theta_test}")
    print(f"Score: {oracle_score(y_test, t_test, theta_test)}")
    print(f"Hessian:\n{oracle_hessian(y_test, t_test, theta_test)}")
    print(f"Target Jacobian: {oracle_target_jacobian(theta_test, t_tilde=0.0)}")

    # Test conditional Lambda
    print(f"\n=== Oracle Lambda (x=0) ===")
    Lambda_0 = oracle_lambda_conditional(0.0, dgp, n_samples=10000)
    print(f"Λ(0) =\n{Lambda_0}")
    print(f"Eigenvalues: {np.linalg.eigvalsh(Lambda_0)}")
