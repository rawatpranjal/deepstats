"""
Regime A DGP: Randomized Controlled Trial with Logit Model

Validates the ComputeLambda strategy (Monte Carlo integration over known F_T).

DGP Specification:
    X ~ Uniform(-1, 1)
    α*(x) = x             # Linear intercept
    β*(x) = 1             # Constant treatment effect
    T ~ Bernoulli(0.5)    # RANDOMIZED - independent of X!
    Y ~ Bernoulli(σ(α*(x) + β*(x)·T))

Target: ATE = E[σ(x+1) - σ(x)]
    = ∫₋₁¹ [σ(x+1) - σ(x)] · (1/2) dx
    ≈ 0.231 (via numerical integration)

Key Properties:
    - Treatment is randomized (not confounded)
    - Hessian depends on θ but NOT on Y
    - Known treatment distribution F_T = Bernoulli(0.5)
    - Enables 2-way cross-fitting (no need for 3-way)
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import torch
from torch import Tensor
from scipy import integrate
from scipy.special import expit  # sigmoid


@dataclass
class RCTLogitDGP:
    """Configuration for the RCT Logit DGP."""

    # True parameter functions (simple for clean validation)
    # α*(x) = x, β*(x) = 1

    # Covariate distribution
    X_low: float = -1.0
    X_high: float = 1.0

    # Treatment distribution (Bernoulli)
    p_treat: float = 0.5

    def alpha_star(self, x: np.ndarray) -> np.ndarray:
        """True α*(x) = x."""
        return x

    def beta_star(self, x: np.ndarray) -> np.ndarray:
        """True β*(x) = 1 (constant)."""
        return np.ones_like(x)

    def theta_star(self, x: np.ndarray) -> np.ndarray:
        """True θ*(x) = [α*(x), β*(x)]."""
        return np.column_stack([self.alpha_star(x), self.beta_star(x)])

    def prob(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """P(Y=1 | X=x, T=t) = σ(α*(x) + β*(x)·t)."""
        alpha = self.alpha_star(x)
        beta = self.beta_star(x)
        return expit(alpha + beta * t)

    def mu_true(self) -> float:
        """
        True target: ATE = E[σ(α*(X)+1) - σ(α*(X))].

        Since β*(x) = 1:
            ATE = E[σ(x+1) - σ(x)] over X ~ Uniform(-1, 1)
        """
        def integrand(x):
            p1 = expit(x + 1)  # P(Y=1|X=x, T=1)
            p0 = expit(x)      # P(Y=1|X=x, T=0)
            # PDF of Uniform(-1, 1) is 1/2
            return (p1 - p0) * 0.5

        result, _ = integrate.quad(integrand, self.X_low, self.X_high)
        return result


def generate_rct_logit_data(
    n: int,
    seed: Optional[int] = None,
    dgp: Optional[RCTLogitDGP] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, float]:
    """
    Generate data from the RCT Logit DGP.

    Args:
        n: Number of observations
        seed: Random seed
        dgp: DGP configuration (default: RCTLogitDGP())

    Returns:
        Y: (n,) binary outcomes
        T: (n,) binary treatments (randomized!)
        X: (n, 1) covariates
        theta_true: (n, 2) true parameters [α*(x), β*(x)]
        mu_true: True target ATE
    """
    if dgp is None:
        dgp = RCTLogitDGP()

    if seed is not None:
        np.random.seed(seed)

    # Generate X ~ Uniform(-1, 1)
    X = np.random.uniform(dgp.X_low, dgp.X_high, n)

    # True parameters
    alpha_true = dgp.alpha_star(X)
    beta_true = dgp.beta_star(X)
    theta_true = np.column_stack([alpha_true, beta_true])

    # Generate T ~ Bernoulli(0.5) - RANDOMIZED!
    T = np.random.binomial(1, dgp.p_treat, n).astype(float)

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


# === ORACLE FORMULAS ===

def oracle_lambda_rct(x: float, theta: np.ndarray, p_treat: float = 0.5) -> np.ndarray:
    """
    Oracle Λ(x) for RCT Logit.

    Since T ~ Bernoulli(p_treat), we compute:
        Λ(x) = E_T[σ(1-σ)·TT']
             = p_treat · σ₁(1-σ₁)·[[1,1],[1,1]] + (1-p_treat) · σ₀(1-σ₀)·[[1,0],[0,0]]

    where σ₀ = σ(α) and σ₁ = σ(α+β)

    For the standard outer product form with [1, T]:
        T=0: [1,0][1,0]' = [[1,0],[0,0]]
        T=1: [1,1][1,1]' = [[1,1],[1,1]]

    Args:
        x: Covariate value (not used for β=1, but kept for API consistency)
        theta: [alpha, beta]
        p_treat: Treatment probability

    Returns:
        (2, 2) Lambda matrix
    """
    alpha, beta = theta[0], theta[1]

    # Probabilities at T=0 and T=1
    p0 = expit(alpha)
    p1 = expit(alpha + beta)

    # Information at T=0 and T=1
    v0 = p0 * (1 - p0)
    v1 = p1 * (1 - p1)

    # Outer products for [1, T] basis
    L0 = v0 * np.array([[1, 0], [0, 0]])  # T=0: [1,0][1,0]'
    L1 = v1 * np.array([[1, 1], [1, 1]])  # T=1: [1,1][1,1]'

    # Expected Lambda
    Lambda = (1 - p_treat) * L0 + p_treat * L1

    return Lambda


def oracle_score_logit(y: float, t: float, theta: np.ndarray) -> np.ndarray:
    """
    Oracle score for logit loss.

    ℓ_θ = [p-y, t(p-y)]
    """
    alpha, beta = theta[0], theta[1]
    p = expit(alpha + beta * t)
    return np.array([p - y, t * (p - y)])


def oracle_hessian_logit(t: float, theta: np.ndarray) -> np.ndarray:
    """
    Oracle Hessian for logit loss.

    ℓ_θθ = p(1-p) · [[1, t], [t, t²]]

    Note: Does NOT depend on y (key for Regime A)
    """
    alpha, beta = theta[0], theta[1]
    p = expit(alpha + beta * t)
    v = p * (1 - p)
    return v * np.array([[1, t], [t, t * t]])


if __name__ == "__main__":
    dgp = RCTLogitDGP()
    print("=== Regime A: RCT Logit DGP ===")
    print(f"α*(x) = x")
    print(f"β*(x) = 1 (constant)")
    print(f"X ~ Uniform({dgp.X_low}, {dgp.X_high})")
    print(f"T ~ Bernoulli({dgp.p_treat})")
    print(f"\nTrue ATE = {dgp.mu_true():.6f}")

    # Generate sample
    Y, T, X, theta_true, mu_true = generate_rct_logit_data(n=1000, seed=42)
    print(f"\n=== Sample Statistics (n=1000) ===")
    print(f"Y: mean={Y.mean():.4f}, P(Y=1)={Y.mean():.3f}")
    print(f"T: mean={T.mean():.4f} (should be ~{dgp.p_treat})")
    print(f"X: mean={X.mean():.4f}, range=[{X.min():.2f}, {X.max():.2f}]")

    # Test oracle Lambda
    print(f"\n=== Oracle Lambda (x=0, θ=[0,1]) ===")
    theta_test = np.array([0.0, 1.0])
    Lambda = oracle_lambda_rct(0.0, theta_test, dgp.p_treat)
    print(f"Λ =\n{Lambda}")
    print(f"Eigenvalues: {np.linalg.eigvalsh(Lambda)}")
