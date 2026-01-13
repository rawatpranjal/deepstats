"""
Regime B DGP: Confounded Linear Model

Validates the AnalyticLambda strategy (Λ = E[TT'|X]).

DGP Specification:
    X ~ Uniform(0, 1)
    α*(x) = x²           # Nonlinear intercept (network must capture!)
    β*(x) = 2x - 1       # Linear CATE
    T = 0.5X + ξ,  where ξ ~ N(0, 1)  [CONFOUNDED]
    Y = α*(x) + β*(x)·T + ε,  where ε ~ N(0, 0.1²)

Target: ATE = E[β*(X)] = E[2X-1] = 0 (since E[X]=0.5)

Key Properties:
    - Treatment is confounded (depends on X)
    - Hessian is CONSTANT (doesn't depend on θ!)
    - Λ(x) = E[TT'|X] has closed form
    - Linear loss → Robinson 1988 closed-form ψ

Why This DGP:
    1. Nonlinear α*(x) = x² tests network flexibility
    2. Confounded T tests proper Lambda estimation
    3. Known closed-form ψ enables exact validation
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import torch
from torch import Tensor


@dataclass
class LinearDGP:
    """Configuration for the Linear DGP."""

    # Covariate distribution
    X_low: float = 0.0
    X_high: float = 1.0

    # Treatment confounding: T = c*X + noise
    T_confound: float = 0.5  # Coefficient on X
    T_noise_std: float = 1.0

    # Outcome noise
    Y_noise_std: float = 0.1

    def alpha_star(self, x: np.ndarray) -> np.ndarray:
        """True α*(x) = x²."""
        return x ** 2

    def beta_star(self, x: np.ndarray) -> np.ndarray:
        """True β*(x) = 2x - 1."""
        return 2 * x - 1

    def theta_star(self, x: np.ndarray) -> np.ndarray:
        """True θ*(x) = [α*(x), β*(x)]."""
        return np.column_stack([self.alpha_star(x), self.beta_star(x)])

    def E_T_given_X(self, x: np.ndarray) -> np.ndarray:
        """E[T|X] = 0.5X."""
        return self.T_confound * x

    def Var_T_given_X(self) -> float:
        """Var(T|X) = 1."""
        return self.T_noise_std ** 2

    def mu_true(self) -> float:
        """
        True target: ATE = E[β*(X)] = E[2X-1].

        Since X ~ Uniform(0, 1), E[X] = 0.5
        → ATE = 2 * 0.5 - 1 = 0
        """
        E_X = (self.X_low + self.X_high) / 2
        return 2 * E_X - 1  # = 0


def generate_linear_data(
    n: int,
    seed: Optional[int] = None,
    dgp: Optional[LinearDGP] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, float]:
    """
    Generate data from the Linear DGP.

    Args:
        n: Number of observations
        seed: Random seed
        dgp: DGP configuration (default: LinearDGP())

    Returns:
        Y: (n,) continuous outcomes
        T: (n,) continuous treatments (confounded!)
        X: (n, 1) covariates
        theta_true: (n, 2) true parameters [α*(x), β*(x)]
        mu_true: True target ATE
    """
    if dgp is None:
        dgp = LinearDGP()

    if seed is not None:
        np.random.seed(seed)

    # Generate X ~ Uniform(0, 1)
    X = np.random.uniform(dgp.X_low, dgp.X_high, n)

    # True parameters
    alpha_true = dgp.alpha_star(X)
    beta_true = dgp.beta_star(X)
    theta_true = np.column_stack([alpha_true, beta_true])

    # Generate T = 0.5X + ξ  [CONFOUNDED!]
    xi = np.random.normal(0, dgp.T_noise_std, n)
    T = dgp.T_confound * X + xi

    # Generate Y = α*(X) + β*(X)·T + ε
    epsilon = np.random.normal(0, dgp.Y_noise_std, n)
    Y = alpha_true + beta_true * T + epsilon

    # Compute true μ*
    mu_true = dgp.mu_true()

    # Convert to tensors
    Y_t = torch.tensor(Y, dtype=torch.float32)
    T_t = torch.tensor(T, dtype=torch.float32)
    X_t = torch.tensor(X.reshape(-1, 1), dtype=torch.float32)
    theta_true_t = torch.tensor(theta_true, dtype=torch.float32)

    return Y_t, T_t, X_t, theta_true_t, mu_true


# === ORACLE FORMULAS ===

def oracle_lambda_linear(x: float, dgp: LinearDGP) -> np.ndarray:
    """
    Oracle Λ(x) = E[TT'|X] for linear model.

    With design vector [1, T]:
        Λ(x) = E[[1,T][1,T]' | X]
             = [[1, E[T|X]], [E[T|X], E[T²|X]]]

    Where:
        E[T|X] = 0.5X
        Var(T|X) = 1
        E[T²|X] = Var + Mean² = 1 + 0.25X²

    Args:
        x: Covariate value
        dgp: DGP configuration

    Returns:
        (2, 2) Lambda matrix
    """
    ET = dgp.T_confound * x
    VarT = dgp.Var_T_given_X()
    ET2 = VarT + ET ** 2

    return np.array([
        [1.0, ET],
        [ET, ET2]
    ])


def oracle_score_linear(y: float, t: float, theta: np.ndarray) -> np.ndarray:
    """
    Oracle score for linear loss.

    ℓ(y, t, θ) = 0.5 * (y - α - β*t)²
    ℓ_θ = -(y - α - β*t) * [1, t]
        = (α + β*t - y) * [1, t]
    """
    alpha, beta = theta[0], theta[1]
    residual = alpha + beta * t - y
    return np.array([residual, t * residual])


def oracle_hessian_linear(t: float) -> np.ndarray:
    """
    Oracle Hessian for linear loss.

    ℓ_θθ = [1, t][1, t]' = [[1, t], [t, t²]]

    KEY: Does NOT depend on θ or y! (This is what makes it Regime B)
    """
    return np.array([
        [1.0, t],
        [t, t * t]
    ])


def oracle_psi_robinson(
    y: float,
    t: float,
    x: float,
    theta: np.ndarray,
    dgp: LinearDGP,
) -> float:
    """
    Robinson 1988 closed-form influence function for linear ATE.

    ψ* = β*(x) + (T - E[T|X]) * (Y - α*(x) - β*(x)*T) / Var(T|X)

    This is the "efficient" semiparametric influence function.
    The generic formula ψ = H - J·Λ⁻¹·S should simplify to this.

    Args:
        y: Outcome
        t: Treatment
        x: Covariate
        theta: [alpha, beta] - true parameters at x
        dgp: DGP configuration

    Returns:
        Influence function value
    """
    alpha, beta = theta[0], theta[1]

    # E[T|X] and Var(T|X)
    ET = dgp.E_T_given_X(np.array([x]))[0]
    VarT = dgp.Var_T_given_X()

    # Residuals
    T_resid = t - ET
    Y_resid = y - alpha - beta * t  # Note: Using fitted, not Y - E[Y|X]

    # Robinson form
    psi = beta + T_resid * Y_resid / VarT

    return psi


if __name__ == "__main__":
    dgp = LinearDGP()
    print("=== Regime B: Linear DGP ===")
    print(f"α*(x) = x²")
    print(f"β*(x) = 2x - 1")
    print(f"X ~ Uniform({dgp.X_low}, {dgp.X_high})")
    print(f"T = {dgp.T_confound}X + N(0, {dgp.T_noise_std}²)")
    print(f"Y = α*(x) + β*(x)·T + N(0, {dgp.Y_noise_std}²)")
    print(f"\nTrue ATE = {dgp.mu_true():.6f}")

    # Generate sample
    Y, T, X, theta_true, mu_true = generate_linear_data(n=1000, seed=42)
    print(f"\n=== Sample Statistics (n=1000) ===")
    print(f"Y: mean={Y.mean():.4f}, std={Y.std():.4f}")
    print(f"T: mean={T.mean():.4f}, std={T.std():.4f}")
    print(f"X: mean={X.mean():.4f}, range=[{X.min():.2f}, {X.max():.2f}]")
    print(f"α*: mean={theta_true[:, 0].mean():.4f}")
    print(f"β*: mean={theta_true[:, 1].mean():.4f}")

    # Test oracle Lambda at x=0.5
    print(f"\n=== Oracle Lambda (x=0.5) ===")
    Lambda = oracle_lambda_linear(0.5, dgp)
    print(f"Λ(0.5) =\n{Lambda}")
    print(f"Eigenvalues: {np.linalg.eigvalsh(Lambda)}")

    # Test Robinson psi
    print(f"\n=== Robinson Psi (single point) ===")
    idx = 0
    x_test = X[idx].item()
    theta_test = theta_true[idx].numpy()
    psi = oracle_psi_robinson(Y[idx].item(), T[idx].item(), x_test, theta_test, dgp)
    print(f"x={x_test:.4f}, θ={theta_test}, ψ={psi:.6f}")
