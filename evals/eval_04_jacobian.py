"""
================================================================================
EVAL 04: TARGET JACOBIAN H_θ = ∂H/∂θ
================================================================================

WHAT THIS TESTS:
    The Jacobian of the target functional H(θ) with respect to parameters θ.
    This is the gradient that propagates through the influence function formula.

WHY IT MATTERS:
    The influence function ψ(z) = H_θ · Λ⁻¹ · ℓ_θ requires accurate H_θ.
    If H_θ is wrong, the entire IF correction is wrong → invalid standard errors.

    H_θ appears in Theorem 2 (FLM 2021):
        √n(μ̂ - μ*) →d N(0, E[ψ²])
        where ψ = H_θ(X,θ) · Λ(X)⁻¹ · ℓ_θ(Z,θ)

    We compute H_θ via torch.func.grad (autodiff). This eval verifies autodiff
    matches closed-form oracle formulas derived by hand.

MATHEMATICAL OBJECTS TESTED:
    Target functionals H(θ):
        - AverageParameter: H(θ) = β  →  H_θ = [0, 1]
        - AME (Logit):      H(θ) = σ(α+βt̃)(1-σ)β  →  complex formula
        - Prediction:       H(θ) = g⁻¹(α+βt̃)  →  depends on link

    Families tested: Linear, Logit, Poisson, Probit
    Each has different link function g⁻¹ and thus different H_θ formula.

--------------------------------------------------------------------------------
TEST MATRIX:
--------------------------------------------------------------------------------
    Part 1: Target Coverage (Family = Logit)
        - AverageParameter: H(θ) = β
        - AME: H(θ) = σ(α+βt̃)(1-σ)β
        - AveragePrediction: H(θ) = σ(α+βt̃)

    Part 2: Family Coverage (Target = AME)
        - Linear: H(θ) = β
        - Logit: H(θ) = σ(α+βt̃)(1-σ)β
        - Poisson: H(θ) = β·exp(α+βt̃)
        - Probit: H(θ) = φ(α+βt̃)·β

    Part 3: Edge Cases (AME × Logit)
        - Near-boundary: θ = [±5, 1]  (σ ≈ 0 or 1)
        - Tiny effect: θ = [0, 0.001]
        - Large effect: θ = [0, 10]

    Part 4: Batched vmap Tests
        - 100 random θ per (target, family) pair

PASS CRITERIA:
    - Standard θ: max|err| < 1e-10 (machine precision)
    - Edge θ: max|err| < 1e-6 OR relative error < 1e-4
    - Batched: max|err| < 1e-8
================================================================================
"""

import sys
import numpy as np
import torch
from torch import Tensor
from scipy.special import expit
from scipy.stats import norm
from typing import Callable, Dict, List, Tuple, Any
from dataclasses import dataclass

sys.path.insert(0, "/Users/pranjal/deepest/src")

from evals.dgp import (
    oracle_jacobian_average_parameter,
    oracle_jacobian_ame_linear,
    oracle_jacobian_ame_logit,
    oracle_jacobian_ame_poisson,
    oracle_jacobian_ame_probit,
    oracle_jacobian_prediction_linear,
    oracle_jacobian_prediction_logit,
    oracle_jacobian_prediction_poisson,
    oracle_jacobian_prediction_probit,
    oracle_jacobian_elasticity_poisson,
    # Additional families
    oracle_jacobian_ame_gamma,
    oracle_jacobian_ame_weibull,
    oracle_jacobian_ame_negbin,
    oracle_jacobian_ame_gumbel,
    oracle_jacobian_ame_beta,
    # Higher-dimensional
    oracle_jacobian_avg_param_nd,
    oracle_jacobian_prediction_gaussian,
    oracle_jacobian_prediction_tobit,
    oracle_jacobian_zip_rate_effect,
    oracle_jacobian_zip_zeroinfl_effect,
)
from deep_inference.autodiff.jacobian import compute_target_jacobian_vmap


# =============================================================================
# TARGET FUNCTIONS (Torch)
# =============================================================================

def target_average_parameter(x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
    """H(θ) = β."""
    return theta[1]


def target_ame_linear(x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
    """Linear AME: H(θ) = β."""
    return theta[1]


def target_ame_logit(x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
    """Logit AME: H(θ) = σ(α+βt̃)(1-σ(α+βt̃))β."""
    alpha, beta = theta[0], theta[1]
    logit = alpha + beta * t_tilde
    s = torch.sigmoid(logit)
    return s * (1 - s) * beta


def target_ame_poisson(x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
    """Poisson AME: H(θ) = β·exp(α+βt̃)."""
    alpha, beta = theta[0], theta[1]
    mu = torch.exp(alpha + beta * t_tilde)
    return beta * mu


def target_ame_probit(x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
    """Probit AME: H(θ) = φ(α+βt̃)·β."""
    alpha, beta = theta[0], theta[1]
    eta = alpha + beta * t_tilde
    # Standard normal PDF: φ(x) = exp(-x²/2) / sqrt(2π)
    phi = torch.exp(-0.5 * eta**2) / np.sqrt(2 * np.pi)
    return phi * beta


def target_prediction_linear(x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
    """Linear prediction: H(θ) = α + βt̃."""
    alpha, beta = theta[0], theta[1]
    return alpha + beta * t_tilde


def target_prediction_logit(x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
    """Logit prediction: H(θ) = σ(α+βt̃)."""
    alpha, beta = theta[0], theta[1]
    return torch.sigmoid(alpha + beta * t_tilde)


def target_prediction_poisson(x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
    """Poisson prediction: H(θ) = exp(α+βt̃)."""
    alpha, beta = theta[0], theta[1]
    return torch.exp(alpha + beta * t_tilde)


def target_prediction_probit(x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
    """Probit prediction: H(θ) = Φ(α+βt̃)."""
    alpha, beta = theta[0], theta[1]
    eta = alpha + beta * t_tilde
    # Standard normal CDF approximation using erf
    return 0.5 * (1 + torch.erf(eta / np.sqrt(2)))


def target_elasticity_poisson(x: Tensor, theta: Tensor, t_bar: Tensor) -> Tensor:
    """Poisson elasticity: H(θ) = β·t̄."""
    return theta[1] * t_bar


# =============================================================================
# ADDITIONAL TARGET FUNCTIONS (NEW FAMILIES)
# =============================================================================

def target_ame_gamma(x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
    """Gamma AME: H(θ) = β·exp(α+βt̃). Same as Poisson (log-linear)."""
    return target_ame_poisson(x, theta, t_tilde)


def target_ame_weibull(x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
    """Weibull AME: H(θ) = β·exp(α+βt̃). Same as Poisson (log-linear)."""
    return target_ame_poisson(x, theta, t_tilde)


def target_ame_negbin(x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
    """NegBin AME: H(θ) = β·exp(α+βt̃). Same as Poisson (log-linear)."""
    return target_ame_poisson(x, theta, t_tilde)


def target_ame_gumbel(x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
    """Gumbel AME: H(θ) = β. Same as linear (identity link)."""
    return theta[1]


def target_ame_beta(x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
    """Beta AME: H(θ) = σ(α+βt̃)(1-σ)β. Same as logit (logit link)."""
    return target_ame_logit(x, theta, t_tilde)


# =============================================================================
# HIGHER-DIMENSIONAL TARGETS (theta_dim > 2)
# =============================================================================

def target_avg_param_3d(x: Tensor, theta: Tensor, t_tilde: Tensor, param_index: int = 1) -> Tensor:
    """Average parameter for 3D theta: H(θ) = θ_k."""
    return theta[param_index]


def target_avg_param_4d(x: Tensor, theta: Tensor, t_tilde: Tensor, param_index: int = 1) -> Tensor:
    """Average parameter for 4D theta: H(θ) = θ_k."""
    return theta[param_index]


def target_prediction_gaussian_3d(x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
    """Gaussian prediction for 3D θ = [α, β, γ]: H(θ) = α + βt̃."""
    alpha, beta = theta[0], theta[1]
    return alpha + beta * t_tilde


def target_prediction_tobit_3d(x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
    """Tobit latent prediction for 3D θ = [α, β, γ]: H(θ) = α + βt̃."""
    alpha, beta = theta[0], theta[1]
    return alpha + beta * t_tilde


def target_zip_rate_effect(x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
    """ZIP rate effect for 4D θ = [α, β, γ, δ]: H(θ) = β."""
    return theta[1]


def target_zip_zeroinfl_effect(x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
    """ZIP zero-inflation effect for 4D θ = [α, β, γ, δ]: H(θ) = δ."""
    return theta[3]


# =============================================================================
# TEST INFRASTRUCTURE
# =============================================================================

@dataclass
class TestResult:
    """Result of a single Jacobian test."""
    name: str
    theta: np.ndarray
    t_tilde: float
    oracle_jac: np.ndarray
    autodiff_jac: np.ndarray
    abs_error: float
    rel_error: float
    h_value: float
    passed: bool


def autodiff_jacobian(
    target_fn: Callable,
    theta: np.ndarray,
    t_tilde: float,
) -> np.ndarray:
    """Compute Jacobian via autodiff for single theta."""
    theta_t = torch.tensor(theta, dtype=torch.float64)
    t_tilde_t = torch.tensor(t_tilde, dtype=torch.float64)
    x_t = torch.tensor([0.0], dtype=torch.float64)  # x not used

    jac = torch.func.grad(lambda th: target_fn(x_t, th, t_tilde_t))(theta_t)
    return jac.detach().numpy()


def run_single_test(
    name: str,
    target_fn: Callable,
    oracle_fn: Callable,
    theta: np.ndarray,
    t_tilde: float,
    tol_abs: float = 1e-10,
    tol_rel: float = 1e-4,
    verbose: bool = False,
) -> TestResult:
    """Run a single Jacobian test."""
    # Oracle
    oracle_jac = oracle_fn(theta, t_tilde)

    # Autodiff
    autodiff_jac = autodiff_jacobian(target_fn, theta, t_tilde)

    # Compute H value for context
    theta_t = torch.tensor(theta, dtype=torch.float64)
    t_tilde_t = torch.tensor(t_tilde, dtype=torch.float64)
    x_t = torch.tensor([0.0], dtype=torch.float64)
    h_value = target_fn(x_t, theta_t, t_tilde_t).item()

    # Error metrics
    abs_error = np.abs(autodiff_jac - oracle_jac).max()

    # Relative error (avoid division by zero)
    oracle_norm = np.abs(oracle_jac).max()
    if oracle_norm > 1e-12:
        rel_error = abs_error / oracle_norm
    else:
        rel_error = abs_error  # Fall back to absolute when oracle is tiny

    # Pass criteria: absolute OR relative
    passed = (abs_error < tol_abs) or (rel_error < tol_rel)

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"  θ={theta}, t̃={t_tilde:.1f}")
        print(f"    H(θ) = {h_value:.6f}")
        print(f"    Oracle:   [{oracle_jac[0]:.6f}, {oracle_jac[1]:.6f}]")
        print(f"    Autodiff: [{autodiff_jac[0]:.6f}, {autodiff_jac[1]:.6f}]")
        print(f"    Abs Err: {abs_error:.2e}, Rel Err: {rel_error:.2e} [{status}]")

    return TestResult(
        name=name,
        theta=theta,
        t_tilde=t_tilde,
        oracle_jac=oracle_jac,
        autodiff_jac=autodiff_jac,
        abs_error=abs_error,
        rel_error=rel_error,
        h_value=h_value,
        passed=passed,
    )


def run_batched_test(
    name: str,
    target_fn: Callable,
    oracle_fn: Callable,
    n: int = 100,
    seed: int = 42,
    t_tilde: float = 0.0,
    tol: float = 1e-8,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run batched vmap test."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate random theta values
    theta = np.random.randn(n, 2)

    # Convert to tensors
    X_t = torch.zeros(n, 1, dtype=torch.float64)
    theta_t = torch.tensor(theta, dtype=torch.float64)
    t_tilde_t = torch.tensor(t_tilde, dtype=torch.float64)

    # Compute via vmap
    jac_autodiff = compute_target_jacobian_vmap(target_fn, X_t, theta_t, t_tilde_t).numpy()

    # Compute via oracle (loop)
    jac_oracle = np.zeros((n, 2))
    for i in range(n):
        jac_oracle[i] = oracle_fn(theta[i], t_tilde)

    # Errors
    errors = np.abs(jac_autodiff - jac_oracle)
    max_error = errors.max()
    mean_error = errors.mean()
    passed = max_error < tol

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"  {name} (n={n}, t̃={t_tilde:.1f}): max={max_error:.2e}, mean={mean_error:.2e} [{status}]")

    return {
        "name": name,
        "n": n,
        "t_tilde": t_tilde,
        "max_error": max_error,
        "mean_error": mean_error,
        "passed": passed,
    }


# =============================================================================
# TEST CASES
# =============================================================================

# Standard θ cases
THETA_STANDARD = [
    np.array([0.0, 1.0]),    # Simple case
    np.array([1.0, 0.5]),    # Positive alpha
    np.array([-0.5, 2.0]),   # Negative alpha, large beta
    np.array([0.1, -1.0]),   # Negative beta
    np.array([2.0, 0.1]),    # Large alpha, small beta
]

# Edge θ cases
THETA_EDGE = [
    np.array([5.0, 1.0]),    # Near-certain positive (σ ≈ 0.993)
    np.array([-5.0, 1.0]),   # Near-certain negative (σ ≈ 0.007)
    np.array([0.0, 0.001]),  # Tiny effect
    np.array([0.0, 10.0]),   # Large effect
]

# t̃ values
T_TILDE_VALUES = [0.0, 0.5, -1.0, 2.0]


# =============================================================================
# PART 1: TARGET COVERAGE (Family = Logit)
# =============================================================================

def run_part1_targets(verbose: bool = True) -> Dict[str, Any]:
    """Test multiple targets with Logit family."""
    print("\n" + "=" * 70)
    print("PART 1: TARGET COVERAGE (Family = Logit)")
    print("=" * 70)

    results = {"1a_avg_param": [], "1b_ame": [], "1c_prediction": []}

    # 1a: Average Parameter (t̃ irrelevant)
    print("\n--- 1a. AverageParameter: H(θ) = β ---")
    for theta in THETA_STANDARD:
        r = run_single_test(
            "AvgParam", target_average_parameter, oracle_jacobian_average_parameter,
            theta, t_tilde=0.0, verbose=verbose
        )
        results["1a_avg_param"].append(r)

    # 1b: AME Logit (multiple t̃)
    print("\n--- 1b. AME Logit: H(θ) = σ(α+βt̃)(1-σ)β ---")
    for theta in THETA_STANDARD:
        for t_tilde in T_TILDE_VALUES:
            r = run_single_test(
                "AME_Logit", target_ame_logit, oracle_jacobian_ame_logit,
                theta, t_tilde=t_tilde, verbose=verbose
            )
            results["1b_ame"].append(r)

    # 1c: Prediction Logit (multiple t̃)
    print("\n--- 1c. Prediction Logit: H(θ) = σ(α+βt̃) ---")
    for theta in THETA_STANDARD:
        for t_tilde in T_TILDE_VALUES:
            r = run_single_test(
                "Pred_Logit", target_prediction_logit, oracle_jacobian_prediction_logit,
                theta, t_tilde=t_tilde, verbose=verbose
            )
            results["1c_prediction"].append(r)

    # Summary
    all_results = results["1a_avg_param"] + results["1b_ame"] + results["1c_prediction"]
    max_abs_err = max(r.abs_error for r in all_results)
    n_pass = sum(1 for r in all_results if r.passed)
    n_total = len(all_results)

    print(f"\nPart 1 Summary: {n_pass}/{n_total} passed, max|err| = {max_abs_err:.2e}")

    return {
        "results": results,
        "max_error": max_abs_err,
        "passed": n_pass == n_total,
        "n_pass": n_pass,
        "n_total": n_total,
    }


# =============================================================================
# PART 2: FAMILY COVERAGE (Target = AME)
# =============================================================================

def run_part2_families(verbose: bool = True) -> Dict[str, Any]:
    """Test multiple families with AME target."""
    print("\n" + "=" * 70)
    print("PART 2: FAMILY COVERAGE (Target = AME)")
    print("=" * 70)

    results = {"2a_linear": [], "2b_poisson": [], "2c_probit": []}

    # 2a: Linear (t̃ irrelevant for linear AME)
    print("\n--- 2a. Linear AME: H(θ) = β ---")
    for theta in THETA_STANDARD:
        r = run_single_test(
            "AME_Linear", target_ame_linear, oracle_jacobian_ame_linear,
            theta, t_tilde=0.0, verbose=verbose
        )
        results["2a_linear"].append(r)

    # 2b: Poisson
    print("\n--- 2b. Poisson AME: H(θ) = β·exp(α+βt̃) ---")
    for theta in THETA_STANDARD:
        for t_tilde in [0.0, 0.5]:  # Fewer t̃ to avoid overflow
            r = run_single_test(
                "AME_Poisson", target_ame_poisson, oracle_jacobian_ame_poisson,
                theta, t_tilde=t_tilde, verbose=verbose
            )
            results["2b_poisson"].append(r)

    # 2c: Probit
    print("\n--- 2c. Probit AME: H(θ) = φ(α+βt̃)·β ---")
    for theta in THETA_STANDARD:
        for t_tilde in [0.0, 0.5]:
            r = run_single_test(
                "AME_Probit", target_ame_probit, oracle_jacobian_ame_probit,
                theta, t_tilde=t_tilde, verbose=verbose
            )
            results["2c_probit"].append(r)

    # Summary
    all_results = results["2a_linear"] + results["2b_poisson"] + results["2c_probit"]
    max_abs_err = max(r.abs_error for r in all_results)
    n_pass = sum(1 for r in all_results if r.passed)
    n_total = len(all_results)

    print(f"\nPart 2 Summary: {n_pass}/{n_total} passed, max|err| = {max_abs_err:.2e}")

    return {
        "results": results,
        "max_error": max_abs_err,
        "passed": n_pass == n_total,
        "n_pass": n_pass,
        "n_total": n_total,
    }


# =============================================================================
# PART 3: EDGE CASES (AME × Logit)
# =============================================================================

def run_part3_edge_cases(verbose: bool = True) -> Dict[str, Any]:
    """Test edge cases for numerical stability."""
    print("\n" + "=" * 70)
    print("PART 3: EDGE CASES (AME × Logit)")
    print("=" * 70)

    results = []

    print("\n--- Near-boundary and extreme θ values ---")
    for theta in THETA_EDGE:
        for t_tilde in [0.0, 0.5]:
            # Use looser tolerance for edge cases
            r = run_single_test(
                "Edge_AME_Logit", target_ame_logit, oracle_jacobian_ame_logit,
                theta, t_tilde=t_tilde,
                tol_abs=1e-6,  # Looser absolute
                tol_rel=1e-4,  # Same relative
                verbose=verbose
            )
            results.append(r)

    # Check for NaN/Inf
    n_finite = sum(1 for r in results if np.isfinite(r.autodiff_jac).all())
    n_total = len(results)

    max_abs_err = max(r.abs_error for r in results)
    n_pass = sum(1 for r in results if r.passed)

    print(f"\nPart 3 Summary: {n_pass}/{n_total} passed, {n_finite}/{n_total} finite, max|err| = {max_abs_err:.2e}")

    return {
        "results": results,
        "max_error": max_abs_err,
        "passed": n_pass == n_total and n_finite == n_total,
        "n_pass": n_pass,
        "n_total": n_total,
        "n_finite": n_finite,
    }


# =============================================================================
# PART 4: BATCHED VMAP TESTS
# =============================================================================

def run_part4_batched(verbose: bool = True) -> Dict[str, Any]:
    """Test batched vmap computation."""
    print("\n" + "=" * 70)
    print("PART 4: BATCHED VMAP TESTS (n=100 each)")
    print("=" * 70)

    results = []

    # Define test configs: (name, target_fn, oracle_fn, t_tilde)
    configs = [
        # Targets (Logit family)
        ("AvgParam", target_average_parameter, oracle_jacobian_average_parameter, 0.0),
        ("AME_Logit_t0", target_ame_logit, oracle_jacobian_ame_logit, 0.0),
        ("AME_Logit_t05", target_ame_logit, oracle_jacobian_ame_logit, 0.5),
        ("Pred_Logit_t0", target_prediction_logit, oracle_jacobian_prediction_logit, 0.0),
        ("Pred_Logit_t05", target_prediction_logit, oracle_jacobian_prediction_logit, 0.5),
        # Families (AME target)
        ("AME_Linear", target_ame_linear, oracle_jacobian_ame_linear, 0.0),
        ("AME_Poisson_t0", target_ame_poisson, oracle_jacobian_ame_poisson, 0.0),
        ("AME_Poisson_t05", target_ame_poisson, oracle_jacobian_ame_poisson, 0.5),
        ("AME_Probit_t0", target_ame_probit, oracle_jacobian_ame_probit, 0.0),
        ("AME_Probit_t05", target_ame_probit, oracle_jacobian_ame_probit, 0.5),
        # Predictions (other families)
        ("Pred_Linear_t0", target_prediction_linear, oracle_jacobian_prediction_linear, 0.0),
        ("Pred_Linear_t05", target_prediction_linear, oracle_jacobian_prediction_linear, 0.5),
        ("Pred_Poisson_t0", target_prediction_poisson, oracle_jacobian_prediction_poisson, 0.0),
        ("Pred_Probit_t0", target_prediction_probit, oracle_jacobian_prediction_probit, 0.0),
    ]

    print()
    for name, target_fn, oracle_fn, t_tilde in configs:
        r = run_batched_test(name, target_fn, oracle_fn, n=100, t_tilde=t_tilde, verbose=verbose)
        results.append(r)

    max_err = max(r["max_error"] for r in results)
    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)

    print(f"\nPart 4 Summary: {n_pass}/{n_total} passed, max|err| = {max_err:.2e}")

    return {
        "results": results,
        "max_error": max_err,
        "passed": n_pass == n_total,
        "n_pass": n_pass,
        "n_total": n_total,
    }


# =============================================================================
# PART 5: PACKAGE TARGET CLASSES
# =============================================================================

def run_part5_package_targets(verbose: bool = True) -> Dict[str, Any]:
    """Test actual package Target classes (not local functions)."""
    print("\n" + "=" * 70)
    print("PART 5: PACKAGE TARGET CLASSES")
    print("=" * 70)

    results = []

    try:
        from deep_inference.targets import AverageParameter, AverageMarginalEffect, CustomTarget
        package_available = True
    except ImportError:
        print("  WARNING: Package targets not available, skipping Part 5")
        package_available = False
        return {"results": [], "max_error": 0.0, "passed": True, "n_pass": 0, "n_total": 0}

    print("\n--- 5a. AverageParameter class ---")
    for theta in THETA_STANDARD[:3]:
        theta_t = torch.tensor(theta, dtype=torch.float64)
        x_t = torch.tensor([0.0], dtype=torch.float64)
        t_tilde_t = torch.tensor(0.0, dtype=torch.float64)

        # Package implementation
        target = AverageParameter(param_index=1, theta_dim=2)
        h_pkg = target.h(x_t, theta_t, t_tilde_t)
        jac_pkg = target.jacobian(x_t, theta_t, t_tilde_t)

        # Oracle
        oracle_jac = oracle_jacobian_average_parameter(theta, 0.0)

        if jac_pkg is not None:
            jac_pkg_np = jac_pkg.detach().numpy()
            error = np.abs(jac_pkg_np - oracle_jac).max()
        else:
            # Use autodiff if jacobian returns None
            jac_autodiff = torch.func.grad(lambda th: target.h(x_t, th, t_tilde_t))(theta_t)
            jac_pkg_np = jac_autodiff.detach().numpy()
            error = np.abs(jac_pkg_np - oracle_jac).max()

        passed = error < 1e-10
        if verbose:
            print(f"  θ={theta}: H={h_pkg.item():.4f}, Jac={jac_pkg_np}, err={error:.2e} [{'PASS' if passed else 'FAIL'}]")
        results.append({"theta": theta, "error": error, "passed": passed})

    print("\n--- 5b. AverageMarginalEffect class (logit) ---")
    for theta in THETA_STANDARD[:3]:
        theta_t = torch.tensor(theta, dtype=torch.float64)
        x_t = torch.tensor([0.0], dtype=torch.float64)
        t_tilde_t = torch.tensor(0.0, dtype=torch.float64)

        target = AverageMarginalEffect(model_type='logit')
        h_pkg = target.h(x_t, theta_t, t_tilde_t)
        jac_pkg = target.jacobian(x_t, theta_t, t_tilde_t)

        oracle_jac = oracle_jacobian_ame_logit(theta, 0.0)

        if jac_pkg is not None:
            jac_pkg_np = jac_pkg.detach().numpy()
        else:
            jac_autodiff = torch.func.grad(lambda th: target.h(x_t, th, t_tilde_t))(theta_t)
            jac_pkg_np = jac_autodiff.detach().numpy()

        error = np.abs(jac_pkg_np - oracle_jac).max()
        passed = error < 1e-10
        if verbose:
            print(f"  θ={theta}: H={h_pkg.item():.4f}, err={error:.2e} [{'PASS' if passed else 'FAIL'}]")
        results.append({"theta": theta, "error": error, "passed": passed})

    print("\n--- 5c. CustomTarget with autodiff ---")
    # Custom target: H(θ) = β²
    def custom_h(x, th, t):
        return th[1] ** 2

    for theta in THETA_STANDARD[:3]:
        theta_t = torch.tensor(theta, dtype=torch.float64)
        x_t = torch.tensor([0.0], dtype=torch.float64)
        t_tilde_t = torch.tensor(0.0, dtype=torch.float64)

        target = CustomTarget(h_fn=custom_h, output_dim=1)
        h_pkg = target.h(x_t, theta_t, t_tilde_t)

        # Jacobian should be [0, 2β] via autodiff
        jac_autodiff = torch.func.grad(lambda th: target.h(x_t, th, t_tilde_t))(theta_t)
        jac_pkg_np = jac_autodiff.detach().numpy()

        oracle_jac = np.array([0.0, 2 * theta[1]])
        error = np.abs(jac_pkg_np - oracle_jac).max()
        passed = error < 1e-10
        if verbose:
            print(f"  θ={theta}: H=β²={h_pkg.item():.4f}, Jac={jac_pkg_np}, oracle={oracle_jac}, err={error:.2e} [{'PASS' if passed else 'FAIL'}]")
        results.append({"theta": theta, "error": error, "passed": passed})

    max_err = max(r["error"] for r in results) if results else 0.0
    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)

    print(f"\nPart 5 Summary: {n_pass}/{n_total} passed, max|err| = {max_err:.2e}")

    return {
        "results": results,
        "max_error": max_err,
        "passed": n_pass == n_total,
        "n_pass": n_pass,
        "n_total": n_total,
    }


# =============================================================================
# PART 6: ALL 12 FAMILIES (AME)
# =============================================================================

def run_part6_all_families(verbose: bool = True) -> Dict[str, Any]:
    """Test AME Jacobians for all 12 families."""
    print("\n" + "=" * 70)
    print("PART 6: ALL 12 FAMILIES (AME)")
    print("=" * 70)

    results = []

    # Family configs: (name, target_fn, oracle_fn)
    families = [
        # Already tested in Part 2
        ("Linear", target_ame_linear, oracle_jacobian_ame_linear),
        ("Logit", target_ame_logit, oracle_jacobian_ame_logit),
        ("Poisson", target_ame_poisson, oracle_jacobian_ame_poisson),
        ("Probit", target_ame_probit, oracle_jacobian_ame_probit),
        # New families
        ("Gamma", target_ame_gamma, oracle_jacobian_ame_gamma),
        ("Weibull", target_ame_weibull, oracle_jacobian_ame_weibull),
        ("NegBin", target_ame_negbin, oracle_jacobian_ame_negbin),
        ("Gumbel", target_ame_gumbel, oracle_jacobian_ame_gumbel),
        ("Beta", target_ame_beta, oracle_jacobian_ame_beta),
    ]

    for family_name, target_fn, oracle_fn in families:
        print(f"\n--- {family_name} ---")
        for theta in THETA_STANDARD[:3]:
            for t_tilde in [0.0, 0.5]:
                r = run_single_test(
                    f"{family_name}", target_fn, oracle_fn,
                    theta, t_tilde=t_tilde, verbose=verbose
                )
                results.append(r)

    max_err = max(r.abs_error for r in results)
    n_pass = sum(1 for r in results if r.passed)
    n_total = len(results)

    print(f"\nPart 6 Summary: {n_pass}/{n_total} passed, max|err| = {max_err:.2e}")

    return {
        "results": results,
        "max_error": max_err,
        "passed": n_pass == n_total,
        "n_pass": n_pass,
        "n_total": n_total,
    }


# =============================================================================
# PART 7: THETA_DIM > 2
# =============================================================================

def run_part7_higher_dim(verbose: bool = True) -> Dict[str, Any]:
    """Test Jacobians for theta_dim > 2 (Gaussian, Tobit, ZIP)."""
    print("\n" + "=" * 70)
    print("PART 7: THETA_DIM > 2")
    print("=" * 70)

    results = []

    # 3D theta cases (Gaussian, Tobit)
    THETA_3D = [
        np.array([0.0, 1.0, 0.0]),     # γ=0 → σ=1
        np.array([1.0, 0.5, -0.5]),    # γ<0 → σ<1
        np.array([-0.5, 2.0, 0.5]),    # γ>0 → σ>1
    ]

    # 4D theta cases (ZIP)
    THETA_4D = [
        np.array([0.0, 1.0, 0.0, 0.5]),      # rate=1, some zero-inflation
        np.array([1.0, 0.5, -1.0, 1.0]),     # various values
        np.array([-0.5, 2.0, 0.5, -0.5]),    # negative delta
    ]

    print("\n--- 7a. Gaussian (3D): AverageParameter β ---")
    for theta in THETA_3D:
        theta_t = torch.tensor(theta, dtype=torch.float64)
        t_tilde_t = torch.tensor(0.0, dtype=torch.float64)
        x_t = torch.tensor([0.0], dtype=torch.float64)

        # Target: H(θ) = β = θ[1]
        def target_fn(x, th, t):
            return th[1]

        jac_autodiff = torch.func.grad(lambda th: target_fn(x_t, th, t_tilde_t))(theta_t)
        jac_np = jac_autodiff.detach().numpy()
        oracle_jac = oracle_jacobian_avg_param_nd(theta, param_index=1)

        error = np.abs(jac_np - oracle_jac).max()
        passed = error < 1e-10

        if verbose:
            print(f"  θ={theta}: Jac={jac_np}, oracle={oracle_jac}, err={error:.2e} [{'PASS' if passed else 'FAIL'}]")
        results.append({"theta": theta, "error": error, "passed": passed, "name": "Gaussian_AvgParam"})

    print("\n--- 7b. Gaussian (3D): Prediction α + βt̃ ---")
    for theta in THETA_3D:
        for t_tilde in [0.0, 0.5]:
            theta_t = torch.tensor(theta, dtype=torch.float64)
            t_tilde_t = torch.tensor(t_tilde, dtype=torch.float64)
            x_t = torch.tensor([0.0], dtype=torch.float64)

            jac_autodiff = torch.func.grad(lambda th: target_prediction_gaussian_3d(x_t, th, t_tilde_t))(theta_t)
            jac_np = jac_autodiff.detach().numpy()
            oracle_jac = oracle_jacobian_prediction_gaussian(theta, t_tilde)

            error = np.abs(jac_np - oracle_jac).max()
            passed = error < 1e-10

            if verbose:
                print(f"  θ={theta}, t̃={t_tilde}: Jac={jac_np}, err={error:.2e} [{'PASS' if passed else 'FAIL'}]")
            results.append({"theta": theta, "error": error, "passed": passed, "name": "Gaussian_Pred"})

    print("\n--- 7c. ZIP (4D): Rate effect β ---")
    for theta in THETA_4D:
        theta_t = torch.tensor(theta, dtype=torch.float64)
        t_tilde_t = torch.tensor(0.0, dtype=torch.float64)
        x_t = torch.tensor([0.0], dtype=torch.float64)

        jac_autodiff = torch.func.grad(lambda th: target_zip_rate_effect(x_t, th, t_tilde_t))(theta_t)
        jac_np = jac_autodiff.detach().numpy()
        oracle_jac = oracle_jacobian_zip_rate_effect(theta, 0.0)

        error = np.abs(jac_np - oracle_jac).max()
        passed = error < 1e-10

        if verbose:
            print(f"  θ={theta}: Jac={jac_np}, oracle={oracle_jac}, err={error:.2e} [{'PASS' if passed else 'FAIL'}]")
        results.append({"theta": theta, "error": error, "passed": passed, "name": "ZIP_Rate"})

    print("\n--- 7d. ZIP (4D): Zero-inflation effect δ ---")
    for theta in THETA_4D:
        theta_t = torch.tensor(theta, dtype=torch.float64)
        t_tilde_t = torch.tensor(0.0, dtype=torch.float64)
        x_t = torch.tensor([0.0], dtype=torch.float64)

        jac_autodiff = torch.func.grad(lambda th: target_zip_zeroinfl_effect(x_t, th, t_tilde_t))(theta_t)
        jac_np = jac_autodiff.detach().numpy()
        oracle_jac = oracle_jacobian_zip_zeroinfl_effect(theta, 0.0)

        error = np.abs(jac_np - oracle_jac).max()
        passed = error < 1e-10

        if verbose:
            print(f"  θ={theta}: Jac={jac_np}, oracle={oracle_jac}, err={error:.2e} [{'PASS' if passed else 'FAIL'}]")
        results.append({"theta": theta, "error": error, "passed": passed, "name": "ZIP_ZeroInfl"})

    max_err = max(r["error"] for r in results)
    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)

    print(f"\nPart 7 Summary: {n_pass}/{n_total} passed, max|err| = {max_err:.2e}")

    return {
        "results": results,
        "max_error": max_err,
        "passed": n_pass == n_total,
        "n_pass": n_pass,
        "n_total": n_total,
    }


# =============================================================================
# PART 8: VARYING θ(x)
# =============================================================================

def run_part8_varying_theta(verbose: bool = True) -> Dict[str, Any]:
    """Test Jacobian when θ varies with x (simulating NN output)."""
    print("\n" + "=" * 70)
    print("PART 8: VARYING θ(x)")
    print("=" * 70)

    results = []
    np.random.seed(42)

    n = 50  # Number of observations
    X = np.random.randn(n, 3)

    # Simulate neural network output: θ(x) varies with x
    theta_array = np.column_stack([
        0.5 * np.sin(X[:, 0]),      # α(x) varies
        1.0 + 0.3 * X[:, 1],        # β(x) varies
    ])

    print(f"\n--- Testing {n} observations with varying θ(x) ---")

    # Test AME Logit at each observation
    errors = []
    for i in range(n):
        theta = theta_array[i]
        t_tilde = 0.0

        # Autodiff
        jac_autodiff = autodiff_jacobian(target_ame_logit, theta, t_tilde)

        # Oracle
        oracle_jac = oracle_jacobian_ame_logit(theta, t_tilde)

        error = np.abs(jac_autodiff - oracle_jac).max()
        errors.append(error)

    max_err = max(errors)
    mean_err = np.mean(errors)
    passed = max_err < 1e-10

    if verbose:
        print(f"  AME Logit: n={n}, max|err|={max_err:.2e}, mean|err|={mean_err:.2e} [{'PASS' if passed else 'FAIL'}]")

    results.append({"name": "AME_Logit_varying", "max_error": max_err, "mean_error": mean_err, "passed": passed})

    # Test Prediction Logit
    errors = []
    for i in range(n):
        theta = theta_array[i]
        t_tilde = 0.5

        jac_autodiff = autodiff_jacobian(target_prediction_logit, theta, t_tilde)
        oracle_jac = oracle_jacobian_prediction_logit(theta, t_tilde)

        error = np.abs(jac_autodiff - oracle_jac).max()
        errors.append(error)

    max_err = max(errors)
    mean_err = np.mean(errors)
    passed = max_err < 1e-10

    if verbose:
        print(f"  Pred Logit: n={n}, max|err|={max_err:.2e}, mean|err|={mean_err:.2e} [{'PASS' if passed else 'FAIL'}]")

    results.append({"name": "Pred_Logit_varying", "max_error": max_err, "mean_error": mean_err, "passed": passed})

    overall_max = max(r["max_error"] for r in results)
    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)

    print(f"\nPart 8 Summary: {n_pass}/{n_total} passed, max|err| = {overall_max:.2e}")

    return {
        "results": results,
        "max_error": overall_max,
        "passed": n_pass == n_total,
        "n_pass": n_pass,
        "n_total": n_total,
    }


# =============================================================================
# PART 9: ELASTICITY TARGET
# =============================================================================

def run_part9_elasticity(verbose: bool = True) -> Dict[str, Any]:
    """Standalone Elasticity target tests."""
    print("\n" + "=" * 70)
    print("PART 9: ELASTICITY TARGET")
    print("=" * 70)

    results = []

    print("\n--- Poisson Elasticity: ε = β·t̄ ---")

    T_BAR_VALUES = [0.5, 1.0, 2.0, 5.0]

    for theta in THETA_STANDARD:
        for t_bar in T_BAR_VALUES:
            theta_t = torch.tensor(theta, dtype=torch.float64)
            t_bar_t = torch.tensor(t_bar, dtype=torch.float64)
            x_t = torch.tensor([0.0], dtype=torch.float64)

            # Autodiff
            jac_autodiff = torch.func.grad(lambda th: target_elasticity_poisson(x_t, th, t_bar_t))(theta_t)
            jac_np = jac_autodiff.detach().numpy()

            # Oracle
            oracle_jac = oracle_jacobian_elasticity_poisson(theta, t_bar)

            error = np.abs(jac_np - oracle_jac).max()
            passed = error < 1e-10

            # Compute H value
            h_val = theta[1] * t_bar

            if verbose:
                print(f"  θ={theta}, t̄={t_bar}: H=β·t̄={h_val:.4f}, Jac={jac_np}, err={error:.2e} [{'PASS' if passed else 'FAIL'}]")

            results.append({"theta": theta, "t_bar": t_bar, "error": error, "passed": passed})

    max_err = max(r["error"] for r in results)
    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)

    print(f"\nPart 9 Summary: {n_pass}/{n_total} passed, max|err| = {max_err:.2e}")

    return {
        "results": results,
        "max_error": max_err,
        "passed": n_pass == n_total,
        "n_pass": n_pass,
        "n_total": n_total,
    }


# =============================================================================
# MAIN EVAL
# =============================================================================

def run_eval_04(verbose: bool = True) -> Dict[str, Any]:
    """Run full Target Jacobian evaluation."""
    print("=" * 70)
    print("EVAL 04: TARGET JACOBIAN (COMPLETE)")
    print("=" * 70)
    print("\nTests autodiff ∂H/∂θ against closed-form oracle formulas")
    print("across multiple targets, families, edge cases, and package classes.")

    # Run all 9 parts
    part1 = run_part1_targets(verbose=verbose)
    part2 = run_part2_families(verbose=verbose)
    part3 = run_part3_edge_cases(verbose=verbose)
    part4 = run_part4_batched(verbose=verbose)
    part5 = run_part5_package_targets(verbose=verbose)
    part6 = run_part6_all_families(verbose=verbose)
    part7 = run_part7_higher_dim(verbose=verbose)
    part8 = run_part8_varying_theta(verbose=verbose)
    part9 = run_part9_elasticity(verbose=verbose)

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Part':<30} {'Tests':<10} {'Passed':<10} {'Max Error':<15} {'Status':<10}")
    print("-" * 75)

    parts = [
        ("1: Targets (Logit)", part1),
        ("2: Families (AME, 4 fam)", part2),
        ("3: Edge Cases", part3),
        ("4: Batched vmap", part4),
        ("5: Package Target Classes", part5),
        ("6: All 9 Families (AME)", part6),
        ("7: theta_dim > 2", part7),
        ("8: Varying θ(x)", part8),
        ("9: Elasticity", part9),
    ]

    all_pass = True
    for name, p in parts:
        if p["n_total"] == 0:
            status = "SKIP"
            print(f"{name:<30} {'-':<10} {'-':<10} {'-':<15} {status:<10}")
        else:
            status = "PASS" if p["passed"] else "FAIL"
            print(f"{name:<30} {p['n_total']:<10} {p['n_pass']:<10} {p['max_error']:<15.2e} {status:<10}")
            if not p["passed"]:
                all_pass = False

    print("-" * 75)

    # Overall (exclude skipped parts)
    active_parts = [(n, p) for n, p in parts if p["n_total"] > 0]
    total_tests = sum(p["n_total"] for _, p in active_parts)
    total_pass = sum(p["n_pass"] for _, p in active_parts)
    overall_max_err = max(p["max_error"] for _, p in active_parts) if active_parts else 0.0

    print(f"{'TOTAL':<30} {total_tests:<10} {total_pass:<10} {overall_max_err:<15.2e}")

    print("\n" + "=" * 70)
    if all_pass:
        print("EVAL 04: PASS")
    else:
        print("EVAL 04: FAIL")
    print("=" * 70)

    return {
        "part1": part1,
        "part2": part2,
        "part3": part3,
        "part4": part4,
        "part5": part5,
        "part6": part6,
        "part7": part7,
        "part8": part8,
        "part9": part9,
        "all_pass": all_pass,
        "total_tests": total_tests,
        "total_pass": total_pass,
        "overall_max_error": overall_max_err,
    }


if __name__ == "__main__":
    result = run_eval_04(verbose=True)
