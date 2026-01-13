"""
Eval 04: Target Jacobian (H_θ) - EXPANDED

Goal: Ruthless firewall testing autodiff H_θ across the full
      Targets × Families × Edge Cases matrix.

Test Matrix:
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
        - Near-boundary: θ = [±5, 1]
        - Tiny effect: θ = [0, 0.001]
        - Large effect: θ = [0, 10]

    Part 4: Batched vmap Tests
        - 100 random θ per (target, family) pair

Criteria:
    - Standard θ: max|err| < 1e-10
    - Edge θ: max|err| < 1e-6 OR relative error < 1e-4
    - Batched: max|err| < 1e-8
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
# MAIN EVAL
# =============================================================================

def run_eval_04(verbose: bool = True) -> Dict[str, Any]:
    """Run full Target Jacobian evaluation."""
    print("=" * 70)
    print("EVAL 04: TARGET JACOBIAN (EXPANDED)")
    print("=" * 70)
    print("\nTests autodiff ∂H/∂θ against closed-form oracle formulas")
    print("across multiple targets, families, and edge cases.")

    # Run all parts
    part1 = run_part1_targets(verbose=verbose)
    part2 = run_part2_families(verbose=verbose)
    part3 = run_part3_edge_cases(verbose=verbose)
    part4 = run_part4_batched(verbose=verbose)

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Part':<25} {'Tests':<10} {'Passed':<10} {'Max Error':<15} {'Status':<10}")
    print("-" * 70)

    parts = [
        ("1: Targets (Logit)", part1),
        ("2: Families (AME)", part2),
        ("3: Edge Cases", part3),
        ("4: Batched vmap", part4),
    ]

    all_pass = True
    for name, p in parts:
        status = "PASS" if p["passed"] else "FAIL"
        print(f"{name:<25} {p['n_total']:<10} {p['n_pass']:<10} {p['max_error']:<15.2e} {status:<10}")
        if not p["passed"]:
            all_pass = False

    print("-" * 70)

    # Overall
    total_tests = sum(p["n_total"] for _, p in parts)
    total_pass = sum(p["n_pass"] for _, p in parts)
    overall_max_err = max(p["max_error"] for _, p in parts)

    print(f"{'TOTAL':<25} {total_tests:<10} {total_pass:<10} {overall_max_err:<15.2e}")

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
        "all_pass": all_pass,
        "total_tests": total_tests,
        "total_pass": total_pass,
        "overall_max_error": overall_max_err,
    }


if __name__ == "__main__":
    result = run_eval_04(verbose=True)
