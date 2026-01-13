"""
Eval 04: Target Jacobian (H_θ)

Goal: Verify autodiff H_θ matches chain rule derivation.

Target: AME at t̃=0
    H(x, θ) = σ(α + β·t̃)·(1 - σ(α + β·t̃))·β

At t̃=0:
    H(θ) = σ(α)·(1-σ(α))·β

Jacobian ∂H/∂θ:
    Let s = σ(α), then H = s(1-s)β

    ∂H/∂α = β·s(1-s)(1-2s)
    ∂H/∂β = s(1-s)

Criteria:
    - max|autodiff - oracle| < 1e-5
"""

import sys
import numpy as np
import torch
from torch import Tensor
from scipy.special import expit

sys.path.insert(0, "/Users/pranjal/deepest/src")

from evals.dgp import oracle_target_jacobian


def ame_target(x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
    """
    AME target functional.

    H(θ) = σ(α + β·t̃)·(1 - σ(α + β·t̃))·β

    Args:
        x: (d_x,) covariates (not used for this target)
        theta: (2,) [alpha, beta]
        t_tilde: scalar evaluation point

    Returns:
        scalar AME value
    """
    alpha = theta[0]
    beta = theta[1]
    logit = alpha + beta * t_tilde
    s = torch.sigmoid(logit)
    return s * (1 - s) * beta


def autodiff_target_jacobian(x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
    """Compute target Jacobian via autodiff."""
    return torch.func.grad(lambda th: ame_target(x, th, t_tilde))(theta)


def run_single_test(theta: np.ndarray, t_tilde: float = 0.0, verbose: bool = True):
    """
    Run single test case comparing autodiff vs oracle.

    Returns dict with error and pass/fail status.
    """
    # Convert to tensors
    x_t = torch.tensor([0.0], dtype=torch.float64)  # x not used
    theta_t = torch.tensor(theta, dtype=torch.float64)
    t_tilde_t = torch.tensor(t_tilde, dtype=torch.float64)

    # Oracle
    oracle_jac = oracle_target_jacobian(theta, t_tilde)

    # Autodiff
    autodiff_jac = autodiff_target_jacobian(x_t, theta_t, t_tilde_t).numpy()

    # Error
    error = np.abs(autodiff_jac - oracle_jac).max()

    # Compute intermediate values for display
    alpha, beta = theta
    s = expit(alpha + beta * t_tilde)
    h = s * (1 - s) * beta

    if verbose:
        print(f"\n  Test: θ={theta}, t̃={t_tilde}")
        print(f"  σ(α + β·t̃) = {s:.6f}")
        print(f"  H(θ) = σ(1-σ)β = {h:.6f}")
        print(f"\n  Target Jacobian ∂H/∂θ:")
        print(f"    Oracle:   [{oracle_jac[0]:.6f}, {oracle_jac[1]:.6f}]")
        print(f"    Autodiff: [{autodiff_jac[0]:.6f}, {autodiff_jac[1]:.6f}]")
        print(f"    Max Error: {error:.2e}")

    return {
        "theta": theta,
        "t_tilde": t_tilde,
        "oracle_jacobian": oracle_jac,
        "autodiff_jacobian": autodiff_jac,
        "error": error,
        "h_value": h,
    }


def run_batched_test(n: int = 100, seed: int = 42, t_tilde: float = 0.0, verbose: bool = True):
    """
    Run batched test using vmap.

    Tests package's vmap-based Jacobian computation.
    """
    from deep_inference.autodiff.jacobian import compute_target_jacobian_vmap

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate random theta values
    theta = np.random.randn(n, 2)

    # Convert to tensors
    X_t = torch.zeros(n, 1, dtype=torch.float64)  # x not used
    theta_t = torch.tensor(theta, dtype=torch.float64)
    t_tilde_t = torch.tensor(t_tilde, dtype=torch.float64)

    # Define target for vmap
    def target_fn(x, th, t_til):
        alpha = th[0]
        beta = th[1]
        logit = alpha + beta * t_til
        s = torch.sigmoid(logit)
        return s * (1 - s) * beta

    # Compute via package autodiff
    jac_autodiff = compute_target_jacobian_vmap(target_fn, X_t, theta_t, t_tilde_t).numpy()

    # Compute via oracle (loop)
    jac_oracle = np.zeros((n, 2))
    for i in range(n):
        jac_oracle[i] = oracle_target_jacobian(theta[i], t_tilde)

    # Errors
    errors = np.abs(jac_autodiff - jac_oracle)
    max_error = errors.max()
    mean_error = errors.mean()

    if verbose:
        print(f"\n  Batched Test (n={n}, t̃={t_tilde}):")
        print(f"    Max error: {max_error:.2e}")
        print(f"    Mean error: {mean_error:.2e}")

    return {
        "n": n,
        "max_error": max_error,
        "mean_error": mean_error,
    }


def run_eval_04(verbose: bool = True):
    """
    Run Target Jacobian evaluation.

    Tests:
    1. Single test cases with various θ values
    2. Batched vmap test with 100 random samples
    3. Tests at different t̃ values
    """
    print("=" * 60)
    print("EVAL 04: TARGET JACOBIAN")
    print("=" * 60)

    print("\nTarget: AME at t̃")
    print("  H(θ) = σ(α + β·t̃)·(1 - σ(α + β·t̃))·β")
    print("\nJacobian:")
    print("  ∂H/∂α = β·σ(1-σ)(1-2σ)")
    print("  ∂H/∂β = σ(1-σ) + β·t̃·σ(1-σ)(1-2σ)")

    # Test cases at t̃=0
    test_cases_t0 = [
        np.array([0.0, 1.0]),    # Simple case
        np.array([1.0, 0.5]),    # Positive alpha
        np.array([-0.5, 2.0]),   # Negative alpha, large beta
        np.array([0.1, -1.0]),   # Negative beta
        np.array([2.0, 0.1]),    # Large alpha
    ]

    print("\n" + "-" * 60)
    print("SINGLE POINT TESTS (t̃=0)")
    print("-" * 60)

    results_t0 = []
    for theta in test_cases_t0:
        result = run_single_test(theta, t_tilde=0.0, verbose=verbose)
        results_t0.append(result)

    # Test cases at t̃=0.5
    print("\n" + "-" * 60)
    print("SINGLE POINT TESTS (t̃=0.5)")
    print("-" * 60)

    results_t05 = []
    for theta in test_cases_t0[:3]:  # Just a few cases
        result = run_single_test(theta, t_tilde=0.5, verbose=verbose)
        results_t05.append(result)

    # Batched tests
    print("\n" + "-" * 60)
    print("BATCHED VMAP TESTS")
    print("-" * 60)

    batched_t0 = run_batched_test(n=100, seed=42, t_tilde=0.0, verbose=verbose)
    batched_t05 = run_batched_test(n=100, seed=42, t_tilde=0.5, verbose=verbose)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    max_single_error_t0 = max(r["error"] for r in results_t0)
    max_single_error_t05 = max(r["error"] for r in results_t05)

    print(f"\nSingle-point tests (t̃=0): max error = {max_single_error_t0:.2e}")
    print(f"Single-point tests (t̃=0.5): max error = {max_single_error_t05:.2e}")
    print(f"Batched test (t̃=0): max error = {batched_t0['max_error']:.2e}")
    print(f"Batched test (t̃=0.5): max error = {batched_t05['max_error']:.2e}")

    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION CRITERIA")
    print("=" * 60)

    criteria = {
        "Single t̃=0: max|err| < 1e-5": max_single_error_t0 < 1e-5,
        "Single t̃=0.5: max|err| < 1e-5": max_single_error_t05 < 1e-5,
        "Batched t̃=0: max|err| < 1e-5": batched_t0["max_error"] < 1e-5,
        "Batched t̃=0.5: max|err| < 1e-5": batched_t05["max_error"] < 1e-5,
    }

    all_pass = True
    for name, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("EVAL 04: PASS")
    else:
        print("EVAL 04: FAIL")
    print("=" * 60)

    return {
        "results_t0": results_t0,
        "results_t05": results_t05,
        "batched_t0": batched_t0,
        "batched_t05": batched_t05,
        "passed": all_pass,
    }


if __name__ == "__main__":
    result = run_eval_04(verbose=True)
