"""
Eval 02: Autodiff vs Calculus

Goal: Verify torch.func autodiff matches hand-derived formulas.

Oracle (Logistic Loss):
    ℓ(y, t, θ) = -[y·log(p) + (1-y)·log(1-p)]
    where p = σ(α + β·t)

    Score ℓ_θ:
        ∂ℓ/∂α = p - y
        ∂ℓ/∂β = t·(p - y)

    Hessian ℓ_θθ:
        ∂²ℓ/∂α² = p(1-p)
        ∂²ℓ/∂α∂β = t·p(1-p)
        ∂²ℓ/∂β² = t²·p(1-p)

Criteria:
    - Score: max|autodiff - oracle| < 1e-6
    - Hessian: max|autodiff - oracle| < 1e-6
"""

import sys
import numpy as np
import torch
from torch import Tensor
from scipy.special import expit

sys.path.insert(0, "/Users/pranjal/deepest/src")

from evals.dgps.regime_c_obs_logit import oracle_score, oracle_hessian


def logistic_loss(y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
    """
    Logistic loss for a single observation.

    Args:
        y: scalar outcome
        t: scalar treatment
        theta: (2,) [alpha, beta]

    Returns:
        scalar loss
    """
    alpha = theta[0]
    beta = theta[1]
    logit = alpha + beta * t
    p = torch.sigmoid(logit)
    eps = 1e-7
    p = torch.clamp(p, eps, 1 - eps)
    return -y * torch.log(p) - (1 - y) * torch.log(1 - p)


def autodiff_score(y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
    """Compute score via autodiff."""
    return torch.func.grad(lambda th: logistic_loss(y, t, th))(theta)


def autodiff_hessian(y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
    """Compute Hessian via autodiff."""
    return torch.func.hessian(lambda th: logistic_loss(y, t, th))(theta)


def run_single_test(y: float, t: float, theta: np.ndarray, verbose: bool = True):
    """
    Run single test case comparing autodiff vs oracle.

    Returns dict with errors and pass/fail status.
    """
    # Convert to tensors
    y_t = torch.tensor(y, dtype=torch.float64)
    t_t = torch.tensor(t, dtype=torch.float64)
    theta_t = torch.tensor(theta, dtype=torch.float64)

    # Oracle values
    oracle_s = oracle_score(y, t, theta)
    oracle_h = oracle_hessian(y, t, theta)

    # Autodiff values
    autodiff_s = autodiff_score(y_t, t_t, theta_t).numpy()
    autodiff_h = autodiff_hessian(y_t, t_t, theta_t).numpy()

    # Compute errors
    score_error = np.abs(autodiff_s - oracle_s).max()
    hessian_error = np.abs(autodiff_h - oracle_h).max()

    if verbose:
        print(f"\n  Test: y={y}, t={t}, θ={theta}")
        print(f"  p = σ(α + β·t) = {expit(theta[0] + theta[1] * t):.6f}")
        print(f"\n  Score (∂ℓ/∂θ):")
        print(f"    Oracle:   [{oracle_s[0]:.6f}, {oracle_s[1]:.6f}]")
        print(f"    Autodiff: [{autodiff_s[0]:.6f}, {autodiff_s[1]:.6f}]")
        print(f"    Max Error: {score_error:.2e}")

        print(f"\n  Hessian (∂²ℓ/∂θ²):")
        print(f"    Oracle:")
        print(f"      [{oracle_h[0,0]:.6f}, {oracle_h[0,1]:.6f}]")
        print(f"      [{oracle_h[1,0]:.6f}, {oracle_h[1,1]:.6f}]")
        print(f"    Autodiff:")
        print(f"      [{autodiff_h[0,0]:.6f}, {autodiff_h[0,1]:.6f}]")
        print(f"      [{autodiff_h[1,0]:.6f}, {autodiff_h[1,1]:.6f}]")
        print(f"    Max Error: {hessian_error:.2e}")

    return {
        "y": y,
        "t": t,
        "theta": theta,
        "oracle_score": oracle_s,
        "autodiff_score": autodiff_s,
        "oracle_hessian": oracle_h,
        "autodiff_hessian": autodiff_h,
        "score_error": score_error,
        "hessian_error": hessian_error,
    }


def run_batched_test(n: int = 100, seed: int = 42, verbose: bool = True):
    """
    Run batched test using vmap.

    Tests that package's vmap-based autodiff matches oracle on n random samples.
    """
    from deep_inference.autodiff.score import compute_score_vmap
    from deep_inference.autodiff.hessian import compute_hessian_vmap

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate random test data
    Y = np.random.binomial(1, 0.5, n).astype(float)
    T = np.random.normal(0, 1, n)
    theta = np.random.randn(n, 2)

    # Convert to tensors
    Y_t = torch.tensor(Y, dtype=torch.float64)
    T_t = torch.tensor(T, dtype=torch.float64)
    theta_t = torch.tensor(theta, dtype=torch.float64)

    # Define loss for vmap
    def loss_fn(y, t, th):
        alpha = th[0]
        beta = th[1]
        logit = alpha + beta * t
        p = torch.sigmoid(logit)
        eps = 1e-7
        p = torch.clamp(p, eps, 1 - eps)
        return -y * torch.log(p) - (1 - y) * torch.log(1 - p)

    # Compute via package autodiff
    scores_autodiff = compute_score_vmap(loss_fn, Y_t, T_t, theta_t).numpy()
    hessians_autodiff = compute_hessian_vmap(loss_fn, Y_t, T_t, theta_t).numpy()

    # Compute via oracle (loop)
    scores_oracle = np.zeros((n, 2))
    hessians_oracle = np.zeros((n, 2, 2))

    for i in range(n):
        scores_oracle[i] = oracle_score(Y[i], T[i], theta[i])
        hessians_oracle[i] = oracle_hessian(Y[i], T[i], theta[i])

    # Compute errors
    score_errors = np.abs(scores_autodiff - scores_oracle)
    hessian_errors = np.abs(hessians_autodiff - hessians_oracle)

    max_score_error = score_errors.max()
    max_hessian_error = hessian_errors.max()

    if verbose:
        print(f"\n  Batched Test (n={n}):")
        print(f"    Score max error: {max_score_error:.2e}")
        print(f"    Score mean error: {score_errors.mean():.2e}")
        print(f"    Hessian max error: {max_hessian_error:.2e}")
        print(f"    Hessian mean error: {hessian_errors.mean():.2e}")

    return {
        "n": n,
        "max_score_error": max_score_error,
        "mean_score_error": score_errors.mean(),
        "max_hessian_error": max_hessian_error,
        "mean_hessian_error": hessian_errors.mean(),
    }


def run_eval_02(verbose: bool = True):
    """
    Run autodiff vs calculus evaluation.

    Tests:
    1. Single test cases with various (y, t, θ) values
    2. Batched vmap test with 100 random samples
    """
    print("=" * 60)
    print("EVAL 02: AUTODIFF VS CALCULUS")
    print("=" * 60)

    print("\nLogistic Loss:")
    print("  ℓ(y, t, θ) = -[y·log(p) + (1-y)·log(1-p)]")
    print("  where p = σ(α + β·t)")

    # Test cases
    test_cases = [
        (1.0, 0.5, np.array([0.1, 2.0])),   # y=1, moderate t
        (0.0, -1.0, np.array([0.0, 1.0])),  # y=0, negative t
        (1.0, 0.0, np.array([1.0, 0.5])),   # y=1, t=0
        (0.0, 2.0, np.array([-0.5, -1.0])), # y=0, large positive t
        (1.0, 0.01, np.array([0.0, 0.0])),  # Near-zero theta
    ]

    print("\n" + "-" * 60)
    print("SINGLE POINT TESTS")
    print("-" * 60)

    results = []
    for y, t, theta in test_cases:
        result = run_single_test(y, t, theta, verbose=verbose)
        results.append(result)

    # Batched test
    print("\n" + "-" * 60)
    print("BATCHED VMAP TEST")
    print("-" * 60)

    batched_result = run_batched_test(n=100, seed=42, verbose=verbose)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    max_score_error = max(r["score_error"] for r in results)
    max_hessian_error = max(r["hessian_error"] for r in results)

    print(f"\nSingle-point tests ({len(results)} cases):")
    print(f"  Max score error: {max_score_error:.2e}")
    print(f"  Max Hessian error: {max_hessian_error:.2e}")

    print(f"\nBatched vmap test (n=100):")
    print(f"  Max score error: {batched_result['max_score_error']:.2e}")
    print(f"  Max Hessian error: {batched_result['max_hessian_error']:.2e}")

    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION CRITERIA")
    print("=" * 60)

    criteria = {
        "Single Score: max|err| < 1e-6": max_score_error < 1e-6,
        "Single Hessian: max|err| < 1e-6": max_hessian_error < 1e-6,
        "Batched Score: max|err| < 1e-5": batched_result["max_score_error"] < 1e-5,
        "Batched Hessian: max|err| < 1e-5": batched_result["max_hessian_error"] < 1e-5,
    }

    all_pass = True
    for name, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("EVAL 02: PASS")
    else:
        print("EVAL 02: FAIL")
    print("=" * 60)

    return {
        "single_results": results,
        "batched_result": batched_result,
        "passed": all_pass,
    }


if __name__ == "__main__":
    result = run_eval_02(verbose=True)
