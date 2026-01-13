"""
Eval 01: Lambda Computation (Regime A)

Goal: Verify ComputeLambda matches analytical formula for RCT.

Oracle Λ*(x) = E_T[σ(1-σ)·TT'] where T∈{0,1} with p=0.5

For Bernoulli(0.5):
    Term 0 (T=0): 0.5 * σ(α)(1-σ(α)) * [[1,0],[0,0]]
    Term 1 (T=1): 0.5 * σ(α+β)(1-σ(α+β)) * [[1,1],[1,1]]

ComputeLambda should:
    1. Sample T from known distribution
    2. Compute Hessians for each T value
    3. Average them

This should match the Oracle EXACTLY (within MC error).

Criteria:
    - Max element-wise error < 0.05 (MC noise)
    - All eigenvalues > 0
"""

import sys
import numpy as np
import torch
from scipy.special import expit

sys.path.insert(0, "/Users/pranjal/deepest/src")

from evals.dgps.regime_a_rct_logit import (
    RCTLogitDGP,
    generate_rct_logit_data,
    oracle_lambda_rct,
)


def compute_lambda_via_package(theta: torch.Tensor, n_mc: int = 10000):
    """
    Compute Lambda using the package's ComputeLambda strategy.

    Note: This requires the ComputeLambda implementation to exist.
    If not implemented yet, this will fail gracefully.
    """
    try:
        from deep_inference.lambda_.compute import ComputeLambda

        # Create Bernoulli distribution
        class BernoulliDist:
            def __init__(self, p):
                self.p = p

            def sample(self, shape):
                return torch.bernoulli(torch.full(shape, self.p))

        strategy = ComputeLambda(
            treatment_dist=BernoulliDist(0.5),
            n_mc_samples=n_mc,
        )

        # Predict Lambda for a single observation
        X = torch.zeros(1, 1)  # X doesn't matter for this DGP
        theta_hat = theta.unsqueeze(0)

        # Need to fit first (may be a no-op for ComputeLambda)
        from deep_inference.models import Logit
        strategy.fit(X, torch.zeros(1), torch.zeros(1), theta_hat, Logit())

        Lambda = strategy.predict(X, theta_hat)
        return Lambda[0].numpy()

    except ImportError as e:
        print(f"  [SKIP] ComputeLambda not implemented: {e}")
        return None


def run_eval_01_lambda_compute(verbose: bool = True):
    """
    Run Lambda computation evaluation for Regime A.
    """
    print("=" * 60)
    print("EVAL 01: LAMBDA COMPUTATION (Regime A)")
    print("=" * 60)

    dgp = RCTLogitDGP()

    print("\nDGP: RCT Logit")
    print(f"  T ~ Bernoulli({dgp.p_treat})")
    print(f"  α*(x) = x, β*(x) = 1")

    print("\nOracle Formula:")
    print("  Λ = 0.5 * σ(α)(1-σ(α)) * [[1,0],[0,0]]")
    print("    + 0.5 * σ(α+β)(1-σ(α+β)) * [[1,1],[1,1]]")

    # Test at several θ values
    test_cases = [
        np.array([0.0, 1.0]),   # α=0, β=1
        np.array([0.5, 1.0]),   # α=0.5, β=1
        np.array([-1.0, 1.0]),  # α=-1, β=1
        np.array([1.0, 2.0]),   # α=1, β=2
    ]

    results = []
    for theta in test_cases:
        # Oracle Lambda
        Lambda_oracle = oracle_lambda_rct(0.0, theta, dgp.p_treat)

        # Package Lambda (if available)
        theta_t = torch.tensor(theta, dtype=torch.float32)
        Lambda_package = compute_lambda_via_package(theta_t, n_mc=10000)

        if verbose:
            print(f"\n--- θ = {theta} ---")
            print(f"Oracle Λ:")
            print(f"  [[{Lambda_oracle[0,0]:.6f}, {Lambda_oracle[0,1]:.6f}],")
            print(f"   [{Lambda_oracle[1,0]:.6f}, {Lambda_oracle[1,1]:.6f}]]")
            print(f"Eigenvalues: {np.linalg.eigvalsh(Lambda_oracle)}")

            if Lambda_package is not None:
                print(f"Package Λ:")
                print(f"  [[{Lambda_package[0,0]:.6f}, {Lambda_package[0,1]:.6f}],")
                print(f"   [{Lambda_package[1,0]:.6f}, {Lambda_package[1,1]:.6f}]]")
                error = np.abs(Lambda_package - Lambda_oracle).max()
                print(f"Max Error: {error:.6f}")

        results.append({
            "theta": theta,
            "oracle": Lambda_oracle,
            "package": Lambda_package,
        })

    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION CRITERIA")
    print("=" * 60)

    # Check if package implementation exists
    package_available = any(r["package"] is not None for r in results)

    if not package_available:
        print("  [SKIP] ComputeLambda not implemented yet")
        print("\nEVAL 01: SKIPPED (implementation pending)")
        return {"results": results, "passed": None, "skipped": True}

    # Check errors
    max_errors = []
    all_psd = True
    for r in results:
        if r["package"] is not None:
            error = np.abs(r["package"] - r["oracle"]).max()
            max_errors.append(error)
            eigvals = np.linalg.eigvalsh(r["package"])
            if eigvals.min() <= 0:
                all_psd = False

    max_error = max(max_errors) if max_errors else float("inf")

    criteria = {
        "Max error < 0.05": max_error < 0.05,
        "All eigenvalues > 0": all_psd,
    }

    all_pass = True
    for name, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("EVAL 01: PASS")
    else:
        print("EVAL 01: FAIL")
    print("=" * 60)

    return {"results": results, "passed": all_pass, "skipped": False}


if __name__ == "__main__":
    result = run_eval_01_lambda_compute(verbose=True)
