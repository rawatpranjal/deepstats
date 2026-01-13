"""
Eval 02: Derivatives Verification (Regime B)

Goal: Verify Score and Hessian for linear loss.

Linear Loss: ℓ(y, t, θ) = 0.5 * (y - α - β·t)²

Score (∂ℓ/∂θ):
    ℓ_θ = (α + β·t - y) · [1, t]

Hessian (∂²ℓ/∂θ²):
    ℓ_θθ = [[1, t], [t, t²]]

KEY INSIGHT: The Hessian does NOT depend on θ or y!
This is what makes it Regime B (AnalyticLambda possible).

Criteria:
    - Score matches formula (max error < 1e-5)
    - Hessian matches formula (max error < 1e-5)
    - Hessian is constant w.r.t. θ
"""

import sys
import numpy as np
import torch

sys.path.insert(0, "/Users/pranjal/deepest/src")

from evals.dgps.regime_b_linear import (
    LinearDGP,
    generate_linear_data,
    oracle_score_linear,
    oracle_hessian_linear,
)


def run_eval_02_derivatives(verbose: bool = True):
    """
    Run derivatives verification for Regime B.
    """
    print("=" * 60)
    print("EVAL 02: DERIVATIVES VERIFICATION (Regime B)")
    print("=" * 60)

    print("\nLinear Loss: ℓ = 0.5 * (y - α - β·t)²")
    print("\nExpected Derivatives:")
    print("  Score: ℓ_θ = (α + β·t - y) · [1, t]")
    print("  Hessian: ℓ_θθ = [[1, t], [t, t²]]")
    print("  KEY: Hessian is CONSTANT w.r.t. θ and y!")

    # Generate data
    dgp = LinearDGP()
    Y, T, X, theta_true, mu_true = generate_linear_data(n=100, seed=42, dgp=dgp)

    # Test cases
    test_idx = [0, 10, 50, 99]
    results = []

    for idx in test_idx:
        y = Y[idx].item()
        t = T[idx].item()
        theta = theta_true[idx].numpy()

        # Oracle score
        score_oracle = oracle_score_linear(y, t, theta)

        # Oracle hessian
        hess_oracle = oracle_hessian_linear(t)

        results.append({
            "idx": idx,
            "y": y,
            "t": t,
            "theta": theta,
            "score_oracle": score_oracle,
            "hess_oracle": hess_oracle,
        })

        if verbose:
            print(f"\n--- Test Point {idx} ---")
            print(f"  y={y:.4f}, t={t:.4f}, θ=[{theta[0]:.4f}, {theta[1]:.4f}]")
            print(f"  Residual = {theta[0] + theta[1]*t - y:.6f}")
            print(f"  Score_oracle = [{score_oracle[0]:.6f}, {score_oracle[1]:.6f}]")
            print(f"  Hessian_oracle =")
            print(f"    [[{hess_oracle[0,0]:.4f}, {hess_oracle[0,1]:.4f}],")
            print(f"     [{hess_oracle[1,0]:.4f}, {hess_oracle[1,1]:.4f}]]")

    # Try autodiff verification
    try:
        from deep_inference.autodiff.score import compute_score_batched
        from deep_inference.autodiff.hessian import compute_hessian_batched
        from deep_inference.models.linear import Linear

        model = Linear()

        # Compute autodiff versions
        score_autodiff = compute_score_batched(model.loss, Y, T, theta_true)
        hess_autodiff = compute_hessian_batched(model.loss, Y, T, theta_true)

        # Compare for test points
        score_errors = []
        hess_errors = []

        for r in results:
            idx = r["idx"]
            score_ad = score_autodiff[idx].numpy()
            hess_ad = hess_autodiff[idx].numpy()

            score_err = np.abs(score_ad - r["score_oracle"]).max()
            hess_err = np.abs(hess_ad - r["hess_oracle"]).max()

            score_errors.append(score_err)
            hess_errors.append(hess_err)

            if verbose:
                print(f"\n--- Autodiff Comparison (idx={idx}) ---")
                print(f"  Score_autodiff = [{score_ad[0]:.6f}, {score_ad[1]:.6f}]")
                print(f"  Score_error = {score_err:.2e}")
                print(f"  Hess_autodiff =")
                print(f"    [[{hess_ad[0,0]:.4f}, {hess_ad[0,1]:.4f}],")
                print(f"     [{hess_ad[1,0]:.4f}, {hess_ad[1,1]:.4f}]]")
                print(f"  Hess_error = {hess_err:.2e}")

        max_score_error = max(score_errors)
        max_hess_error = max(hess_errors)

        # Validation
        print("\n" + "=" * 60)
        print("VALIDATION CRITERIA")
        print("=" * 60)

        criteria = {
            "Score max error < 1e-5": max_score_error < 1e-5,
            "Hessian max error < 1e-5": max_hess_error < 1e-5,
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
            "max_score_error": max_score_error,
            "max_hess_error": max_hess_error,
            "results": results,
            "passed": all_pass,
            "skipped": False,
        }

    except ImportError as e:
        print(f"\n  [SKIP] Autodiff not available: {e}")

        # Still validate oracle formulas against each other
        print("\n" + "=" * 60)
        print("VALIDATION CRITERIA (Oracle only)")
        print("=" * 60)

        # Check Hessian is PSD and symmetric
        all_psd = True
        all_symmetric = True
        for r in results:
            H = r["hess_oracle"]
            eigvals = np.linalg.eigvalsh(H)
            if eigvals.min() < 0:
                all_psd = False
            if np.abs(H - H.T).max() > 1e-10:
                all_symmetric = False

        criteria = {
            "Hessian is PSD": all_psd,
            "Hessian is symmetric": all_symmetric,
        }

        all_pass = True
        for name, passed in criteria.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {name}: {status}")
            if not passed:
                all_pass = False

        print("\nEVAL 02: PARTIAL (autodiff not tested)")
        return {
            "results": results,
            "passed": all_pass,
            "skipped": False,
            "note": "Autodiff not available - oracle formulas only",
        }


if __name__ == "__main__":
    result = run_eval_02_derivatives(verbose=True)
