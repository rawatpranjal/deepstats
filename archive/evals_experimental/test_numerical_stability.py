"""
Numerical Stability Tests

Tests for edge cases that could cause numerical issues:
1. Near-singular Lambda matrices
2. Extreme probability predictions (p ≈ 0 or 1)
3. Large/small treatment values

These tests verify the package handles edge cases gracefully
without producing NaN/Inf values.
"""

import sys
import numpy as np
import torch
from typing import Dict, Any

sys.path.insert(0, "/Users/pranjal/deepest/src")


def test_near_singular_lambda(verbose: bool = True) -> Dict[str, Any]:
    """
    Test STAB1: Near-Singular Λ

    Verify regularization kicks in when Λ is near-singular.
    This happens when Var(T|X) ≈ 0 for some observations.

    Expected behavior:
    - (Λ + εI)⁻¹ regularization activates
    - SE should be large (reflecting uncertainty)
    - No NaN/Inf values
    """
    from deep_inference import inference
    from deep_inference.utils.linalg import safe_inverse

    if verbose:
        print("=" * 60)
        print("TEST STAB1: NEAR-SINGULAR LAMBDA")
        print("=" * 60)

    results = {"passed": True, "metrics": {}}

    # Test 1: Near-singular 2x2 matrix
    if verbose:
        print("\n--- Test 1: Safe inverse of near-singular matrix ---")

    # Create near-singular Lambda (det ≈ 0)
    eps_values = [1e-6, 1e-8, 1e-10, 1e-12]
    for eps in eps_values:
        Lambda = torch.tensor([[1.0, 1.0 - eps], [1.0 - eps, 1.0]], dtype=torch.float32)
        det = torch.det(Lambda).item()
        cond = torch.linalg.cond(Lambda).item()

        try:
            Lambda_inv, n_reg = safe_inverse(Lambda.unsqueeze(0))
            Lambda_inv = Lambda_inv.squeeze(0)
            has_nan = torch.isnan(Lambda_inv).any().item()
            has_inf = torch.isinf(Lambda_inv).any().item()

            if verbose:
                print(f"  eps={eps:.0e}: det={det:.2e}, cond={cond:.2e}, "
                      f"regularized={n_reg>0}, nan={has_nan}, inf={has_inf}")

            if has_nan or has_inf:
                results["passed"] = False
                results["metrics"][f"singular_eps_{eps}"] = "FAIL: NaN/Inf"

        except Exception as e:
            if verbose:
                print(f"  eps={eps:.0e}: ERROR - {e}")
            results["passed"] = False
            results["metrics"][f"singular_eps_{eps}"] = f"FAIL: {e}"

    # Test 2: Batch of matrices with varying condition numbers
    if verbose:
        print("\n--- Test 2: Batch with varying condition numbers ---")

    batch_size = 100
    cond_numbers = np.logspace(0, 10, batch_size)  # 1 to 1e10

    Lambda_batch = []
    for cond in cond_numbers:
        # Create matrix with specific condition number
        L = np.array([[1.0, 0.9], [0.9, 1.0]])
        U, S, Vh = np.linalg.svd(L)
        S_new = np.array([1.0, 1.0 / cond])
        L_new = U @ np.diag(S_new) @ Vh
        Lambda_batch.append(L_new)

    Lambda_batch = torch.tensor(np.array(Lambda_batch), dtype=torch.float32)

    try:
        Lambda_inv, n_reg = safe_inverse(Lambda_batch)
        nan_count = torch.isnan(Lambda_inv).any(dim=(1, 2)).sum().item()
        inf_count = torch.isinf(Lambda_inv).any(dim=(1, 2)).sum().item()

        results["metrics"]["batch_nan_count"] = nan_count
        results["metrics"]["batch_inf_count"] = inf_count
        results["metrics"]["batch_regularized"] = n_reg

        if verbose:
            print(f"  Batch of {batch_size} matrices:")
            print(f"    Condition numbers: 1 to 1e10")
            print(f"    NaN count: {nan_count}")
            print(f"    Inf count: {inf_count}")
            print(f"    Regularized: {n_reg}")

        if nan_count > 0 or inf_count > 0:
            results["passed"] = False

    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
        results["passed"] = False
        results["metrics"]["batch_error"] = str(e)

    if verbose:
        status = "PASS" if results["passed"] else "FAIL"
        print(f"\n--- TEST STAB1: {status} ---")

    return results


def test_extreme_predictions(verbose: bool = True) -> Dict[str, Any]:
    """
    Test STAB2: Extreme Predictions

    Verify stability when σ(θ'T) → 0 or 1 for logit model.
    The Hessian p(1-p) → 0 at extremes.

    Expected behavior:
    - No NaN in Lambda, score, or psi
    - Regularization handles near-zero Hessian
    """
    from deep_inference.core.autodiff import compute_score, compute_hessian
    from deep_inference.families.logit import LogitFamily

    if verbose:
        print("\n" + "=" * 60)
        print("TEST STAB2: EXTREME PREDICTIONS")
        print("=" * 60)

    results = {"passed": True, "metrics": {}}
    family = LogitFamily()

    # Test extreme theta values that lead to p ≈ 0 or 1
    extreme_thetas = [
        (10.0, 0.1, "p ≈ 0.9999"),
        (-10.0, 0.1, "p ≈ 0.0001"),
        (0.0, 0.0, "p = 0.5"),
        (20.0, 0.0, "p ≈ 1"),
        (-20.0, 0.0, "p ≈ 0"),
        (5.0, 5.0, "high beta"),
    ]

    if verbose:
        print("\n--- Testing extreme theta values ---")

    for alpha, beta, desc in extreme_thetas:
        theta = torch.tensor([[alpha, beta]], dtype=torch.float32)
        y = torch.tensor([1.0], dtype=torch.float32)
        t = torch.tensor([0.5], dtype=torch.float32)

        try:
            # Compute score
            score = family.score(y, t, theta)
            score_nan = torch.isnan(score).any().item()
            score_inf = torch.isinf(score).any().item()

            # Compute Hessian
            hess = family.hessian(y, t, theta)
            hess_nan = torch.isnan(hess).any().item()
            hess_inf = torch.isinf(hess).any().item()

            # Compute loss
            loss = family.loss(y, t, theta)
            loss_nan = torch.isnan(loss).any().item()
            loss_inf = torch.isinf(loss).any().item()

            if verbose:
                print(f"  {desc} (α={alpha}, β={beta}):")
                print(f"    score: nan={score_nan}, inf={score_inf}")
                print(f"    hess:  nan={hess_nan}, inf={hess_inf}")
                print(f"    loss:  nan={loss_nan}, inf={loss_inf}")

            if score_nan or score_inf or hess_nan or hess_inf or loss_nan or loss_inf:
                results["passed"] = False
                results["metrics"][f"extreme_{desc}"] = "FAIL"
            else:
                results["metrics"][f"extreme_{desc}"] = "PASS"

        except Exception as e:
            if verbose:
                print(f"  {desc}: ERROR - {e}")
            results["passed"] = False
            results["metrics"][f"extreme_{desc}"] = f"ERROR: {e}"

    # Test with batch of gradually extreme values
    if verbose:
        print("\n--- Testing gradient of extremity ---")

    alphas = np.linspace(-20, 20, 41)
    n_nan_scores = 0
    n_nan_hessians = 0

    for alpha in alphas:
        theta = torch.tensor([[alpha, 0.5]], dtype=torch.float32)
        y = torch.tensor([1.0], dtype=torch.float32)
        t = torch.tensor([0.0], dtype=torch.float32)

        score = family.score(y, t, theta)
        hess = family.hessian(y, t, theta)

        if torch.isnan(score).any():
            n_nan_scores += 1
        if torch.isnan(hess).any():
            n_nan_hessians += 1

    results["metrics"]["extreme_gradient_nan_scores"] = n_nan_scores
    results["metrics"]["extreme_gradient_nan_hessians"] = n_nan_hessians

    if verbose:
        print(f"  Alpha range [-20, 20], 41 values:")
        print(f"    NaN scores: {n_nan_scores}")
        print(f"    NaN hessians: {n_nan_hessians}")

    if n_nan_scores > 0 or n_nan_hessians > 0:
        results["passed"] = False

    if verbose:
        status = "PASS" if results["passed"] else "FAIL"
        print(f"\n--- TEST STAB2: {status} ---")

    return results


def test_large_treatment_values(verbose: bool = True) -> Dict[str, Any]:
    """
    Test STAB3: Large Treatment Values

    Verify stability when T takes very large or very small values.
    """
    from deep_inference.families.logit import LogitFamily
    from deep_inference.families.linear import LinearFamily

    if verbose:
        print("\n" + "=" * 60)
        print("TEST STAB3: LARGE TREATMENT VALUES")
        print("=" * 60)

    results = {"passed": True, "metrics": {}}

    # Test with large T values
    large_t_values = [0.0, 1.0, 10.0, 100.0, 1000.0, -1000.0]

    if verbose:
        print("\n--- Logit family with large T ---")

    logit = LogitFamily()
    for t_val in large_t_values:
        theta = torch.tensor([[0.0, 0.1]], dtype=torch.float32)
        y = torch.tensor([1.0], dtype=torch.float32)
        t = torch.tensor([t_val], dtype=torch.float32)

        try:
            score = logit.score(y, t, theta)
            hess = logit.hessian(y, t, theta)

            score_ok = not (torch.isnan(score).any() or torch.isinf(score).any())
            hess_ok = not (torch.isnan(hess).any() or torch.isinf(hess).any())

            if verbose:
                print(f"  T={t_val}: score_ok={score_ok}, hess_ok={hess_ok}")

            if not (score_ok and hess_ok):
                results["passed"] = False

        except Exception as e:
            if verbose:
                print(f"  T={t_val}: ERROR - {e}")
            results["passed"] = False

    if verbose:
        print("\n--- Linear family with large T ---")

    linear = LinearFamily()
    for t_val in large_t_values:
        theta = torch.tensor([[0.0, 0.1]], dtype=torch.float32)
        y = torch.tensor([1.0], dtype=torch.float32)
        t = torch.tensor([t_val], dtype=torch.float32)

        try:
            score = linear.score(y, t, theta)
            hess = linear.hessian(y, t, theta)

            score_ok = not (torch.isnan(score).any() or torch.isinf(score).any())
            hess_ok = not (torch.isnan(hess).any() or torch.isinf(hess).any())

            if verbose:
                print(f"  T={t_val}: score_ok={score_ok}, hess_ok={hess_ok}")

            if not (score_ok and hess_ok):
                results["passed"] = False

        except Exception as e:
            if verbose:
                print(f"  T={t_val}: ERROR - {e}")
            results["passed"] = False

    if verbose:
        status = "PASS" if results["passed"] else "FAIL"
        print(f"\n--- TEST STAB3: {status} ---")

    return results


def run_stability_tests(verbose: bool = True) -> Dict[str, Any]:
    """
    Run all numerical stability tests.

    Returns:
        Dictionary with results from all stability tests
    """
    print("\n" + "#" * 60)
    print("# NUMERICAL STABILITY TESTS")
    print("#" * 60)

    results = {}

    # Test 1: Near-singular Lambda
    results["stab1_singular_lambda"] = test_near_singular_lambda(verbose=verbose)

    # Test 2: Extreme predictions
    results["stab2_extreme_predictions"] = test_extreme_predictions(verbose=verbose)

    # Test 3: Large treatment values
    results["stab3_large_treatment"] = test_large_treatment_values(verbose=verbose)

    # Summary
    all_pass = all(r.get("passed", False) for r in results.values())

    print("\n" + "=" * 60)
    print("STABILITY TESTS SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        status = "PASS" if result.get("passed") else "FAIL"
        print(f"  {name}: {status}")
    print("-" * 60)
    print(f"  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)

    return {"results": results, "passed": all_pass}


if __name__ == "__main__":
    run_stability_tests(verbose=True)
