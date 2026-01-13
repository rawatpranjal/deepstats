"""
Eval 03: Analytic Lambda (Regime B)

Goal: Verify AnalyticLambda matches E[TT'|X].

For the Linear DGP:
    T = 0.5X + ξ,  ξ ~ N(0, 1)

So:
    E[T|X] = 0.5X
    Var(T|X) = 1
    E[T²|X] = Var + Mean² = 1 + 0.25X²

Lambda(x) = E[[1,T][1,T]' | X] = [[1, E[T|X]], [E[T|X], E[T²|X]]]
          = [[1, 0.5x], [0.5x, 1 + 0.25x²]]

KEY: Lambda does NOT depend on θ! This is Regime B.

Criteria:
    - Element-wise error < 0.1 (allows for regression error)
    - All eigenvalues > 0
"""

import sys
import numpy as np
import torch

sys.path.insert(0, "/Users/pranjal/deepest/src")

from evals.dgps.regime_b_linear import LinearDGP, generate_linear_data, oracle_lambda_linear


def run_eval_03_lambda_analytic(verbose: bool = True):
    """
    Run Analytic Lambda evaluation for Regime B.
    """
    print("=" * 60)
    print("EVAL 03: ANALYTIC LAMBDA (Regime B)")
    print("=" * 60)

    dgp = LinearDGP()

    print("\nLinear Model Lambda:")
    print("  Λ(x) = E[TT'|X] = [[1, E[T|X]], [E[T|X], E[T²|X]]]")
    print(f"  E[T|X] = {dgp.T_confound}·X")
    print(f"  Var(T|X) = {dgp.Var_T_given_X()}")
    print(f"  E[T²|X] = {dgp.Var_T_given_X()} + ({dgp.T_confound}·X)²")
    print("\nKEY: Λ does NOT depend on θ! (Regime B shortcut)")

    # Test at several X values
    test_x_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    results = []
    for x in test_x_values:
        Lambda_oracle = oracle_lambda_linear(x, dgp)
        eigvals = np.linalg.eigvalsh(Lambda_oracle)

        result = {
            "x": x,
            "Lambda": Lambda_oracle,
            "eigvals": eigvals,
            "is_psd": eigvals.min() > 0,
        }
        results.append(result)

        if verbose:
            print(f"\n--- x = {x:.2f} ---")
            print(f"  E[T|X] = {dgp.T_confound * x:.4f}")
            print(f"  E[T²|X] = {dgp.Var_T_given_X() + (dgp.T_confound * x)**2:.4f}")
            print(f"  Λ_oracle =")
            print(f"    [[{Lambda_oracle[0,0]:.4f}, {Lambda_oracle[0,1]:.4f}],")
            print(f"     [{Lambda_oracle[1,0]:.4f}, {Lambda_oracle[1,1]:.4f}]]")
            print(f"  Eigenvalues: [{eigvals[0]:.4f}, {eigvals[1]:.4f}]")

    # Try package AnalyticLambda
    try:
        from deep_inference.lambda_.analytic import AnalyticLambda
        from deep_inference.models.linear import Linear

        print("\n" + "-" * 60)
        print("TESTING PACKAGE AnalyticLambda")
        print("-" * 60)

        # Generate data
        Y, T, X, theta_true, mu_true = generate_linear_data(n=1000, seed=42, dgp=dgp)

        # Fit AnalyticLambda
        strategy = AnalyticLambda()
        model = Linear()
        strategy.fit(X, T, Y, None, model)  # theta_hat=None since Λ doesn't depend on θ!

        # Predict Lambda at test points
        errors = []
        for r in results:
            x_val = r["x"]
            X_test = torch.tensor([[x_val]], dtype=torch.float32)
            Lambda_package = strategy.predict(X_test, None)[0].numpy()

            error = np.abs(Lambda_package - r["Lambda"]).max()
            errors.append(error)
            r["Lambda_package"] = Lambda_package
            r["error"] = error

            if verbose:
                print(f"\n--- x = {x_val:.2f} (Package) ---")
                print(f"  Λ_package =")
                print(f"    [[{Lambda_package[0,0]:.4f}, {Lambda_package[0,1]:.4f}],")
                print(f"     [{Lambda_package[1,0]:.4f}, {Lambda_package[1,1]:.4f}]]")
                print(f"  Max error: {error:.6f}")

        max_error = max(errors)
        all_psd = all(r["is_psd"] for r in results)

        # Validation
        print("\n" + "=" * 60)
        print("VALIDATION CRITERIA")
        print("=" * 60)

        criteria = {
            "Max error < 0.1": max_error < 0.1,
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
            print("EVAL 03: PASS")
        else:
            print("EVAL 03: FAIL")
        print("=" * 60)

        return {
            "max_error": max_error,
            "results": results,
            "passed": all_pass,
            "skipped": False,
        }

    except ImportError as e:
        print(f"\n  [SKIP] AnalyticLambda not available: {e}")

        # Still validate oracle formulas
        print("\n" + "=" * 60)
        print("VALIDATION CRITERIA (Oracle only)")
        print("=" * 60)

        all_psd = all(r["is_psd"] for r in results)

        criteria = {
            "All eigenvalues > 0": all_psd,
        }

        all_pass = True
        for name, passed in criteria.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {name}: {status}")
            if not passed:
                all_pass = False

        print("\nEVAL 03: SKIPPED (AnalyticLambda implementation pending)")
        return {
            "results": results,
            "passed": None,
            "skipped": True,
        }


if __name__ == "__main__":
    result = run_eval_03_lambda_analytic(verbose=True)
