"""
Run All Regime B Evaluations

Regime B: Linear Model with AnalyticLambda
"""

import sys
sys.path.insert(0, "/Users/pranjal/deepest/src")


def run_regime_b(quick: bool = False, verbose: bool = True, M: int = None):
    """
    Run all Regime B evaluations.

    Args:
        quick: Use smaller samples for faster testing
        verbose: Print detailed output
        M: Override number of MC simulations for coverage test
    """
    # Set default M based on quick mode
    if M is None:
        M = 10 if quick else 100  # INCREASED from 20

    results = {}

    print("\n" + "#" * 80)
    print("# REGIME B: LINEAR MODEL (CONFOUNDED)")
    print("# Lambda Strategy: AnalyticLambda (E[TT'|X])")
    print("#" * 80)

    # Eval 01: θ Recovery
    print("\n\n")
    from evals.regime_b.eval_01_theta_recovery import run_eval_01_theta_recovery
    n = 500 if quick else 1000
    results["eval_01"] = run_eval_01_theta_recovery(n=n, verbose=verbose)

    # Eval 02: Derivatives
    print("\n\n")
    from evals.regime_b.eval_02_derivatives import run_eval_02_derivatives
    results["eval_02"] = run_eval_02_derivatives(verbose=verbose)

    # Eval 03: Analytic Lambda
    print("\n\n")
    from evals.regime_b.eval_03_lambda_analytic import run_eval_03_lambda_analytic
    results["eval_03"] = run_eval_03_lambda_analytic(verbose=verbose)

    # Eval 04: ψ Closed Form
    print("\n\n")
    from evals.regime_b.eval_04_psi_closed_form import run_eval_04_psi_closed_form
    n = 500 if quick else 1000
    results["eval_04"] = run_eval_04_psi_closed_form(n=n, verbose=verbose)

    # Eval 05: Coverage
    print("\n\n")
    from evals.regime_b.eval_05_coverage import run_eval_05_coverage
    n = 300 if quick else 500
    results["eval_05"] = run_eval_05_coverage(M=M, n=n, verbose=verbose)

    # Summary
    print("\n" + "=" * 60)
    print("REGIME B SUMMARY")
    print("=" * 60)

    for name, result in results.items():
        if result.get("skipped"):
            status = "SKIPPED"
        elif result.get("passed"):
            status = "PASS"
        elif result.get("passed") is None:
            status = "SKIPPED"
        else:
            status = "FAIL"
        print(f"  {name}: {status}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    run_regime_b(quick=args.quick)
