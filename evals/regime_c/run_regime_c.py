"""
Run All Regime C Evaluations

Regime C: Observational Logit with EstimateLambda (3-way split)
"""

import sys
sys.path.insert(0, "/Users/pranjal/deepest/src")


def run_regime_c(quick: bool = False, verbose: bool = True, M: int = None):
    """
    Run all Regime C evaluations.

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
    print("# REGIME C: OBSERVATIONAL LOGIT (CONFOUNDED)")
    print("# Lambda Strategy: EstimateLambda (3-way cross-fitting)")
    print("#" * 80)

    # Eval 01: θ Recovery
    print("\n\n")
    from evals.regime_c.eval_01_theta import run_eval_01
    n = 2000 if quick else 5000
    epochs = 100 if quick else 200
    results["eval_01"] = run_eval_01(n=n, epochs=epochs, verbose=verbose)

    # Eval 02: Autodiff
    print("\n\n")
    from evals.regime_c.eval_02_autodiff import run_eval_02
    results["eval_02"] = run_eval_02(verbose=verbose)

    # Eval 03: Lambda Estimation
    print("\n\n")
    from evals.regime_c.eval_03_lambda import run_eval_03
    n = 500 if quick else 1000
    results["eval_03"] = run_eval_03(n=n, verbose=verbose)

    # Eval 04: Target Jacobian
    print("\n\n")
    from evals.regime_c.eval_04_jacobian import run_eval_04
    results["eval_04"] = run_eval_04(verbose=verbose)

    # Eval 05: Psi Assembly
    print("\n\n")
    from evals.regime_c.eval_05_psi import run_eval_05
    n = 500 if quick else 1000
    results["eval_05"] = run_eval_05(n=n, verbose=verbose)

    # Eval 06: Coverage
    print("\n\n")
    from evals.regime_c.eval_06_coverage import run_eval_06
    n = 500 if quick else 1000
    results["eval_06"] = run_eval_06(M=M, n=n, verbose=verbose)

    # Summary
    print("\n" + "=" * 60)
    print("REGIME C SUMMARY")
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

    run_regime_c(quick=args.quick)
