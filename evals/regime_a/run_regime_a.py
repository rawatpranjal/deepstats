"""
Run All Regime A Evaluations

Regime A: RCT Logit with ComputeLambda
"""

import sys
sys.path.insert(0, "/Users/pranjal/deepest/src")


def run_regime_a(quick: bool = False, verbose: bool = True, M: int = None):
    """
    Run all Regime A evaluations.

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
    print("# REGIME A: RANDOMIZED CONTROLLED TRIAL (RCT)")
    print("# Lambda Strategy: ComputeLambda (Monte Carlo)")
    print("#" * 80)

    # Eval 01: Lambda Computation
    print("\n\n")
    from evals.regime_a.eval_01_lambda_compute import run_eval_01_lambda_compute
    results["eval_01"] = run_eval_01_lambda_compute(verbose=verbose)

    # Eval 02: Two-Way Split
    print("\n\n")
    from evals.regime_a.eval_02_two_way_split import run_eval_02_two_way_split
    results["eval_02"] = run_eval_02_two_way_split(verbose=verbose)

    # Eval 03: Boundary Stability
    print("\n\n")
    from evals.regime_a.eval_03_boundary_stability import run_eval_03_boundary
    results["eval_03"] = run_eval_03_boundary(verbose=verbose)

    # Eval 04: Coverage
    print("\n\n")
    from evals.regime_a.eval_04_coverage import run_eval_04_coverage
    n = 300 if quick else 500
    results["eval_04"] = run_eval_04_coverage(M=M, n=n, verbose=verbose)

    # Summary
    print("\n" + "=" * 60)
    print("REGIME A SUMMARY")
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

    run_regime_a(quick=args.quick)
