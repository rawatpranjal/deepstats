"""
Run All Evaluations (All 3 Regimes)

Comprehensive ground truth validation for deep_inference package.
Validates all 3 Lambda regimes:
    - Regime A: RCT Logit with ComputeLambda
    - Regime B: Linear with AnalyticLambda
    - Regime C: Observational Logit with EstimateLambda

Usage:
    python -m evals.run_all > evals_report.txt 2>&1
    python -m evals.run_all --quick  # Faster with smaller samples (M=10)
    python -m evals.run_all --regime a  # Run only Regime A
    python -m evals.run_all --M 100  # Override simulation count
    python -m evals.run_all --output-dir evals/reports  # Custom report output
"""

import sys
import datetime
from typing import Dict, Any, Optional

sys.path.insert(0, "/Users/pranjal/deepest/src")


def run_all_evals(
    verbose: bool = True,
    quick: bool = False,
    regime: Optional[str] = None,
    M: Optional[int] = None,
    output_dir: str = "evals/reports",
) -> Dict[str, Any]:
    """
    Run evaluations for all 3 regimes (or a specific one).

    Args:
        verbose: Print all output
        quick: Use smaller sample sizes for faster testing
        regime: Run only specific regime ('a', 'b', or 'c')
        M: Override simulation count for coverage tests
           Default: 10 (quick), 100 (full)
        output_dir: Directory for report output files
    """
    # Set default M based on quick mode
    if M is None:
        M = 10 if quick else 100  # INCREASED from 20

    results = {}

    # Print header
    print("=" * 80)
    print("COMPREHENSIVE EVALUATION REPORT - ALL 3 REGIMES")
    print("=" * 80)
    print(f"\nGenerated: {datetime.datetime.now().isoformat()}")
    print(f"Quick mode: {quick}")
    print(f"Coverage simulations (M): {M}")
    print(f"Report output: {output_dir}")
    if regime:
        print(f"Running only: Regime {regime.upper()}")
    print("\n" + "=" * 80)

    print("\n")
    print("The Three Lambda Regimes:")
    print("  Regime A: RCT + Known F_T      -> ComputeLambda (Monte Carlo)")
    print("  Regime B: Linear Model         -> AnalyticLambda (E[TT'|X])")
    print("  Regime C: Observational Logit  -> EstimateLambda (3-way split)")
    print("\n" + "=" * 80)

    # Regime A: RCT Logit
    if regime is None or regime.lower() == 'a':
        print("\n\n")
        from evals.regime_a.run_regime_a import run_regime_a
        results["regime_a"] = run_regime_a(quick=quick, verbose=verbose, M=M)

    # Regime B: Linear
    if regime is None or regime.lower() == 'b':
        print("\n\n")
        from evals.regime_b.run_regime_b import run_regime_b
        results["regime_b"] = run_regime_b(quick=quick, verbose=verbose, M=M)

    # Regime C: Observational Logit
    if regime is None or regime.lower() == 'c':
        print("\n\n")
        from evals.regime_c.run_regime_c import run_regime_c
        results["regime_c"] = run_regime_c(quick=quick, verbose=verbose, M=M)

    # Final Summary
    print("\n\n")
    print("#" * 80)
    print("# FINAL SUMMARY - ALL REGIMES")
    print("#" * 80)
    print("\n")

    print("=" * 70)
    print("EVALUATION SCORECARD")
    print("=" * 70)
    print(f"\n{'Regime':<15} {'Eval':<20} {'Status':<10}")
    print("-" * 50)

    all_pass = True
    for regime_name, regime_results in results.items():
        regime_display = regime_name.replace("_", " ").upper()
        for eval_name, result in regime_results.items():
            if isinstance(result, dict):
                if result.get("skipped"):
                    status = "SKIPPED"
                elif result.get("passed"):
                    status = "PASS"
                elif result.get("passed") is None:
                    status = "SKIPPED"
                else:
                    status = "FAIL"
                    all_pass = False
                print(f"{regime_display:<15} {eval_name:<20} {status:<10}")
                regime_display = ""  # Only print regime name once

    print("-" * 50)
    overall = "ALL PASS" if all_pass else "SOME FAILED"
    print(f"{'OVERALL':<35} {overall:<10}")
    print("=" * 70)

    # Summary table by regime
    print("\n--- Regime Summary ---\n")
    print(f"{'Regime':<15} {'Passed':<10} {'Failed':<10} {'Skipped':<10}")
    print("-" * 45)

    for regime_name, regime_results in results.items():
        passed = 0
        failed = 0
        skipped = 0
        for eval_name, result in regime_results.items():
            if isinstance(result, dict):
                if result.get("skipped") or result.get("passed") is None:
                    skipped += 1
                elif result.get("passed"):
                    passed += 1
                else:
                    failed += 1
        regime_display = regime_name.replace("_", " ").upper()
        print(f"{regime_display:<15} {passed:<10} {failed:<10} {skipped:<10}")

    print("\n" + "=" * 80)
    print("END OF CONSOLE REPORT")
    print("=" * 80)

    # Generate report files
    from evals.report import generate_report
    report_paths = generate_report(results, output_dir=output_dir)

    return results


# Backward compatibility: run_all_evals with old regime C evals only
def run_regime_c_only(verbose: bool = True, quick: bool = False) -> Dict[str, Any]:
    """Run only Regime C evaluations (backward compatible)."""
    return run_all_evals(verbose=verbose, quick=quick, regime='c')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run all evaluations")
    parser.add_argument("--quick", action="store_true", help="Quick mode with smaller samples (M=10)")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--regime", type=str, choices=['a', 'b', 'c', 'A', 'B', 'C'],
                        help="Run only specific regime")
    parser.add_argument("--M", type=int, default=None,
                        help="Number of MC simulations for coverage tests (default: 10 quick, 100 full)")
    parser.add_argument("--output-dir", type=str, default="evals/reports",
                        help="Directory for report output files")
    args = parser.parse_args()

    results = run_all_evals(
        verbose=not args.quiet,
        quick=args.quick,
        regime=args.regime,
        M=args.M,
        output_dir=args.output_dir
    )
