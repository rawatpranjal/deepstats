"""
Compare Python and R results.

Loads results from both implementations and prints comparison table.
"""

import json
from pathlib import Path


def main():
    results_dir = Path(__file__).parent / 'results'

    # Load results
    try:
        with open(results_dir / 'python_results.json') as f:
            python = json.load(f)
    except FileNotFoundError:
        print("ERROR: Python results not found. Run run_python.py first.")
        return

    try:
        with open(results_dir / 'r_results.json') as f:
            r = json.load(f)
    except FileNotFoundError:
        print("ERROR: R results not found. Run run_r.R first.")
        return

    print("=" * 70)
    print("R vs Python Comparison: FLM Influence Function Implementation")
    print("=" * 70)
    print()

    # Header
    print(f"{'Metric':<25} {'Python':>15} {'R':>15} {'Diff':>12}")
    print("-" * 70)

    # IF Results
    print("\n--- INFLUENCE FUNCTION ---")
    py_cov = python['if_coverage']
    r_cov = r['if_coverage']
    print(f"{'Coverage':<25} {py_cov:>14.1%} {r_cov:>14.1%} {abs(py_cov-r_cov):>11.1%}")

    py_se = python['if_se_estimated']
    r_se = r['if_se_estimated']
    print(f"{'SE (estimated)':<25} {py_se:>15.4f} {r_se:>15.4f} {abs(py_se-r_se):>12.4f}")

    py_ratio = python['if_se_ratio']
    r_ratio = r['if_se_ratio']
    print(f"{'SE Ratio':<25} {py_ratio:>15.2f} {r_ratio:>15.2f} {abs(py_ratio-r_ratio):>12.2f}")

    py_bias = python['if_bias']
    r_bias = r['if_bias']
    print(f"{'Bias':<25} {py_bias:>15.4f} {r_bias:>15.4f} {abs(py_bias-r_bias):>12.4f}")

    # Naive Results
    print("\n--- NAIVE (baseline) ---")
    py_cov_n = python['naive_coverage']
    r_cov_n = r['naive_coverage']
    print(f"{'Coverage':<25} {py_cov_n:>14.1%} {r_cov_n:>14.1%} {abs(py_cov_n-r_cov_n):>11.1%}")

    py_ratio_n = python['naive_se_ratio']
    r_ratio_n = r['naive_se_ratio']
    print(f"{'SE Ratio':<25} {py_ratio_n:>15.2f} {r_ratio_n:>15.2f} {abs(py_ratio_n-r_ratio_n):>12.2f}")

    # Pass/Fail Criteria
    print()
    print("=" * 70)
    print("PASS/FAIL CRITERIA")
    print("=" * 70)

    all_pass = True

    # Both achieve ~95% coverage
    py_cov_ok = 0.85 <= py_cov <= 1.0
    r_cov_ok = 0.85 <= r_cov <= 1.0
    status = "PASS" if (py_cov_ok and r_cov_ok) else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"Both IF coverage in [85%, 100%]: {status}")
    print(f"  Python: {py_cov:.1%} {'OK' if py_cov_ok else 'FAIL'}")
    print(f"  R:      {r_cov:.1%} {'OK' if r_cov_ok else 'FAIL'}")

    # Both SE ratios reasonable
    py_ratio_ok = 0.5 <= py_ratio <= 2.0
    r_ratio_ok = 0.5 <= r_ratio <= 2.0
    status = "PASS" if (py_ratio_ok and r_ratio_ok) else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"\nBoth IF SE ratio in [0.5, 2.0]: {status}")
    print(f"  Python: {py_ratio:.2f} {'OK' if py_ratio_ok else 'FAIL'}")
    print(f"  R:      {r_ratio:.2f} {'OK' if r_ratio_ok else 'FAIL'}")

    # Naive fails in both (shows IF is needed)
    py_naive_fails = py_cov_n < 0.7
    r_naive_fails = r_cov_n < 0.7
    status = "PASS" if (py_naive_fails and r_naive_fails) else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"\nBoth naive coverage < 70%: {status}")
    print(f"  Python: {py_cov_n:.1%} {'OK' if py_naive_fails else 'FAIL'}")
    print(f"  R:      {r_cov_n:.1%} {'OK' if r_naive_fails else 'FAIL'}")

    # Results are similar (within sampling error)
    cov_diff = abs(py_cov - r_cov)
    se_diff = abs(py_se - r_se) / max(py_se, r_se, 1e-6)
    similar = cov_diff < 0.2 and se_diff < 0.5
    status = "PASS" if similar else "NOTE"
    print(f"\nResults similar (coverage diff < 20%, SE diff < 50%): {status}")
    print(f"  Coverage diff: {cov_diff:.1%}")
    print(f"  SE relative diff: {se_diff:.1%}")

    print()
    print("=" * 70)
    if all_pass:
        print("OVERALL: PASS - Both implementations produce valid inference")
    else:
        print("OVERALL: CHECK - Review the failed criteria above")
    print("=" * 70)
    print()
    print("Note: Exact numerical match is NOT expected due to NN randomness.")
    print("What matters is that both achieve valid 95% coverage.")


if __name__ == '__main__':
    main()
