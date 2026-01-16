"""
================================================================================
EVAL 08: DOCUMENTATION QUICK START VERIFICATION
================================================================================

WHAT THIS TESTS:
    Verifies that the Quick Start examples in docs actually run and produce
    valid output. This prevents documentation from containing fabricated
    or placeholder values.

EXAMPLES TESTED:
    1. docs/index.md - Logit example (E[beta] = 0.5)
    2. docs/getting_started/quickstart.md - Linear example

PASS CRITERIA:
    - Output is finite (no NaN/Inf)
    - Standard error > 0
    - 95% CI covers ground truth
    - Diagnostics are reasonable (min_lambda_eigenvalue > 0)

--------------------------------------------------------------------------------
"""

import sys
import numpy as np
import torch

sys.path.insert(0, "/Users/pranjal/deepest/src")

from deep_inference import structural_dml


def run_logit_example() -> dict:
    """
    Run the logit example from docs/index.md.

    Ground truth: E[beta] = E[0.5 + 0.3*X1] = 0.5 (since E[X1] = 0)
    """
    print("=" * 78)
    print("LOGIT EXAMPLE (docs/index.md)")
    print("=" * 78)
    print()

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Heterogeneous logistic demand (binary outcomes)
    n = 2000
    X = np.random.randn(n, 5)
    T = np.random.randn(n)

    # Heterogeneous treatment effect: β(X) = 0.5 + 0.3*X₁
    alpha = 0.2 * X[:, 0]
    beta = 0.5 + 0.3 * X[:, 1]
    prob = 1 / (1 + np.exp(-(alpha + beta * T)))
    Y = np.random.binomial(1, prob).astype(float)

    # Ground truth
    mu_true = 0.5  # E[beta] = E[0.5 + 0.3*X1] = 0.5

    print(f"Ground truth: E[beta] = {mu_true}")
    print(f"Sample E[beta]: {beta.mean():.6f}")
    print()

    # Run influence function inference
    print("Running structural_dml (logit)...")
    result = structural_dml(
        Y=Y, T=T, X=X,
        family='logit',
        hidden_dims=[64, 32],
        epochs=100,
        n_folds=50
    )

    print()
    print(result.summary())
    print()

    # Validation checks
    checks = {
        'mu_hat_finite': np.isfinite(result.mu_hat),
        'se_positive': result.se > 0,
        'se_finite': np.isfinite(result.se),
        'ci_covers_truth': result.ci_lower <= mu_true <= result.ci_upper,
        'min_lambda_positive': result.diagnostics.get('min_lambda_eigenvalue', 0) > 0,
    }

    print("Validation Checks:")
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {check}: {status}")

    all_pass = all(checks.values())
    print()
    print(f"LOGIT EXAMPLE: {'PASS' if all_pass else 'FAIL'}")
    print()

    return {
        'passed': all_pass,
        'checks': checks,
        'mu_hat': result.mu_hat,
        'se': result.se,
        'ci_lower': result.ci_lower,
        'ci_upper': result.ci_upper,
        'mu_true': mu_true,
    }


def run_linear_example() -> dict:
    """
    Run the linear example from docs/getting_started/quickstart.md.

    Ground truth: E[beta] = E[cos(π*X0)*(X1>0) + 0.5*X2]
    Since X ~ N(0,1), this is approximately 0 (cos averages out, X1>0 is 50%)
    """
    print("=" * 78)
    print("LINEAR EXAMPLE (docs/getting_started/quickstart.md)")
    print("=" * 78)
    print()

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Example: Generate synthetic data
    n = 2000
    X = np.random.randn(n, 10)  # Covariates
    T = np.random.randn(n)       # Treatment
    beta_true = np.cos(np.pi * X[:, 0]) * (X[:, 1] > 0) + 0.5 * X[:, 2]
    Y = X[:, 0] + beta_true * T + np.random.randn(n)

    # Ground truth (empirical mean of beta_true)
    mu_true = beta_true.mean()

    print(f"Ground truth: E[beta] = {mu_true:.6f}")
    print()

    # Run influence function inference
    print("Running structural_dml (linear)...")
    result = structural_dml(
        Y=Y,
        T=T,
        X=X,
        family='linear',
        hidden_dims=[64, 32],
        epochs=100,
        n_folds=50,
        lr=0.01
    )

    print()
    print(result.summary())
    print()

    # Validation checks
    checks = {
        'mu_hat_finite': np.isfinite(result.mu_hat),
        'se_positive': result.se > 0,
        'se_finite': np.isfinite(result.se),
        'ci_covers_truth': result.ci_lower <= mu_true <= result.ci_upper,
        'min_lambda_positive': result.diagnostics.get('min_lambda_eigenvalue', 0) > 0,
    }

    print("Validation Checks:")
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {check}: {status}")

    all_pass = all(checks.values())
    print()
    print(f"LINEAR EXAMPLE: {'PASS' if all_pass else 'FAIL'}")
    print()

    return {
        'passed': all_pass,
        'checks': checks,
        'mu_hat': result.mu_hat,
        'se': result.se,
        'ci_lower': result.ci_lower,
        'ci_upper': result.ci_upper,
        'mu_true': mu_true,
    }


def main():
    """Run all documentation example verification."""
    print()
    print("=" * 78)
    print("EVAL 08: DOCUMENTATION QUICK START VERIFICATION")
    print("=" * 78)
    print()
    print("This eval verifies that Quick Start examples in the documentation")
    print("actually work and produce valid output.")
    print()

    results = {}

    # Run logit example
    results['logit'] = run_logit_example()

    # Run linear example
    results['linear'] = run_linear_example()

    # Summary
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print()

    all_pass = True
    for name, res in results.items():
        status = "PASS" if res['passed'] else "FAIL"
        print(f"{name.upper()}: {status}")
        print(f"  mu_hat: {res['mu_hat']:.6f}")
        print(f"  se: {res['se']:.6f}")
        print(f"  95% CI: [{res['ci_lower']:.6f}, {res['ci_upper']:.6f}]")
        print(f"  mu_true: {res['mu_true']:.6f}")
        print(f"  CI covers truth: {res['checks']['ci_covers_truth']}")
        print()
        if not res['passed']:
            all_pass = False

    print("=" * 78)
    if all_pass:
        print("EVAL 08 DOCUMENTATION VERIFICATION: PASS")
    else:
        print("EVAL 08 DOCUMENTATION VERIFICATION: FAIL")
    print("=" * 78)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
