"""
Comprehensive Validation Suite for src2 Structural Deep Learning

Tests:
1. Linear DGP coverage (baseline - must pass)
2. Logit DGP coverage (three-way splitting)
3. Sample size robustness
4. Custom loss function (pure autodiff)
"""

import sys
sys.path.insert(0, '/Users/pranjal/deepest')

import numpy as np
import torch
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time
from joblib import Parallel, delayed

from src2 import structural_dml, compute_coverage
from src.deepstats.dgp import get_dgp


@dataclass
class ValidationResult:
    """Results from a validation run."""
    family: str
    n_sims: int
    n_obs: int
    n_folds: int
    mu_true: float
    coverage: float
    se_ratio: float
    bias: float
    rmse: float
    naive_coverage: float
    mean_se: float
    empirical_sd: float
    elapsed_time: float


def run_single_simulation(
    dgp,
    family: str,
    n_obs: int,
    n_folds: int,
    epochs: int,
    seed: int,
    mu_true: float,
    use_custom_loss: bool = False,
) -> Dict:
    """Run a single simulation."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate data
    data = dgp.generate(n_obs)

    # Run structural DML
    if use_custom_loss:
        # Pure autodiff mode - define loss manually
        def custom_linear_loss(y, t, theta):
            alpha, beta = theta[:, 0], theta[:, 1]
            mu = alpha + beta * t
            return (y - mu) ** 2

        result = structural_dml(
            Y=data.Y,
            T=data.T,
            X=data.X,
            loss_fn=custom_linear_loss,
            theta_dim=2,
            n_folds=n_folds,
            epochs=epochs,
            hidden_dims=[64, 32],
            verbose=False,
        )
    else:
        result = structural_dml(
            Y=data.Y,
            T=data.T,
            X=data.X,
            family=family,
            n_folds=n_folds,
            epochs=epochs,
            hidden_dims=[64, 32],
            verbose=False,
        )

    covered = compute_coverage(mu_true, result)

    return {
        'mu_hat': result.mu_hat,
        'se': result.se,
        'ci_lower': result.ci_lower,
        'ci_upper': result.ci_upper,
        'covered': covered,
        'mu_naive': result.mu_naive,
        'three_way': result.diagnostics.get('three_way', None),
    }


def validate_family(
    family: str,
    n_sims: int = 100,
    n_obs: int = 2000,
    n_folds: int = 20,
    epochs: int = 50,
    n_jobs: int = -1,
    use_custom_loss: bool = False,
) -> ValidationResult:
    """
    Run full validation for a family.

    Args:
        family: Family name ('linear', 'logit')
        n_sims: Number of simulations
        n_obs: Sample size per simulation
        n_folds: Number of cross-fitting folds
        epochs: Training epochs per fold
        n_jobs: Number of parallel jobs (-1 for all cores)
        use_custom_loss: If True, use custom loss (autodiff mode)

    Returns:
        ValidationResult with coverage, SE ratio, etc.
    """
    print(f"\n{'='*60}")
    print(f"VALIDATING: {family.upper()}")
    print(f"M={n_sims}, N={n_obs}, K={n_folds}, epochs={epochs}")
    if use_custom_loss:
        print("MODE: Custom loss function (pure autodiff)")
    print('='*60)

    start_time = time.time()

    # Get DGP and true mu
    dgp = get_dgp(family)
    mu_true = dgp.compute_true_mu()
    print(f"True mu*: {mu_true:.4f}")

    # Run simulations in parallel
    print(f"Running {n_sims} simulations...")

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_simulation)(
            dgp=dgp,
            family=family,
            n_obs=n_obs,
            n_folds=n_folds,
            epochs=epochs,
            seed=sim,
            mu_true=mu_true,
            use_custom_loss=use_custom_loss,
        )
        for sim in range(n_sims)
    )

    elapsed = time.time() - start_time

    # Aggregate results
    mu_hats = np.array([r['mu_hat'] for r in results])
    ses = np.array([r['se'] for r in results])
    coverages = np.array([r['covered'] for r in results])
    mu_naives = np.array([r['mu_naive'] for r in results])

    # Compute metrics
    coverage = coverages.mean()
    empirical_sd = mu_hats.std()
    mean_se = ses.mean()
    se_ratio = mean_se / empirical_sd if empirical_sd > 0 else float('inf')
    bias = mu_hats.mean() - mu_true
    rmse = np.sqrt(np.mean((mu_hats - mu_true) ** 2))

    # Naive coverage
    naive_se = mu_naives.std()
    naive_covered = np.mean(
        (mu_naives - 1.96 * mean_se <= mu_true) &
        (mu_true <= mu_naives + 1.96 * mean_se)
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: {family.upper()}")
    print('='*60)
    print(f"True mu*:            {mu_true:.4f}")
    print(f"Mean mu_hat:         {mu_hats.mean():.4f}")
    print(f"Bias:                {bias:.4f}")
    print(f"Empirical SD:        {empirical_sd:.4f}")
    print(f"Mean SE:             {mean_se:.4f}")
    print(f"SE Ratio:            {se_ratio:.3f}")
    print(f"RMSE:                {rmse:.4f}")
    print()
    print(f"INFLUENCE COVERAGE:  {coverage*100:.1f}%")
    print(f"Naive coverage:      {naive_covered*100:.1f}%")
    print(f"Elapsed time:        {elapsed:.1f}s")

    # Status
    if 0.90 <= coverage <= 0.98:
        print(f"\n[PASS] Coverage {coverage*100:.1f}% in acceptable range (90-98%)")
    else:
        print(f"\n[FAIL] Coverage {coverage*100:.1f}% OUTSIDE acceptable range!")

    if 0.85 <= se_ratio <= 1.15:
        print(f"[PASS] SE Ratio {se_ratio:.2f} in acceptable range (0.85-1.15)")
    else:
        print(f"[WARN] SE Ratio {se_ratio:.2f} outside ideal range")

    return ValidationResult(
        family=family,
        n_sims=n_sims,
        n_obs=n_obs,
        n_folds=n_folds,
        mu_true=mu_true,
        coverage=coverage,
        se_ratio=se_ratio,
        bias=bias,
        rmse=rmse,
        naive_coverage=naive_covered,
        mean_se=mean_se,
        empirical_sd=empirical_sd,
        elapsed_time=elapsed,
    )


def validate_sample_sizes(
    family: str,
    sample_sizes: List[int] = [500, 1000, 2000, 5000],
    n_sims: int = 50,
    n_folds: int = 20,
    epochs: int = 50,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Validate across different sample sizes.

    Coverage should be stable across N.
    RMSE should decrease as N increases.
    """
    print(f"\n{'='*60}")
    print(f"SAMPLE SIZE ROBUSTNESS: {family.upper()}")
    print(f"N = {sample_sizes}")
    print('='*60)

    results = []

    for n_obs in sample_sizes:
        result = validate_family(
            family=family,
            n_sims=n_sims,
            n_obs=n_obs,
            n_folds=n_folds,
            epochs=epochs,
            n_jobs=n_jobs,
        )
        results.append({
            'N': n_obs,
            'Coverage': result.coverage,
            'SE_Ratio': result.se_ratio,
            'Bias': result.bias,
            'RMSE': result.rmse,
        })

    df = pd.DataFrame(results)

    print(f"\n{'='*60}")
    print(f"SAMPLE SIZE ROBUSTNESS SUMMARY: {family.upper()}")
    print('='*60)
    print(df.to_string(index=False))

    # Check if coverage is stable
    coverage_stable = all(0.88 <= r['Coverage'] <= 0.99 for r in results)
    rmse_decreasing = all(
        results[i]['RMSE'] >= results[i+1]['RMSE'] * 0.9  # Allow 10% noise
        for i in range(len(results) - 1)
    )

    print()
    if coverage_stable:
        print("[PASS] Coverage stable across sample sizes")
    else:
        print("[WARN] Coverage varies significantly with N")

    if rmse_decreasing:
        print("[PASS] RMSE decreases with N (as expected)")
    else:
        print("[WARN] RMSE not consistently decreasing")

    return df


def run_full_validation_suite(
    n_sims_main: int = 100,
    n_sims_robustness: int = 50,
    n_obs: int = 2000,
    n_folds: int = 20,
    epochs: int = 50,
    n_jobs: int = -1,
) -> Dict:
    """
    Run the complete validation suite.

    Returns:
        Dictionary with all results
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE VALIDATION SUITE")
    print("="*60)
    print(f"Main tests: M={n_sims_main}, N={n_obs}, K={n_folds}")
    print(f"Robustness tests: M={n_sims_robustness}")

    total_start = time.time()
    results = {}

    # Test 1: Linear DGP (baseline)
    results['linear'] = validate_family(
        family='linear',
        n_sims=n_sims_main,
        n_obs=n_obs,
        n_folds=n_folds,
        epochs=epochs,
        n_jobs=n_jobs,
    )

    # Test 2: Logit DGP (three-way splitting)
    results['logit'] = validate_family(
        family='logit',
        n_sims=n_sims_main,
        n_obs=n_obs,
        n_folds=n_folds,
        epochs=epochs,
        n_jobs=n_jobs,
    )

    # Test 3: Custom loss (pure autodiff)
    results['custom_linear'] = validate_family(
        family='linear',
        n_sims=n_sims_robustness,
        n_obs=n_obs,
        n_folds=n_folds,
        epochs=epochs,
        n_jobs=n_jobs,
        use_custom_loss=True,
    )

    # Test 4: Sample size robustness (Linear)
    results['linear_robustness'] = validate_sample_sizes(
        family='linear',
        sample_sizes=[500, 1000, 2000],
        n_sims=n_sims_robustness,
        n_folds=n_folds,
        epochs=epochs,
        n_jobs=n_jobs,
    )

    # Test 5: Sample size robustness (Logit)
    results['logit_robustness'] = validate_sample_sizes(
        family='logit',
        sample_sizes=[500, 1000, 2000],
        n_sims=n_sims_robustness,
        n_folds=n_folds,
        epochs=epochs,
        n_jobs=n_jobs,
    )

    total_elapsed = time.time() - total_start

    # Final Summary
    print("\n" + "="*60)
    print("FINAL VALIDATION SUMMARY")
    print("="*60)

    summary_data = []
    for name, res in results.items():
        if isinstance(res, ValidationResult):
            status = "PASS" if 0.90 <= res.coverage <= 0.98 else "FAIL"
            summary_data.append({
                'Test': name,
                'Coverage': f"{res.coverage*100:.1f}%",
                'SE_Ratio': f"{res.se_ratio:.2f}",
                'RMSE': f"{res.rmse:.4f}",
                'Status': status,
            })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    print(f"\nTotal validation time: {total_elapsed/60:.1f} minutes")

    # Overall pass/fail
    all_pass = all(
        0.90 <= r.coverage <= 0.98
        for r in results.values()
        if isinstance(r, ValidationResult)
    )

    if all_pass:
        print("\n" + "="*60)
        print("OVERALL: ALL TESTS PASSED")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("OVERALL: SOME TESTS FAILED - INVESTIGATION NEEDED")
        print("="*60)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run validation suite")
    parser.add_argument("--quick", action="store_true", help="Quick test (M=20)")
    parser.add_argument("--full", action="store_true", help="Full test (M=100)")
    parser.add_argument("--family", type=str, default=None, help="Test single family")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs")

    args = parser.parse_args()

    if args.quick:
        n_sims_main = 20
        n_sims_robustness = 10
    elif args.full:
        n_sims_main = 100
        n_sims_robustness = 50
    else:
        n_sims_main = 50
        n_sims_robustness = 25

    if args.family:
        # Single family test
        result = validate_family(
            family=args.family,
            n_sims=n_sims_main,
            n_obs=2000,
            n_folds=20,
            epochs=50,
            n_jobs=args.n_jobs,
        )
    else:
        # Full suite
        results = run_full_validation_suite(
            n_sims_main=n_sims_main,
            n_sims_robustness=n_sims_robustness,
            n_obs=2000,
            n_folds=20,
            epochs=50,
            n_jobs=args.n_jobs,
        )
