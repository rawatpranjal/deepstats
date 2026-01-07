"""
Poisson Family Validation Test.

Goal: Validate Poisson family implementation in src2 achieves:
- Coverage 93-97%
- SE Ratio 0.9-1.2
- Regularization Rate < 20%
"""

import sys
sys.path.insert(0, '/Users/pranjal/deepest')

import numpy as np
import torch
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

from src2.core.algorithm import structural_dml_core
from src2.families import get_family


def generate_poisson_dgp(n: int, seed: int = None):
    """Generate Poisson DGP data.

    Model: Y ~ Poisson(lambda), lambda = exp(alpha + beta*T)

    Uses simple heterogeneity and scaled parameters to keep
    lambda in reasonable range.
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.randn(n, 10)
    T = np.random.uniform(-1, 1, n)  # Continuous treatment

    # Simple heterogeneity
    alpha_star = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    beta_star = 0.3 * X[:, 2] - 0.2 * X[:, 3]

    # Scale parameters to keep lambda reasonable
    scale = 0.3
    alpha_s = alpha_star * scale
    beta_s = beta_star * scale

    # True target (scaled)
    mu_true = beta_star.mean() * scale

    # Generate Y from Poisson
    eta = np.clip(alpha_s + beta_s * T, -5, 5)
    lam = np.exp(eta)
    Y = np.random.poisson(lam).astype(float)

    return X, T, Y, mu_true


def run_single_sim(
    N: int,
    n_folds: int,
    lambda_method: str,
    seed: int,
) -> Dict[str, Any]:
    """Run a single simulation."""
    fam = get_family('poisson')

    X, T, Y, mu_true = generate_poisson_dgp(N, seed=seed)

    result = structural_dml_core(
        Y=Y,
        T=T,
        X=X,
        loss_fn=fam.loss,
        target_fn=fam.default_target,
        theta_dim=fam.theta_dim,
        n_folds=n_folds,
        hidden_dims=[64, 32],
        epochs=50,
        lr=0.01,
        three_way=True,  # Poisson Hessian depends on theta
        gradient_fn=fam.gradient,
        hessian_fn=fam.hessian,
        lambda_method=lambda_method,
        verbose=False,
    )

    covered = result.ci_lower <= mu_true <= result.ci_upper

    return {
        'mu_true': mu_true,
        'mu_hat': result.mu_hat,
        'se': result.se,
        'covered': covered,
        'min_eigenvalue': result.diagnostics.get('min_lambda_eigenvalue', 0),
        'pct_regularized': result.diagnostics.get('pct_regularized', 0),
    }


def run_mc_test(
    M: int,
    N: int,
    n_folds: int,
    lambda_method: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run Monte Carlo test with specified configuration."""

    results = []

    for sim in range(M):
        if verbose and (sim + 1) % 10 == 0:
            print(f"    Sim {sim + 1}/{M}...")

        try:
            r = run_single_sim(N, n_folds, lambda_method, sim)
            results.append(r)
        except Exception as e:
            print(f"    Sim {sim} failed: {e}")
            continue

    if not results:
        return {'error': 'All simulations failed'}

    # Aggregate results
    mu_hats = np.array([r['mu_hat'] for r in results])
    ses = np.array([r['se'] for r in results])
    coverages = np.array([r['covered'] for r in results])
    pct_regs = np.array([r['pct_regularized'] for r in results])
    min_eigs = np.array([r['min_eigenvalue'] for r in results])

    empirical_sd = mu_hats.std()
    mean_se = ses.mean()
    se_ratio = mean_se / empirical_sd if empirical_sd > 0 else float('inf')

    return {
        'lambda_method': lambda_method,
        'M': len(results),
        'N': N,
        'n_folds': n_folds,
        'coverage': coverages.mean() * 100,
        'se_ratio': se_ratio,
        'mean_pct_regularized': pct_regs.mean(),
        'mean_min_eigenvalue': min_eigs.mean(),
        'empirical_sd': empirical_sd,
        'mean_se': mean_se,
    }


def main():
    """Run Poisson validation tests."""
    print("=" * 70)
    print("POISSON FAMILY VALIDATION TEST")
    print("=" * 70)

    M = 30  # Simulations
    N = 2000  # Sample size
    K = 50  # Folds

    print(f"\nConfiguration: M={M}, N={N}, K={K}")
    print("-" * 70)

    results_table = []

    # Test 1: Aggregate Lambda (recommended)
    print("\n" + "=" * 70)
    print("TEST 1: Poisson with Aggregate Lambda")
    print("=" * 70)

    print("\nTesting Aggregate Lambda...")
    result = run_mc_test(
        M=M,
        N=N,
        n_folds=K,
        lambda_method='aggregate',
        verbose=True
    )
    results_table.append(result)

    print(f"  Coverage: {result['coverage']:.1f}%")
    print(f"  SE Ratio: {result['se_ratio']:.2f}")
    print(f"  Reg Rate: {result['mean_pct_regularized']:.1f}%")
    print(f"  Min Eigenvalue: {result['mean_min_eigenvalue']:.4f}")

    # Test 2: MLP Lambda (for comparison)
    print("\n" + "=" * 70)
    print("TEST 2: Poisson with MLP Lambda")
    print("=" * 70)

    print("\nTesting MLP Lambda...")
    result = run_mc_test(
        M=M,
        N=N,
        n_folds=K,
        lambda_method='mlp',
        verbose=True
    )
    results_table.append(result)

    print(f"  Coverage: {result['coverage']:.1f}%")
    print(f"  SE Ratio: {result['se_ratio']:.2f}")
    print(f"  Reg Rate: {result['mean_pct_regularized']:.1f}%")
    print(f"  Min Eigenvalue: {result['mean_min_eigenvalue']:.4f}")

    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<20} {'Coverage':<10} {'SE Ratio':<10} {'Reg Rate':<10} {'Min Eig':<12}")
    print("-" * 62)

    for r in results_table:
        method = r['lambda_method']
        print(f"{method:<20} {r['coverage']:.1f}%      {r['se_ratio']:.2f}       {r['mean_pct_regularized']:.1f}%        {r['mean_min_eigenvalue']:.4f}")

    # Validation assessment
    print("\n" + "=" * 70)
    print("VALIDATION ASSESSMENT")
    print("=" * 70)

    agg_result = results_table[0]  # Aggregate result

    coverage_pass = 93 <= agg_result['coverage'] <= 97
    se_ratio_pass = 0.9 <= agg_result['se_ratio'] <= 1.2
    reg_rate_pass = agg_result['mean_pct_regularized'] < 20

    print(f"\nAggregate Lambda Results:")
    print(f"  Coverage: {agg_result['coverage']:.1f}% {'PASS' if coverage_pass else 'FAIL'} (target: 93-97%)")
    print(f"  SE Ratio: {agg_result['se_ratio']:.2f} {'PASS' if se_ratio_pass else 'FAIL'} (target: 0.9-1.2)")
    print(f"  Reg Rate: {agg_result['mean_pct_regularized']:.1f}% {'PASS' if reg_rate_pass else 'FAIL'} (target: <20%)")

    overall_pass = coverage_pass and se_ratio_pass and reg_rate_pass
    print(f"\n  Overall: {'PASS' if overall_pass else 'FAIL'}")

    return results_table


if __name__ == "__main__":
    results = main()
