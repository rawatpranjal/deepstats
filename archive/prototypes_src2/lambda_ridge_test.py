"""
Test Ridge Lambda estimator with different alpha values.

Goal: Find optimal Ridge alpha that achieves:
- SE Ratio closer to 1.0 (not 1.66)
- Coverage 93-97%
- Regularization < 10%
"""

import sys
sys.path.insert(0, '/Users/pranjal/deepest')

import numpy as np
import torch
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

from src2.core.algorithm import structural_dml_core
from src2.families import get_family


def generate_logit_dgp(n: int, binary_t: bool = True, seed: int = None):
    """Generate Logit DGP data."""
    if seed is not None:
        np.random.seed(seed)

    X = np.random.randn(n, 10)

    if binary_t:
        T = np.random.binomial(1, 0.5, n).astype(float)
    else:
        T = np.random.uniform(-1, 1, n)

    # Simple heterogeneity
    alpha_star = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    beta_star = 0.3 * X[:, 2] - 0.2 * X[:, 3]

    mu_true = beta_star.mean()

    # Generate Y
    logits = alpha_star + beta_star * T
    p = 1 / (1 + np.exp(-logits))
    Y = np.random.binomial(1, p).astype(float)

    return X, T, Y, mu_true


def run_single_sim(
    N: int,
    n_folds: int,
    lambda_method: str,
    ridge_alpha: float,
    binary_t: bool,
    seed: int,
) -> Dict[str, Any]:
    """Run a single simulation."""
    fam = get_family('logit')

    X, T, Y, mu_true = generate_logit_dgp(N, binary_t=binary_t, seed=seed)

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
        three_way=True,
        gradient_fn=fam.gradient,
        hessian_fn=fam.hessian,
        lambda_method=lambda_method,
        ridge_alpha=ridge_alpha,
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
    ridge_alpha: float,
    binary_t: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run Monte Carlo test with specified configuration."""

    results = []

    for sim in range(M):
        if verbose and (sim + 1) % 10 == 0:
            print(f"    Sim {sim + 1}/{M}...")

        try:
            r = run_single_sim(N, n_folds, lambda_method, ridge_alpha, binary_t, sim)
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
        'ridge_alpha': ridge_alpha,
        'binary_t': binary_t,
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
    """Run Ridge alpha comparison tests."""
    print("=" * 70)
    print("RIDGE ALPHA COMPARISON TEST")
    print("=" * 70)

    M = 30  # Simulations
    N = 2000  # Sample size
    K = 50  # Folds

    # Test configurations
    alphas_to_test = [0.01, 0.1, 1.0, 10.0, 100.0]

    print(f"\nConfiguration: M={M}, N={N}, K={K}")
    print("-" * 70)

    results_table = []

    # Test 1: Ridge with different alphas
    print("\n" + "=" * 70)
    print("TEST 1: Ridge Lambda with Different Alpha Values")
    print("=" * 70)

    for alpha in alphas_to_test:
        print(f"\nTesting Ridge(alpha={alpha})...")
        result = run_mc_test(
            M=M,
            N=N,
            n_folds=K,
            lambda_method='ridge',
            ridge_alpha=alpha,
            binary_t=True,
            verbose=True
        )
        results_table.append(result)

        print(f"  Coverage: {result['coverage']:.1f}%")
        print(f"  SE Ratio: {result['se_ratio']:.2f}")
        print(f"  Reg Rate: {result['mean_pct_regularized']:.1f}%")
        print(f"  Min Eig: {result['mean_min_eigenvalue']:.6f}")

    # Test 2: Aggregate baseline
    print("\n" + "=" * 70)
    print("TEST 2: Aggregate Lambda (Baseline)")
    print("=" * 70)

    print("\nTesting Aggregate...")
    result = run_mc_test(
        M=M,
        N=N,
        n_folds=K,
        lambda_method='aggregate',
        ridge_alpha=1.0,  # Doesn't matter for aggregate
        binary_t=True,
        verbose=True
    )
    results_table.append(result)

    print(f"  Coverage: {result['coverage']:.1f}%")
    print(f"  SE Ratio: {result['se_ratio']:.2f}")
    print(f"  Reg Rate: {result['mean_pct_regularized']:.1f}%")

    # Test 3: MLP baseline
    print("\n" + "=" * 70)
    print("TEST 3: MLP Lambda (Original Default)")
    print("=" * 70)

    print("\nTesting MLP...")
    result = run_mc_test(
        M=M,
        N=N,
        n_folds=K,
        lambda_method='mlp',
        ridge_alpha=1.0,  # Doesn't matter for MLP
        binary_t=True,
        verbose=True
    )
    results_table.append(result)

    print(f"  Coverage: {result['coverage']:.1f}%")
    print(f"  SE Ratio: {result['se_ratio']:.2f}")
    print(f"  Reg Rate: {result['mean_pct_regularized']:.1f}%")

    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<20} {'Alpha':<10} {'Coverage':<10} {'SE Ratio':<10} {'Reg Rate':<10}")
    print("-" * 60)

    for r in results_table:
        method = r['lambda_method']
        alpha = r['ridge_alpha'] if method == 'ridge' else '-'
        alpha_str = f"{alpha}" if alpha != '-' else '-'
        print(f"{method:<20} {alpha_str:<10} {r['coverage']:.1f}%      {r['se_ratio']:.2f}       {r['mean_pct_regularized']:.1f}%")

    # Find best configuration
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Filter valid results
    valid_results = [r for r in results_table if r['coverage'] >= 93 and r['mean_pct_regularized'] < 20]

    if valid_results:
        # Sort by SE ratio closest to 1.0
        best = min(valid_results, key=lambda x: abs(x['se_ratio'] - 1.0))
        print(f"\nBest configuration (SE ratio closest to 1.0):")
        print(f"  Method: {best['lambda_method']}")
        if best['lambda_method'] == 'ridge':
            print(f"  Alpha: {best['ridge_alpha']}")
        print(f"  Coverage: {best['coverage']:.1f}%")
        print(f"  SE Ratio: {best['se_ratio']:.2f}")
        print(f"  Reg Rate: {best['mean_pct_regularized']:.1f}%")
    else:
        print("\nNo configuration achieved both valid coverage (>=93%) and low regularization (<20%)")

    return results_table


def test_sample_size_effect():
    """Test if larger sample size improves SE ratio."""
    print("\n" + "=" * 70)
    print("TEST 4: Sample Size Effect on SE Ratio")
    print("=" * 70)

    M = 30
    K = 50

    sample_sizes = [2000, 5000]

    results = []
    for N in sample_sizes:
        print(f"\nTesting N={N} with Aggregate Lambda...")
        result = run_mc_test(
            M=M,
            N=N,
            n_folds=K,
            lambda_method='aggregate',
            ridge_alpha=1.0,
            binary_t=True,
            verbose=True
        )
        results.append(result)

        print(f"  Coverage: {result['coverage']:.1f}%")
        print(f"  SE Ratio: {result['se_ratio']:.2f}")
        print(f"  Empirical SD: {result['empirical_sd']:.4f}")
        print(f"  Mean SE: {result['mean_se']:.4f}")

    print("\n" + "=" * 70)
    print("SAMPLE SIZE COMPARISON")
    print("=" * 70)

    print(f"\n{'N':<10} {'Coverage':<10} {'SE Ratio':<10} {'Emp SD':<12} {'Mean SE':<12}")
    print("-" * 54)
    for r in results:
        print(f"{r['N']:<10} {r['coverage']:.1f}%      {r['se_ratio']:.2f}       {r['empirical_sd']:.4f}       {r['mean_se']:.4f}")


if __name__ == "__main__":
    results = main()

    # Uncomment to run sample size test
    # test_sample_size_effect()
