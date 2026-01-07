"""
Full structural_dml test with different Lambda methods.

Tests Aggregate vs MLP Lambda estimation in the full cross-fitting algorithm.
"""

import sys
sys.path.insert(0, '/Users/pranjal/deepest')

import numpy as np
import torch
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

from src2 import structural_dml
from src2.core.algorithm import structural_dml_core
from src2.core.lambda_estimator import LambdaEstimator, AggregateLambdaEstimator
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


def run_mc_test(
    M: int = 30,
    N: int = 2000,
    n_folds: int = 50,
    lambda_method: str = 'mlp',
    binary_t: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run Monte Carlo test with specified Lambda method."""

    results = []
    fam = get_family('logit')

    for sim in range(M):
        if verbose and (sim + 1) % 10 == 0:
            print(f"  Simulation {sim + 1}/{M}...")

        try:
            # Generate data
            X, T, Y, mu_true = generate_logit_dgp(N, binary_t=binary_t, seed=sim)

            # Run structural DML
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
                lambda_method=lambda_method,  # Pass lambda method
                verbose=False,
            )

            # Check coverage
            covered = result.ci_lower <= mu_true <= result.ci_upper

            results.append({
                'mu_true': mu_true,
                'mu_hat': result.mu_hat,
                'se': result.se,
                'covered': covered,
                'min_eigenvalue': result.diagnostics.get('min_lambda_eigenvalue', 0),
                'pct_regularized': result.diagnostics.get('pct_regularized', 0),
            })
        except Exception as e:
            print(f"  Simulation {sim} failed: {e}")
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
        'binary_t': binary_t,
        'M': len(results),
        'n_folds': n_folds,
        'coverage': coverages.mean() * 100,
        'se_ratio': se_ratio,
        'mean_pct_regularized': pct_regs.mean(),
        'mean_min_eigenvalue': min_eigs.mean(),
        'mean_mu_hat': mu_hats.mean(),
        'empirical_sd': empirical_sd,
        'mean_se': mean_se,
    }


def main():
    """Run comparison test."""
    print("=" * 70)
    print("FULL STRUCTURAL_DML TEST: Lambda Method Comparison")
    print("=" * 70)

    M = 30  # Simulations
    N = 2000  # Sample size
    K = 50  # Folds

    test_configs = [
        ('mlp', True),
        ('ridge', True),
        ('aggregate', True),
    ]

    print(f"\nConfiguration: M={M}, N={N}, K={K}")
    print("-" * 70)

    results_table = []

    for lambda_method, binary_t in test_configs:
        t_type = "Binary" if binary_t else "Continuous"
        print(f"\nTesting: {lambda_method.upper()} with {t_type} T...")

        result = run_mc_test(
            M=M,
            N=N,
            n_folds=K,
            lambda_method=lambda_method,
            binary_t=binary_t,
            verbose=True
        )

        results_table.append(result)

        print(f"  Coverage: {result['coverage']:.1f}%")
        print(f"  SE Ratio: {result['se_ratio']:.2f}")
        print(f"  Regularization Rate: {result['mean_pct_regularized']:.1f}%")
        print(f"  Min Eigenvalue: {result['mean_min_eigenvalue']:.6f}")

    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<12} {'Coverage':<12} {'SE Ratio':<12} {'Reg Rate':<12} {'Min Eig':<12}")
    print("-" * 60)

    for r in results_table:
        print(f"{r['lambda_method']:<12} {r['coverage']:.1f}%        {r['se_ratio']:.2f}         {r['mean_pct_regularized']:.1f}%         {r['mean_min_eigenvalue']:.6f}")


if __name__ == "__main__":
    main()
