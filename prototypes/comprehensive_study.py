"""
Comprehensive Simulation Study for FLM Influence Function Validation.

Publication-ready validation with:
- Large samples (N=50,000)
- Many replications (M=100)
- High-dimensional covariates (d=20: 10 signal + 10 noise)
- Multiple model families

Usage:
    python3 prototypes/comprehensive_study.py --N 50000 --M 100 --K 50 --family poisson
"""

import sys
sys.path.insert(0, '/Users/pranjal/deepest')

import argparse
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from src2.core.algorithm import structural_dml_core
from src2.families import get_family


def generate_dgp(
    n: int,
    d: int = 20,
    family: str = 'poisson',
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Generate DGP data with high-dimensional covariates.

    Args:
        n: Sample size
        d: Number of covariates (first 10 are signal, rest are noise)
        family: Model family ('poisson', 'linear', 'logit')
        seed: Random seed

    Returns:
        X, T, Y, mu_true, alpha_true, beta_true
    """
    if seed is not None:
        np.random.seed(seed)

    # High-dimensional covariates: d features (10 signal + noise)
    X = np.random.uniform(-1, 1, (n, d))

    # Continuous treatment
    T = np.random.uniform(-1, 1, n)

    # True heterogeneous parameters (only use first 4 features for simplicity)
    alpha_star = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    beta_star = 0.3 * X[:, 2] - 0.2 * X[:, 3]

    # Scale for different families
    if family == 'poisson':
        scale = 0.3
        alpha_s = alpha_star * scale
        beta_s = beta_star * scale

        # Generate Y from Poisson
        eta = np.clip(alpha_s + beta_s * T, -5, 5)
        lam = np.exp(eta)
        Y = np.random.poisson(lam).astype(float)

        mu_true = beta_star.mean() * scale

    elif family == 'linear':
        scale = 1.0
        alpha_s = alpha_star * scale
        beta_s = beta_star * scale

        # Generate Y from linear model
        eps = np.random.randn(n) * 0.5
        Y = alpha_s + beta_s * T + eps

        mu_true = beta_star.mean() * scale

    elif family == 'logit':
        scale = 0.5
        alpha_s = alpha_star * scale
        beta_s = beta_star * scale

        # Generate Y from logit model
        eta = alpha_s + beta_s * T
        prob = 1 / (1 + np.exp(-eta))
        Y = (np.random.rand(n) < prob).astype(float)

        mu_true = beta_star.mean() * scale

    else:
        raise ValueError(f"Unknown family: {family}")

    return X, T, Y, mu_true, alpha_s, beta_s


def run_single_sim(
    N: int,
    K: int,
    family: str,
    d: int,
    seed: int
) -> Dict[str, Any]:
    """Run a single simulation."""

    fam = get_family(family)

    X, T, Y, mu_true, alpha_true, beta_true = generate_dgp(
        n=N, d=d, family=family, seed=seed
    )

    # Determine if three-way splitting is needed
    three_way = fam.hessian_depends_on_theta() if hasattr(fam, 'hessian_depends_on_theta') else False

    result = structural_dml_core(
        Y=Y, T=T, X=X,
        loss_fn=fam.loss,
        target_fn=fam.default_target,
        theta_dim=fam.theta_dim,
        n_folds=K,
        hidden_dims=[64, 32],
        epochs=100,
        lr=0.01,
        three_way=three_way,
        gradient_fn=fam.gradient if hasattr(fam, 'gradient') else None,
        hessian_fn=fam.hessian if hasattr(fam, 'hessian') else None,
        lambda_method='aggregate',
        verbose=False,
    )

    # Get estimated parameters
    alpha_hat = result.theta_hat[:, 0]
    beta_hat = result.theta_hat[:, 1]

    # Compute correlations
    corr_alpha = np.corrcoef(alpha_true, alpha_hat)[0, 1]
    corr_beta = np.corrcoef(beta_true, beta_hat)[0, 1]

    # Compute RMSE
    rmse_alpha = np.sqrt(np.mean((alpha_hat - alpha_true) ** 2))
    rmse_beta = np.sqrt(np.mean((beta_hat - beta_true) ** 2))

    # Coverage
    covered = result.ci_lower <= mu_true <= result.ci_upper

    return {
        'mu_true': mu_true,
        'mu_hat': result.mu_hat,
        'se': result.se,
        'ci_lower': result.ci_lower,
        'ci_upper': result.ci_upper,
        'covered': covered,
        'corr_alpha': corr_alpha,
        'corr_beta': corr_beta,
        'rmse_alpha': rmse_alpha,
        'rmse_beta': rmse_beta,
        'pct_regularized': result.diagnostics.get('pct_regularized', 0),
        'min_eigenvalue': result.diagnostics.get('min_lambda_eigenvalue', 0),
    }


def run_study(
    N: int,
    M: int,
    K: int,
    family: str,
    d: int = 20,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run comprehensive simulation study.

    Args:
        N: Sample size
        M: Number of simulations
        K: Number of cross-fitting folds
        family: Model family
        d: Number of covariates
        verbose: Print progress

    Returns:
        DataFrame with all simulation results
    """

    results = []
    start_time = datetime.now()

    if verbose:
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE STUDY: {family.upper()}")
        print(f"{'='*70}")
        print(f"N={N:,}, M={M}, K={K}, d={d}")
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

    for sim in range(M):
        try:
            r = run_single_sim(N, K, family, d, seed=sim)
            r['sim_id'] = sim
            r['family'] = family
            r['N'] = N
            r['K'] = K
            r['d'] = d
            results.append(r)

            if verbose and (sim + 1) % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                eta = elapsed / (sim + 1) * (M - sim - 1)
                print(f"  Sim {sim + 1:3d}/{M} | "
                      f"Elapsed: {elapsed:.1f}m | "
                      f"ETA: {eta:.1f}m | "
                      f"Cov: {np.mean([r['covered'] for r in results])*100:.1f}%")

        except Exception as e:
            print(f"  Sim {sim} failed: {e}")
            continue

    df = pd.DataFrame(results)

    # Print summary
    if verbose and len(df) > 0:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")

        coverage = df['covered'].mean() * 100
        se_ratio = df['se'].mean() / df['mu_hat'].std()

        print(f"\nInference Metrics:")
        print(f"  Coverage: {coverage:.1f}%")
        print(f"  SE Ratio: {se_ratio:.2f}")
        print(f"  Mean Bias: {(df['mu_hat'] - df['mu_true']).mean():.5f}")
        print(f"  Reg Rate: {df['pct_regularized'].mean():.1f}%")

        print(f"\nParameter Recovery:")
        print(f"  Corr(α): {df['corr_alpha'].mean():.3f}")
        print(f"  Corr(β): {df['corr_beta'].mean():.3f}")
        print(f"  RMSE(α): {df['rmse_alpha'].mean():.4f}")
        print(f"  RMSE(β): {df['rmse_beta'].mean():.4f}")

        # Check targets
        print(f"\n{'='*70}")
        print("VALIDATION")
        print(f"{'='*70}")

        cov_pass = 93 <= coverage <= 97
        se_pass = 0.9 <= se_ratio <= 1.1
        alpha_pass = df['corr_alpha'].mean() > 0.90
        beta_pass = df['corr_beta'].mean() > 0.70

        print(f"  Coverage 93-97%: {coverage:.1f}% {'PASS' if cov_pass else 'FAIL'}")
        print(f"  SE Ratio 0.9-1.1: {se_ratio:.2f} {'PASS' if se_pass else 'FAIL'}")
        print(f"  Corr(α) > 0.90: {df['corr_alpha'].mean():.3f} {'PASS' if alpha_pass else 'FAIL'}")
        print(f"  Corr(β) > 0.70: {df['corr_beta'].mean():.3f} {'PASS' if beta_pass else 'FAIL'}")

        overall = cov_pass and se_pass and alpha_pass and beta_pass
        print(f"\n  Overall: {'PASS' if overall else 'FAIL'}")

        elapsed = (datetime.now() - start_time).total_seconds() / 60
        print(f"\nTotal time: {elapsed:.1f} minutes")

    return df


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Simulation Study')
    parser.add_argument('--N', type=int, default=50000, help='Sample size')
    parser.add_argument('--M', type=int, default=100, help='Number of simulations')
    parser.add_argument('--K', type=int, default=50, help='Number of folds')
    parser.add_argument('--family', type=str, default='poisson',
                        choices=['poisson', 'linear', 'logit'],
                        help='Model family')
    parser.add_argument('--d', type=int, default=20, help='Number of covariates')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path')

    args = parser.parse_args()

    # Run study
    df = run_study(
        N=args.N,
        M=args.M,
        K=args.K,
        family=args.family,
        d=args.d,
        verbose=True
    )

    # Save results
    if args.output is None:
        args.output = f'/Users/pranjal/deepest/results/comprehensive_{args.family}_N{args.N}_M{args.M}.csv'

    # Create results directory if needed
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
