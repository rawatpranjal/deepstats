"""
Rigorous Linear Validation Study

Focus on LINEAR family only with validated settings from CHANGELOG:
- N=20,000 achieved 96% coverage with Corr(β)=0.58
- K=50 folds required for valid IF

This is the canonical example for documentation.
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))
from src2 import structural_dml

# =============================================================================
# RIGOROUS CONFIGURATION (validated settings)
# =============================================================================

CONFIG = {
    'N': 20_000,           # Validated: achieves Corr(β)=0.58, Coverage=96%
    'M': 100,              # More sims for precise coverage estimate
    'n_folds': 50,         # Required for valid IF
    'd': 10,               # Standard dimension
    'epochs': 100,         # Sufficient with early stopping
    'lr': 0.01,            # Standard learning rate
    'hidden_dims': [64, 32],  # Standard architecture (validated)
}

# =============================================================================
# VALIDATED DGP (matches CHANGELOG experiments)
# =============================================================================

def generate_linear_data(N: int, d: int = 10) -> dict:
    """
    Generate linear DGP with heterogeneous treatment effects.

    Model: Y = α*(X) + β*(X)·T + ε

    This is the standard DGP used in the CHANGELOG validation experiments.
    """
    # Covariates
    X = np.random.randn(N, d)

    # True functions (moderate heterogeneity)
    alpha_star = (
        np.sin(np.pi * X[:, 0]) +
        X[:, 1] ** 2 -
        np.cos(np.pi * X[:, 2])
    )

    beta_star = (
        0.5 +  # baseline effect
        0.3 * X[:, 0] +
        0.2 * np.tanh(X[:, 1]) -
        0.1 * X[:, 2]
    )

    # Treatment (confounded)
    T = 0.3 * beta_star + 0.2 * X[:, 3:6].sum(axis=1) + np.random.randn(N)

    # Outcome
    Y = alpha_star + beta_star * T + np.random.randn(N)

    # True target
    mu_true = beta_star.mean()

    return {
        'X': X,
        'T': T,
        'Y': Y,
        'alpha_star': alpha_star,
        'beta_star': beta_star,
        'mu_true': mu_true,
    }

# =============================================================================
# SINGLE SIMULATION
# =============================================================================

def run_single_sim(sim_id: int, config: dict) -> dict:
    """Run one simulation with full diagnostics."""

    data = generate_linear_data(config['N'], config['d'])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = structural_dml(
            Y=data['Y'],
            T=data['T'],
            X=data['X'],
            family='linear',
            n_folds=config['n_folds'],
            hidden_dims=config['hidden_dims'],
            epochs=config['epochs'],
            lr=config['lr'],
            verbose=False,
        )

    # Extract estimates
    alpha_hat = result.theta_hat[:, 0]
    beta_hat = result.theta_hat[:, 1]

    # Naive inference
    naive_se = beta_hat.std() / np.sqrt(len(beta_hat))
    naive_ci_lo = result.mu_naive - 1.96 * naive_se
    naive_ci_hi = result.mu_naive + 1.96 * naive_se
    naive_covered = naive_ci_lo <= data['mu_true'] <= naive_ci_hi

    # IF inference
    if_covered = result.ci_lower <= data['mu_true'] <= result.ci_upper

    # Correlations
    def safe_corr(a, b):
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return 0.0
        return np.corrcoef(a, b)[0, 1]

    return {
        'sim_id': sim_id,
        'mu_true': data['mu_true'],

        # Naive
        'naive_mu': result.mu_naive,
        'naive_se': naive_se,
        'naive_covered': naive_covered,

        # IF
        'if_mu': result.mu_hat,
        'if_se': result.se,
        'if_covered': if_covered,

        # Parameter recovery
        'rmse_alpha': np.sqrt(((alpha_hat - data['alpha_star']) ** 2).mean()),
        'rmse_beta': np.sqrt(((beta_hat - data['beta_star']) ** 2).mean()),
        'corr_alpha': safe_corr(alpha_hat, data['alpha_star']),
        'corr_beta': safe_corr(beta_hat, data['beta_star']),

        # Diagnostics
        'min_eigenvalue': result.diagnostics.get('min_lambda_eigenvalue', 0),
        'pct_regularized': result.diagnostics.get('pct_regularized', 0),
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("RIGOROUS LINEAR VALIDATION STUDY")
    print("=" * 60)
    print(f"N = {CONFIG['N']:,}")
    print(f"M = {CONFIG['M']} simulations")
    print(f"K = {CONFIG['n_folds']} folds")
    print(f"Architecture: {CONFIG['hidden_dims']}")
    print(f"Epochs: {CONFIG['epochs']}, lr: {CONFIG['lr']}")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = []
    progress_file = results_dir / f'linear_validation_{timestamp}_progress.csv'

    with tqdm(range(CONFIG['M']), desc='LINEAR', ncols=80) as pbar:
        for sim_id in pbar:
            result = run_single_sim(sim_id, CONFIG)
            results.append(result)

            # Update progress
            df = pd.DataFrame(results)
            naive_cov = df['naive_covered'].mean()
            if_cov = df['if_covered'].mean()
            corr_b = df['corr_beta'].mean()

            pbar.set_postfix({
                'Nv': f'{naive_cov:.0%}',
                'IF': f'{if_cov:.0%}',
                'Corr': f'{corr_b:.2f}'
            })

            # Incremental save
            df.to_csv(progress_file, index=False)

    df = pd.DataFrame(results)

    # Final results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Parameter Recovery
    print(f"\nParameter Recovery:")
    print(f"  RMSE(α): {df['rmse_alpha'].mean():.4f} ± {df['rmse_alpha'].std():.4f}")
    print(f"  RMSE(β): {df['rmse_beta'].mean():.4f} ± {df['rmse_beta'].std():.4f}")
    print(f"  Corr(α): {df['corr_alpha'].mean():.3f} ± {df['corr_alpha'].std():.3f}")
    print(f"  Corr(β): {df['corr_beta'].mean():.3f} ± {df['corr_beta'].std():.3f}")

    # Inference Comparison
    print(f"\nInference Comparison:")
    print(f"{'':20} {'Naive':>12} {'IF':>12}")
    print("-" * 44)

    naive_se_emp = df['naive_mu'].std()
    if_se_emp = df['if_mu'].std()
    naive_se_est = df['naive_se'].mean()
    if_se_est = df['if_se'].mean()

    print(f"{'SE (empirical)':20} {naive_se_emp:>12.4f} {if_se_emp:>12.4f}")
    print(f"{'SE (estimated)':20} {naive_se_est:>12.4f} {if_se_est:>12.4f}")

    naive_ratio = naive_se_est / naive_se_emp if naive_se_emp > 0 else float('inf')
    if_ratio = if_se_est / if_se_emp if if_se_emp > 0 else float('inf')
    print(f"{'SE Ratio':20} {naive_ratio:>12.2f} {if_ratio:>12.2f}")

    naive_cov = df['naive_covered'].mean()
    if_cov = df['if_covered'].mean()
    print(f"{'Coverage':20} {naive_cov:>11.0%} {if_cov:>11.0%}")

    # Diagnostics
    print(f"\nDiagnostics:")
    print(f"  Min eigenvalue: {df['min_eigenvalue'].mean():.4f}")
    print(f"  Pct regularized: {df['pct_regularized'].mean():.1f}%")

    # Save final
    final_file = results_dir / f'linear_validation_{timestamp}.csv'
    df.to_csv(final_file, index=False)

    # Validation check
    print("\n" + "=" * 60)
    print("VALIDATION CHECK")
    print("=" * 60)

    if 0.93 <= if_cov <= 0.97:
        print(f"✓ IF Coverage: {if_cov:.0%} (TARGET: 93-97%)")
    else:
        print(f"✗ IF Coverage: {if_cov:.0%} (TARGET: 93-97%)")

    if 0.9 <= if_ratio <= 1.2:
        print(f"✓ SE Ratio: {if_ratio:.2f} (TARGET: 0.9-1.2)")
    else:
        print(f"✗ SE Ratio: {if_ratio:.2f} (TARGET: 0.9-1.2)")

    print(f"\nResults: {final_file}")
    print("=" * 60)


if __name__ == '__main__':
    main()
