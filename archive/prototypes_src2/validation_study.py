"""
Comprehensive Validation Study for Structural Deep Learning (FLM Framework)

This script demonstrates that influence function corrections provide valid
95% confidence intervals for E[β(X)] under optimal conditions.

Key comparisons:
- Naive: μ̂ = mean(β̂), SE = std(β̂)/√N  (underestimates uncertainty)
- IF:    μ̂ = mean(ψ), SE from cross-fitting  (valid inference)

Usage:
    python prototypes/validation_study.py

Output:
    results/validation_study_YYYYMMDD_HHMMSS.csv
    results/validation_study_YYYYMMDD_HHMMSS.json
    results/validation_study_YYYYMMDD_HHMMSS/*.png
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src2 import structural_dml

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'N': 50_000,           # Sample size per simulation
    'M': 50,               # Number of simulations per family
    'n_folds': 50,         # Cross-fitting folds (K=50 required for valid IF)
    'd': 20,               # Covariate dimension (6 signal + 14 noise)
    'epochs': 100,         # Training epochs
    'lr': 0.001,           # Learning rate (slower for deep nets)
    'hidden_dims': [64, 64, 64, 32],  # Deep and narrow architecture
    'families': ['linear', 'logit', 'poisson'],  # Sequential run order
}

# =============================================================================
# DATA GENERATING PROCESS
# =============================================================================

def generate_data(family: str, N: int, d: int = 20) -> dict:
    """
    Generate synthetic data with complex nonlinear heterogeneity.

    True functions:
        α*(X) = sin(2πX₁) + X₂³ - 2cos(πX₃) + exp(X₄/3)·I(X₄>0) + 0.5·X₅·X₆
        β*(X) = cos(2πX₁)·sin(πX₂) + 0.8·tanh(3X₃) - 0.5·X₄² + 0.3·X₅·I(X₆>0)

    Args:
        family: 'linear', 'logit', or 'poisson'
        N: Sample size
        d: Covariate dimension (first 6 are signal features)

    Returns:
        dict with X, T, Y, alpha_star, beta_star, mu_true
    """
    # Covariates: d features, first 6 are signal
    X = np.random.uniform(-1, 1, (N, d))

    # Complex nonlinear alpha*(X) - baseline function
    alpha_star = (
        np.sin(2 * np.pi * X[:, 0]) +
        X[:, 1] ** 3 -
        2 * np.cos(np.pi * X[:, 2]) +
        np.exp(X[:, 3] / 3) * (X[:, 3] > 0) +
        0.5 * X[:, 4] * X[:, 5]
    )

    # Complex nonlinear beta*(X) - heterogeneous treatment effect
    beta_star = (
        np.cos(2 * np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1]) +
        0.8 * np.tanh(3 * X[:, 2]) -
        0.5 * X[:, 3] ** 2 +
        0.3 * X[:, 4] * (X[:, 5] > 0)
    )

    # Treatment (confounded - depends on beta and other X)
    T = 0.5 * beta_star + 0.2 * X[:, 6:10].sum(axis=1) + 0.5 * np.random.randn(N)

    # Outcome based on family
    if family == 'linear':
        # Y = α(X) + β(X)·T + ε
        Y = alpha_star + beta_star * T + np.random.randn(N)
        mu_true = beta_star.mean()

    elif family == 'logit':
        # P(Y=1|X,T) = sigmoid(0.5·α(X) + 0.5·β(X)·T)
        scale = 0.5
        linear_pred = scale * alpha_star + scale * beta_star * T
        prob = 1 / (1 + np.exp(-linear_pred))
        Y = np.random.binomial(1, prob, N).astype(float)
        mu_true = (scale * beta_star).mean()

    elif family == 'poisson':
        # Y ~ Poisson(exp(0.3·α(X) + 0.3·β(X)·T))
        scale = 0.3
        lam = np.exp(scale * alpha_star + scale * beta_star * T)
        # Clip lambda to avoid numerical issues
        lam = np.clip(lam, 1e-10, 1e6)
        Y = np.random.poisson(lam).astype(float)
        mu_true = (scale * beta_star).mean()

    else:
        raise ValueError(f"Unknown family: {family}")

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

def run_single_sim(sim_id: int, family: str, config: dict) -> dict:
    """
    Run one simulation: generate data, run FLM inference, compute metrics.

    Returns dict with all metrics for this simulation.
    """
    # Generate fresh data
    data = generate_data(family, config['N'], config['d'])

    # Run structural deep learning with IF correction
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = structural_dml(
            Y=data['Y'],
            T=data['T'],
            X=data['X'],
            family=family,
            n_folds=config['n_folds'],
            hidden_dims=config['hidden_dims'],
            epochs=config['epochs'],
            lr=config['lr'],
            verbose=False,
        )

    # Extract theta estimates
    alpha_hat = result.theta_hat[:, 0]
    beta_hat = result.theta_hat[:, 1]

    # Naive inference: SE = std(beta_hat) / sqrt(N)
    naive_se = beta_hat.std() / np.sqrt(len(beta_hat))
    naive_ci_lower = result.mu_naive - 1.96 * naive_se
    naive_ci_upper = result.mu_naive + 1.96 * naive_se
    naive_covered = naive_ci_lower <= data['mu_true'] <= naive_ci_upper

    # IF inference (from structural_dml)
    if_covered = result.ci_lower <= data['mu_true'] <= result.ci_upper

    # Extract training diagnostics
    histories = result.diagnostics.get('histories', [])
    if histories:
        train_losses = [h.train_losses[-1] for h in histories if h.train_losses]
        val_losses = [h.val_losses[-1] for h in histories if h.val_losses]
        best_epochs = [h.best_epoch for h in histories if hasattr(h, 'best_epoch')]
        mean_train_loss = np.mean(train_losses) if train_losses else 0
        mean_val_loss = np.mean(val_losses) if val_losses else 0
        mean_best_epoch = np.mean(best_epochs) if best_epochs else 0
    else:
        mean_train_loss = mean_val_loss = mean_best_epoch = 0

    # Compute correlations (handle edge cases)
    def safe_corr(a, b):
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return 0.0
        return np.corrcoef(a, b)[0, 1]

    return {
        'sim_id': sim_id,
        'family': family,
        'mu_true': data['mu_true'],

        # Naive results
        'naive_mu_hat': result.mu_naive,
        'naive_se': naive_se,
        'naive_ci_lower': naive_ci_lower,
        'naive_ci_upper': naive_ci_upper,
        'naive_covered': naive_covered,

        # IF results
        'if_mu_hat': result.mu_hat,
        'if_se': result.se,
        'if_ci_lower': result.ci_lower,
        'if_ci_upper': result.ci_upper,
        'if_covered': if_covered,

        # Parameter recovery
        'rmse_alpha': np.sqrt(((alpha_hat - data['alpha_star']) ** 2).mean()),
        'rmse_beta': np.sqrt(((beta_hat - data['beta_star']) ** 2).mean()),
        'corr_alpha': safe_corr(alpha_hat, data['alpha_star']),
        'corr_beta': safe_corr(beta_hat, data['beta_star']),

        # Training quality
        'train_loss': mean_train_loss,
        'val_loss': mean_val_loss,
        'train_val_gap': mean_val_loss - mean_train_loss,
        'best_epoch': mean_best_epoch,

        # IF diagnostics
        'correction_ratio': result.diagnostics.get('correction_ratio', 0),
        'min_eigenvalue': result.diagnostics.get('min_lambda_eigenvalue', 0),
        'pct_regularized': result.diagnostics.get('pct_regularized', 0),
    }

# =============================================================================
# STUDY RUNNER
# =============================================================================

def run_study(family: str, config: dict, output_dir: str) -> pd.DataFrame:
    """
    Run M simulations for a single family with tqdm progress.

    Args:
        family: 'linear', 'logit', or 'poisson'
        config: Configuration dict
        output_dir: Directory for incremental saves

    Returns:
        DataFrame with all simulation results
    """
    results = []
    progress_file = os.path.join(output_dir, f'{family}_progress.csv')

    with tqdm(range(config['M']), desc=f'{family.upper():>8}', ncols=80) as pbar:
        for sim_id in pbar:
            result = run_single_sim(sim_id, family, config)
            results.append(result)

            # Update progress bar with running metrics
            df = pd.DataFrame(results)
            if_cov = df['if_covered'].mean()
            naive_cov = df['naive_covered'].mean()
            corr_beta = df['corr_beta'].mean()

            pbar.set_postfix({
                'IF': f'{if_cov:.0%}',
                'Nv': f'{naive_cov:.0%}',
                'Corr': f'{corr_beta:.2f}'
            })

            # Incremental save
            df.to_csv(progress_file, index=False)

    return pd.DataFrame(results)

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def print_summary(results: pd.DataFrame, family: str):
    """Print 4-table summary comparing naive vs IF."""

    # Table 1: Parameter Recovery
    print(f"\n{'='*60}")
    print(f"PHASE 1: PARAMETER RECOVERY ({family.upper()})")
    print(f"{'='*60}")
    print(f"  RMSE(α):  {results['rmse_alpha'].mean():.4f} ± {results['rmse_alpha'].std():.4f}")
    print(f"  RMSE(β):  {results['rmse_beta'].mean():.4f} ± {results['rmse_beta'].std():.4f}")
    print(f"  Corr(α):  {results['corr_alpha'].mean():.3f} ± {results['corr_alpha'].std():.3f}")
    print(f"  Corr(β):  {results['corr_beta'].mean():.3f} ± {results['corr_beta'].std():.3f}")

    # Table 2: Inference Comparison
    print(f"\n{'='*60}")
    print(f"PHASE 2: INFERENCE COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Naive':>15} {'IF':>15}")
    print(f"{'-'*50}")

    # Point estimate RMSE
    naive_rmse = np.sqrt(((results['naive_mu_hat'] - results['mu_true'])**2).mean())
    if_rmse = np.sqrt(((results['if_mu_hat'] - results['mu_true'])**2).mean())
    print(f"{'RMSE(μ̂)':<20} {naive_rmse:>15.6f} {if_rmse:>15.6f}")

    # Bias
    naive_bias = (results['naive_mu_hat'] - results['mu_true']).mean()
    if_bias = (results['if_mu_hat'] - results['mu_true']).mean()
    print(f"{'Bias':<20} {naive_bias:>15.6f} {if_bias:>15.6f}")

    # SE comparison
    naive_se_emp = results['naive_mu_hat'].std()
    if_se_emp = results['if_mu_hat'].std()
    naive_se_est = results['naive_se'].mean()
    if_se_est = results['if_se'].mean()

    print(f"{'SE(empirical)':<20} {naive_se_emp:>15.6f} {if_se_emp:>15.6f}")
    print(f"{'SE(estimated)':<20} {naive_se_est:>15.6f} {if_se_est:>15.6f}")

    naive_ratio = naive_se_est / naive_se_emp if naive_se_emp > 0 else float('inf')
    if_ratio = if_se_est / if_se_emp if if_se_emp > 0 else float('inf')
    print(f"{'SE Ratio':<20} {naive_ratio:>15.2f} {if_ratio:>15.2f}")

    # Coverage (highlight with color codes)
    naive_cov = results['naive_covered'].mean()
    if_cov = results['if_covered'].mean()
    print(f"{'Coverage':<20} {naive_cov:>14.0%} {if_cov:>14.0%}")

    # Table 3: Training Quality
    print(f"\n{'='*60}")
    print(f"PHASE 3: TRAINING QUALITY")
    print(f"{'='*60}")
    print(f"  Train Loss:    {results['train_loss'].mean():.4f} ± {results['train_loss'].std():.4f}")
    print(f"  Val Loss:      {results['val_loss'].mean():.4f} ± {results['val_loss'].std():.4f}")
    print(f"  Gap:           {results['train_val_gap'].mean():.4f} ± {results['train_val_gap'].std():.4f}")
    print(f"  Best Epoch:    {results['best_epoch'].mean():.0f} ± {results['best_epoch'].std():.0f}")

    # Table 4: IF Diagnostics
    print(f"\n{'='*60}")
    print(f"PHASE 4: IF DIAGNOSTICS")
    print(f"{'='*60}")
    print(f"  Correction Ratio:   {results['correction_ratio'].mean():.3f} ± {results['correction_ratio'].std():.3f}")
    print(f"  Min Eigenvalue:     {results['min_eigenvalue'].mean():.6f}")
    print(f"  Pct Regularized:    {results['pct_regularized'].mean():.1f}%")

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_coverage_comparison(all_results: pd.DataFrame, output_dir: str):
    """Bar chart comparing naive vs IF coverage across families."""
    fig, ax = plt.subplots(figsize=(10, 6))

    families = all_results['family'].unique()
    x = np.arange(len(families))
    width = 0.35

    naive_cov = [all_results[all_results['family'] == f]['naive_covered'].mean() for f in families]
    if_cov = [all_results[all_results['family'] == f]['if_covered'].mean() for f in families]

    bars1 = ax.bar(x - width/2, naive_cov, width, label='Naive', color='#d62728', alpha=0.8)
    bars2 = ax.bar(x + width/2, if_cov, width, label='IF', color='#2ca02c', alpha=0.8)

    ax.axhline(y=0.95, color='black', linestyle='--', linewidth=2, label='Target (95%)')
    ax.axhline(y=0.93, color='gray', linestyle=':', linewidth=1)
    ax.axhline(y=0.97, color='gray', linestyle=':', linewidth=1)

    ax.set_ylabel('Coverage', fontsize=12)
    ax.set_xlabel('Family', fontsize=12)
    ax.set_title('Coverage Comparison: Naive vs IF Correction', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f.upper() for f in families], fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.05)

    # Add value labels on bars
    for bar, val in zip(bars1, naive_cov):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, if_cov):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coverage_comparison.png'), dpi=150)
    plt.close()

def plot_se_ratio_comparison(all_results: pd.DataFrame, output_dir: str):
    """Bar chart comparing SE ratios across families."""
    fig, ax = plt.subplots(figsize=(10, 6))

    families = all_results['family'].unique()
    x = np.arange(len(families))
    width = 0.35

    naive_ratios = []
    if_ratios = []

    for f in families:
        fdata = all_results[all_results['family'] == f]
        naive_se_emp = fdata['naive_mu_hat'].std()
        if_se_emp = fdata['if_mu_hat'].std()
        naive_se_est = fdata['naive_se'].mean()
        if_se_est = fdata['if_se'].mean()

        naive_ratios.append(naive_se_est / naive_se_emp if naive_se_emp > 0 else 0)
        if_ratios.append(if_se_est / if_se_emp if if_se_emp > 0 else 0)

    bars1 = ax.bar(x - width/2, naive_ratios, width, label='Naive', color='#d62728', alpha=0.8)
    bars2 = ax.bar(x + width/2, if_ratios, width, label='IF', color='#2ca02c', alpha=0.8)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Target (1.0)')
    ax.axhline(y=0.9, color='gray', linestyle=':', linewidth=1)
    ax.axhline(y=1.1, color='gray', linestyle=':', linewidth=1)

    ax.set_ylabel('SE Ratio (estimated / empirical)', fontsize=12)
    ax.set_xlabel('Family', fontsize=12)
    ax.set_title('SE Calibration: Naive vs IF', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f.upper() for f in families], fontsize=11)
    ax.legend(loc='upper right', fontsize=10)

    # Add value labels
    for bar, val in zip(bars1, naive_ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, if_ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'se_ratio_comparison.png'), dpi=150)
    plt.close()

def plot_parameter_recovery(all_results: pd.DataFrame, output_dir: str):
    """Plot parameter recovery (correlation) across families."""
    fig, ax = plt.subplots(figsize=(10, 6))

    families = all_results['family'].unique()
    x = np.arange(len(families))
    width = 0.35

    corr_alpha = [all_results[all_results['family'] == f]['corr_alpha'].mean() for f in families]
    corr_beta = [all_results[all_results['family'] == f]['corr_beta'].mean() for f in families]

    bars1 = ax.bar(x - width/2, corr_alpha, width, label='Corr(α)', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, corr_beta, width, label='Corr(β)', color='#ff7f0e', alpha=0.8)

    ax.set_ylabel('Correlation with True Function', fontsize=12)
    ax.set_xlabel('Family', fontsize=12)
    ax.set_title('Parameter Recovery: Correlation with α*(X) and β*(X)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f.upper() for f in families], fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.05)

    # Add value labels
    for bar, val in zip(bars1, corr_alpha):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, corr_beta):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_recovery.png'), dpi=150)
    plt.close()

# =============================================================================
# JSON REPORT
# =============================================================================

def save_json_report(all_results: pd.DataFrame, config: dict, timestamp: str, output_dir: str):
    """Save comprehensive JSON report with all metrics."""

    report = {
        'meta': {
            'generated': datetime.now().isoformat(),
            'framework': 'FLM (Farrell-Liang-Misra)',
            'version': '1.0',
        },
        'config': config,
        'results_by_family': {},
    }

    for family in config['families']:
        fdata = all_results[all_results['family'] == family]

        # Compute aggregate metrics
        naive_se_emp = fdata['naive_mu_hat'].std()
        if_se_emp = fdata['if_mu_hat'].std()
        naive_se_est = fdata['naive_se'].mean()
        if_se_est = fdata['if_se'].mean()

        report['results_by_family'][family] = {
            'parameter_recovery': {
                'rmse_alpha': float(fdata['rmse_alpha'].mean()),
                'rmse_beta': float(fdata['rmse_beta'].mean()),
                'corr_alpha': float(fdata['corr_alpha'].mean()),
                'corr_beta': float(fdata['corr_beta'].mean()),
            },
            'naive_inference': {
                'mu_hat_mean': float(fdata['naive_mu_hat'].mean()),
                'mu_hat_std': float(fdata['naive_mu_hat'].std()),
                'se_estimated': float(naive_se_est),
                'se_empirical': float(naive_se_emp),
                'se_ratio': float(naive_se_est / naive_se_emp) if naive_se_emp > 0 else None,
                'coverage': float(fdata['naive_covered'].mean()),
            },
            'if_inference': {
                'mu_hat_mean': float(fdata['if_mu_hat'].mean()),
                'mu_hat_std': float(fdata['if_mu_hat'].std()),
                'se_estimated': float(if_se_est),
                'se_empirical': float(if_se_emp),
                'se_ratio': float(if_se_est / if_se_emp) if if_se_emp > 0 else None,
                'coverage': float(fdata['if_covered'].mean()),
            },
            'training_quality': {
                'train_loss': float(fdata['train_loss'].mean()),
                'val_loss': float(fdata['val_loss'].mean()),
                'train_val_gap': float(fdata['train_val_gap'].mean()),
                'best_epoch': float(fdata['best_epoch'].mean()),
            },
            'diagnostics': {
                'correction_ratio': float(fdata['correction_ratio'].mean()),
                'min_eigenvalue': float(fdata['min_eigenvalue'].mean()),
                'pct_regularized': float(fdata['pct_regularized'].mean()),
            },
            'mu_true': float(fdata['mu_true'].iloc[0]),
        }

    json_path = os.path.join(output_dir, f'validation_study_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    return json_path

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the complete validation study."""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create output directories
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    plots_dir = results_dir / f'validation_study_{timestamp}'
    plots_dir.mkdir(exist_ok=True)

    all_results = []

    # Header
    print("=" * 60)
    print("COMPREHENSIVE VALIDATION STUDY (FLM Framework)")
    print("=" * 60)
    print(f"N = {CONFIG['N']:,} samples")
    print(f"M = {CONFIG['M']} simulations per family")
    print(f"K = {CONFIG['n_folds']} cross-fitting folds")
    print(f"Architecture: {CONFIG['hidden_dims']}")
    print(f"Epochs: {CONFIG['epochs']}, lr: {CONFIG['lr']}")
    print(f"Families: {', '.join(CONFIG['families'])}")
    print("=" * 60)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {plots_dir}")
    print()

    # Run each family sequentially
    for family in CONFIG['families']:
        print(f"\n>>> Running {family.upper()} <<<")
        print("-" * 40)

        results = run_study(family, CONFIG, str(plots_dir))
        all_results.append(results)

        print_summary(results, family)

    # Combine all results
    final_df = pd.concat(all_results, ignore_index=True)

    # Save CSV
    csv_path = results_dir / f'validation_study_{timestamp}.csv'
    final_df.to_csv(csv_path, index=False)

    # Save JSON report
    json_path = save_json_report(final_df, CONFIG, timestamp, str(results_dir))

    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    plot_coverage_comparison(final_df, str(plots_dir))
    print("  - coverage_comparison.png")

    plot_se_ratio_comparison(final_df, str(plots_dir))
    print("  - se_ratio_comparison.png")

    plot_parameter_recovery(final_df, str(plots_dir))
    print("  - parameter_recovery.png")

    # Final summary
    print("\n" + "=" * 60)
    print("STUDY COMPLETE")
    print("=" * 60)
    print(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to:")
    print(f"  CSV:   {csv_path}")
    print(f"  JSON:  {json_path}")
    print(f"  Plots: {plots_dir}/")
    print("=" * 60)

    # Quick summary table
    print("\n" + "=" * 60)
    print("COVERAGE SUMMARY")
    print("=" * 60)
    print(f"{'Family':<12} {'Naive':<12} {'IF':<12} {'Target':<12}")
    print("-" * 48)
    for family in CONFIG['families']:
        fdata = final_df[final_df['family'] == family]
        naive = fdata['naive_covered'].mean()
        if_cov = fdata['if_covered'].mean()
        status = "PASS" if 0.93 <= if_cov <= 0.97 else "FAIL"
        print(f"{family.upper():<12} {naive:>10.0%} {if_cov:>10.0%} {'95%':<12} {status}")
    print("=" * 60)


if __name__ == '__main__':
    main()
