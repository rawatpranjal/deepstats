#!/usr/bin/env python3
"""KDE plots for Tobit stress test - both latent and observed targets.

Creates a 2-row × 3-column figure:
- Row 1: Latent target E[β] - KDE, Coverage, RMSE
- Row 2: Observed target E[β·Φ(z)] - KDE, Coverage, RMSE

Reads from single mc_results.csv with 'target' column.

Usage:
    python logs/plot_tobit.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

os.chdir('/Users/pranjal/deepest')

# Path to combined results
RESULTS_DIR = 'logs/tobit_both_test'
CSV_PATH = f'{RESULTS_DIR}/mc_results.csv'


def compute_metrics_for_target(df, target_name):
    """Compute metrics for a specific target from combined DataFrame."""
    target_df = df[df['target'] == target_name]
    if len(target_df) == 0:
        return None

    metrics = []
    for method in target_df['method'].unique():
        method_df = target_df[target_df['method'] == method]
        mu_true = method_df['mu_true'].iloc[0]

        metrics.append({
            'method': method,
            'mu_true': mu_true,
            'bias': method_df['bias'].mean(),
            'variance': method_df['mu_hat'].var(),
            'rmse_mu': np.sqrt(method_df['bias'].mean()**2 + method_df['mu_hat'].var()),
            'empirical_se': method_df['mu_hat'].std(),
            'se_mean': method_df['se'].mean(),
            'se_ratio': method_df['se'].mean() / method_df['mu_hat'].std() if method_df['mu_hat'].std() > 0 else np.nan,
            'coverage': method_df['covered'].mean(),
            'n_sims': len(method_df),
        })

    return pd.DataFrame(metrics)


def plot_target_row(axes, df, target_name, target_label, row_idx):
    """Plot one row for a target (KDE, Coverage, RMSE)."""
    target_df = df[df['target'] == target_name]
    if len(target_df) == 0:
        print(f"No data for target={target_name}")
        for ax in axes:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
        return False

    mu_true = target_df['mu_true'].iloc[0]
    metrics = compute_metrics_for_target(df, target_name)

    # Panel A: KDE comparison
    ax1 = axes[0]
    for method, color, style in [('naive', 'red', '--'), ('influence', 'blue', '-')]:
        subset = target_df[target_df['method'] == method]
        if len(subset) > 0:
            kde = stats.gaussian_kde(subset['mu_hat'].dropna())
            x = np.linspace(subset['mu_hat'].min()-0.2, subset['mu_hat'].max()+0.2, 200)
            ax1.plot(x, kde(x), color=color, linestyle=style, lw=2, label=method.capitalize())
            ax1.fill_between(x, kde(x), alpha=0.15, color=color)
    ax1.axvline(mu_true, color='black', linestyle=':', lw=2, label=f'True: {mu_true:.3f}')
    ax1.set_xlabel('Estimate')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{chr(65 + row_idx*3)}. {target_label} Distribution')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Panel B: Coverage comparison
    ax2 = axes[1]
    methods = metrics['method'].tolist()
    coverage = metrics['coverage'].tolist()

    colors = ['red' if m == 'naive' else 'blue' for m in methods]
    bars = ax2.bar(methods, coverage, color=colors, alpha=0.7)
    ax2.axhline(0.95, color='green', linestyle='--', lw=2, label='95% target')
    ax2.set_ylabel('Coverage')
    ax2.set_title(f'{chr(66 + row_idx*3)}. Coverage')
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1.1)

    # Add value labels
    for bar, cov in zip(bars, coverage):
        ax2.annotate(f'{cov:.1%}', xy=(bar.get_x() + bar.get_width()/2, cov + 0.02),
                    ha='center', fontsize=9)

    # Panel C: RMSE comparison
    ax3 = axes[2]
    rmse = metrics['rmse_mu'].tolist()
    colors = ['red' if m == 'naive' else 'blue' for m in methods]
    bars = ax3.bar(methods, rmse, color=colors, alpha=0.7)
    ax3.set_ylabel('RMSE')
    ax3.set_title(f'{chr(67 + row_idx*3)}. RMSE (lower is better)')

    # Add value labels and improvement
    for bar, r in zip(bars, rmse):
        ax3.annotate(f'{r:.4f}', xy=(bar.get_x() + bar.get_width()/2, r + max(rmse)*0.02),
                    ha='center', fontsize=9)

    if len(rmse) >= 2:
        naive_row = metrics[metrics['method'] == 'naive']
        inf_row = metrics[metrics['method'] == 'influence']
        if len(naive_row) > 0 and len(inf_row) > 0:
            naive_rmse = naive_row['rmse_mu'].iloc[0]
            inf_rmse = inf_row['rmse_mu'].iloc[0]
            if inf_rmse > 0:
                improvement = naive_rmse / inf_rmse
                ax3.annotate(f'IF is {improvement:.1f}x better',
                            xy=(0.5, max(rmse)*0.5), fontsize=10, ha='center',
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    return True


def plot_tobit_both():
    """Create 2-row figure for latent and observed targets."""
    if not os.path.exists(CSV_PATH):
        print(f"Waiting for {CSV_PATH}...")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    print(f"Targets: {df['target'].unique()}")
    print(f"Methods: {df['method'].unique()}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Row 1: Latent target E[β]
    latent_ok = plot_target_row(
        axes[0], df, 'latent',
        r'Latent $\mathbb{E}[\beta]$',
        row_idx=0
    )

    # Row 2: Observed target E[β·Φ(z)]
    observed_ok = plot_target_row(
        axes[1], df, 'observed',
        r'Observed $\mathbb{E}[\beta \cdot \Phi(z)]$',
        row_idx=1
    )

    plt.suptitle('Tobit Stress Test: Two Estimands (M=50, N=10000, K=50, Deep Network)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('logs/tobit_both_targets.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: logs/tobit_both_targets.png")
    plt.close()

    # Print summary table
    print("\n" + "="*70)
    print("TOBIT STRESS TEST RESULTS - BOTH TARGETS")
    print("="*70)

    for target_name, target_label in [('latent', 'Latent E[β]'), ('observed', 'Observed E[β·Φ(z)]')]:
        print(f"\n--- {target_label} ---")
        metrics = compute_metrics_for_target(df, target_name)
        if metrics is not None and len(metrics) > 0:
            print(f"{'Method':<12} {'RMSE':>10} {'Coverage':>10} {'SE Ratio':>10}")
            print("-" * 45)
            for _, row in metrics.iterrows():
                print(f"{row['method']:<12} {row['rmse_mu']:>10.4f} {row['coverage']:>9.1%} {row['se_ratio']:>10.2f}")
        else:
            print("  Results not yet available")

    print("\n" + "="*70)


if __name__ == '__main__':
    plot_tobit_both()
