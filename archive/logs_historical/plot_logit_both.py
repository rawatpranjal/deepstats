#!/usr/bin/env python3
"""KDE plots for Logit stress test - both β and AME targets."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

os.chdir('/Users/pranjal/deepest')


def plot_target_row(axes, csv_path, metrics_path, target_name, target_label, row_idx):
    """Plot one row for a target (KDE, Coverage, RMSE)."""
    if not os.path.exists(csv_path):
        print(f"Waiting for {csv_path}...")
        for ax in axes:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
        return False

    df = pd.read_csv(csv_path)
    mu_true = df['mu_true'].iloc[0]

    # Panel A: KDE comparison
    ax1 = axes[0]
    for method, color, style in [('naive', 'red', '--'), ('influence', 'blue', '-')]:
        subset = df[df['method'] == method]
        if len(subset) > 0:
            kde = stats.gaussian_kde(subset['mu_hat'])
            x = np.linspace(subset['mu_hat'].min()-0.1, subset['mu_hat'].max()+0.1, 200)
            ax1.plot(x, kde(x), color=color, linestyle=style, lw=2, label=method.capitalize())
    ax1.axvline(mu_true, color='black', linestyle=':', lw=2, label=f'True: {mu_true:.3f}')
    ax1.set_xlabel(f'Estimate')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{chr(65 + row_idx*3)}. {target_label} Distribution')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Panel B: Coverage comparison
    ax2 = axes[1]
    if not os.path.exists(metrics_path):
        ax2.text(0.5, 0.5, 'Metrics not available', ha='center', va='center', transform=ax2.transAxes)
        return False

    metrics = pd.read_csv(metrics_path)
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
        ax3.annotate(f'{r:.4f}', xy=(bar.get_x() + bar.get_width()/2, r + 0.002),
                    ha='center', fontsize=9)

    if len(rmse) >= 2:
        naive_rmse = metrics[metrics['method'] == 'naive']['rmse_mu'].iloc[0]
        inf_rmse = metrics[metrics['method'] == 'influence']['rmse_mu'].iloc[0]
        if inf_rmse > 0:
            improvement = naive_rmse / inf_rmse
            ax3.annotate(f'IF is {improvement:.1f}x better',
                        xy=(0.5, max(rmse)*0.5), fontsize=10, ha='center',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    return True


def plot_logit_both():
    """Create 2-row figure for β and AME targets."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Row 1: β target (log-odds ratio)
    beta_ok = plot_target_row(
        axes[0],
        'logs/logit_stress_test/mc_results.csv',
        'logs/logit_stress_test/mc_results.metrics.csv',
        'beta',
        r'$\beta$ (Log-odds Ratio)',
        row_idx=0
    )

    # Row 2: AME target (average marginal effect)
    ame_ok = plot_target_row(
        axes[1],
        'logs/logit_ame_test/mc_results.csv',
        'logs/logit_ame_test/mc_results.metrics.csv',
        'ame',
        r'AME $p(1-p)\beta$',
        row_idx=1
    )

    plt.suptitle('Logit Stress Test: Two Targets', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('logs/logit_both_targets.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: logs/logit_both_targets.png")
    plt.close()

    # Print summary table
    print("\n" + "="*70)
    print("LOGIT STRESS TEST RESULTS - BOTH TARGETS")
    print("="*70)

    for target_name, csv_path, metrics_path in [
        ('β (Log-odds)', 'logs/logit_stress_test/mc_results.csv', 'logs/logit_stress_test/mc_results.metrics.csv'),
        ('AME', 'logs/logit_ame_test/mc_results.csv', 'logs/logit_ame_test/mc_results.metrics.csv')
    ]:
        print(f"\n--- {target_name} ---")
        if os.path.exists(metrics_path):
            metrics = pd.read_csv(metrics_path)
            print(f"{'Method':<12} {'RMSE':>10} {'Coverage':>10} {'SE Ratio':>10}")
            print("-" * 45)
            for _, row in metrics.iterrows():
                print(f"{row['method']:<12} {row['rmse_mu']:>10.4f} {row['coverage']:>9.1%} {row['se_ratio']:>10.2f}")
        else:
            print("  Results not yet available")

    print("\n" + "="*70)


if __name__ == '__main__':
    plot_logit_both()
