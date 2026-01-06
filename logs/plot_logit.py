#!/usr/bin/env python3
"""KDE plots for Logit stress test - β target."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

os.chdir('/Users/pranjal/deepest')

def plot_logit_results():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Load results
    csv_path = 'logs/logit_stress_test/mc_results.csv'
    if not os.path.exists(csv_path):
        print(f"Waiting for {csv_path}...")
        return

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
    ax1.set_xlabel(r'$\hat{\beta}$ (Log-odds ratio)')
    ax1.set_ylabel('Density')
    ax1.set_title(r'A. Distribution of $\beta$ Estimates')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Panel B: Coverage comparison
    ax2 = axes[1]
    metrics_path = 'logs/logit_stress_test/mc_results.metrics.csv'
    if not os.path.exists(metrics_path):
        print(f"Waiting for {metrics_path}...")
        return

    metrics = pd.read_csv(metrics_path)
    methods = metrics['method'].tolist()
    coverage = metrics['coverage'].tolist()

    colors = ['red' if m == 'naive' else 'blue' for m in methods]
    bars = ax2.bar(methods, coverage, color=colors, alpha=0.7)
    ax2.axhline(0.95, color='green', linestyle='--', lw=2, label='95% target')
    ax2.set_ylabel('Coverage')
    ax2.set_title('B. Coverage Comparison')
    ax2.legend()
    ax2.set_ylim(0, 1.1)

    # Add value labels
    for bar, cov in zip(bars, coverage):
        ax2.annotate(f'{cov:.1%}', xy=(bar.get_x() + bar.get_width()/2, cov + 0.02),
                    ha='center', fontsize=10)

    # Panel C: RMSE comparison
    ax3 = axes[2]
    rmse = metrics['rmse_mu'].tolist()
    colors = ['red' if m == 'naive' else 'blue' for m in methods]
    bars = ax3.bar(methods, rmse, color=colors, alpha=0.7)
    ax3.set_ylabel('RMSE')
    ax3.set_title('C. RMSE (lower is better)')

    # Add value labels and improvement
    for bar, r in zip(bars, rmse):
        ax3.annotate(f'{r:.4f}', xy=(bar.get_x() + bar.get_width()/2, r + 0.002),
                    ha='center', fontsize=10)

    if len(rmse) >= 2:
        naive_rmse = metrics[metrics['method'] == 'naive']['rmse_mu'].iloc[0]
        inf_rmse = metrics[metrics['method'] == 'influence']['rmse_mu'].iloc[0]
        if inf_rmse > 0:
            improvement = naive_rmse / inf_rmse
            ax3.annotate(f'IF is {improvement:.1f}x better',
                        xy=(0.5, max(rmse)*0.5), fontsize=11, ha='center',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.suptitle(r'Logit Stress Test: $\beta$ Target (Log-odds Ratio)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('logs/logit_stress_test/logit_results.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: logs/logit_stress_test/logit_results.png")
    plt.close()

    # Print summary
    print("\n=== LOGIT STRESS TEST RESULTS ===")
    print(f"{'Method':<12} {'RMSE':>10} {'Coverage':>10} {'SE Ratio':>10}")
    print("-" * 45)
    for _, row in metrics.iterrows():
        print(f"{row['method']:<12} {row['rmse_mu']:>10.4f} {row['coverage']:>9.1%} {row['se_ratio']:>10.2f}")
    print("-" * 45)

if __name__ == '__main__':
    plot_logit_results()
