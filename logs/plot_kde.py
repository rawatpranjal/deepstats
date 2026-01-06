#!/usr/bin/env python3
"""Generate KDE comparison plots with full hyperparameter legends below."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Ensure we're in the right directory
os.chdir('/Users/pranjal/deepest')

# Experiment configurations with full hyperparams
EXPERIMENTS = {
    'exp_K50': {
        'label': 'K=50',
        'full': 'Net=[64,32], K=50, E=50',
        'color': 'blue',
        'csv': 'logs/exp_K50/mc_results.csv'
    },
    'exp_deep': {
        'label': 'Deep',
        'full': 'Net=[128,64,32], K=20, E=50',
        'color': 'green',
        'csv': 'logs/exp_deep/mc_results.csv'
    },
    'exp_e100': {
        'label': 'E=100',
        'full': 'Net=[64,32], K=20, E=100',
        'color': 'orange',
        'csv': 'logs/exp_e100/mc_results.csv'
    },
    'exp_deep_K50': {
        'label': 'Deep+K50',
        'full': 'Net=[128,64,32], K=50, E=50',
        'color': 'purple',
        'csv': 'logs/exp_deep_K50/mc_results.csv'
    },
}

def load_experiment(exp_key):
    """Load experiment results."""
    exp = EXPERIMENTS[exp_key]
    if not os.path.exists(exp['csv']):
        return None
    df = pd.read_csv(exp['csv'])
    return df

def plot_kde_comparison():
    """Create KDE comparison plot with legends below."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: KDE overlay
    ax1 = axes[0]

    # Get mu_true from first available experiment
    mu_true = None
    for exp_key in EXPERIMENTS:
        df = load_experiment(exp_key)
        if df is not None:
            mu_true = df['mu_true'].iloc[0]
            break

    if mu_true is None:
        print("No experiment data found!")
        return

    legend_handles = []
    legend_labels = []

    # First plot Naive (from any experiment - they're all the same)
    for exp_key in EXPERIMENTS:
        df = load_experiment(exp_key)
        if df is not None:
            naive_df = df[df['method'] == 'naive']
            if len(naive_df) > 0:
                mu_naive = naive_df['mu_hat'].values
                kde_naive = stats.gaussian_kde(mu_naive)
                x = np.linspace(mu_naive.min() - 0.1, mu_naive.max() + 0.1, 200)
                line, = ax1.plot(x, kde_naive(x), 'r--', lw=2, alpha=0.8)
                legend_handles.append(line)
                legend_labels.append('Naive (all configs)')
                break

    # Plot Influence for each experiment
    for exp_key, exp in EXPERIMENTS.items():
        df = load_experiment(exp_key)
        if df is None:
            continue

        inf_df = df[df['method'] == 'influence']
        if len(inf_df) == 0:
            continue

        mu_inf = inf_df['mu_hat'].values
        kde_inf = stats.gaussian_kde(mu_inf)

        # Use wider x range for smooth plot
        x_min = min(mu_inf.min(), mu_true - 0.5)
        x_max = max(mu_inf.max(), mu_true + 0.5)
        x = np.linspace(x_min, x_max, 200)

        line, = ax1.plot(x, kde_inf(x), color=exp['color'], lw=2, label=exp['label'])
        legend_handles.append(line)
        legend_labels.append(exp['full'])

    # True value line
    ax1.axvline(mu_true, color='black', linestyle=':', lw=2)
    ax1.annotate(f'True: {mu_true:.3f}', xy=(mu_true, ax1.get_ylim()[1]*0.9),
                 fontsize=10, ha='center')

    ax1.set_xlabel(r'$\hat{\mu}$ (ATE Estimate)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('A. Distribution of ATE Estimates', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Panel B: Parameter Recovery (R^2)
    ax2 = axes[1]

    exp_names = []
    r2_alpha = []
    r2_beta = []
    colors = []

    for exp_key, exp in EXPERIMENTS.items():
        df = load_experiment(exp_key)
        if df is None:
            continue

        inf_df = df[df['method'] == 'influence']
        if len(inf_df) == 0:
            continue

        # R^2 from correlation
        corr_a = inf_df['corr_alpha'].mean()
        corr_b = inf_df['corr_beta'].mean()

        exp_names.append(exp['label'])
        r2_alpha.append(corr_a ** 2)
        r2_beta.append(corr_b ** 2)
        colors.append(exp['color'])

    if exp_names:
        x_pos = np.arange(len(exp_names))
        width = 0.35

        bars1 = ax2.bar(x_pos - width/2, r2_alpha, width, label=r'$R^2(\alpha)$',
                        color=[c for c in colors], alpha=0.7)
        bars2 = ax2.bar(x_pos + width/2, r2_beta, width, label=r'$R^2(\beta)$',
                        color=[c for c in colors], alpha=0.4, hatch='//')

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(exp_names)
        ax2.set_ylabel(r'$R^2$', fontsize=12)
        ax2.set_title('B. Parameter Recovery', fontsize=12, fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')

    # Add legend below with full hyperparameters
    fig.legend(legend_handles, legend_labels,
               loc='lower center', ncol=min(len(legend_labels), 5),
               bbox_to_anchor=(0.5, -0.02), fontsize=10,
               frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)  # Make room for legend

    plt.savefig('logs/kde_money_slide.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: logs/kde_money_slide.png")
    plt.close()

if __name__ == '__main__':
    plot_kde_comparison()
