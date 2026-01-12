#!/usr/bin/env python
"""Generate plots for MC simulation results."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load results
df = pd.read_csv("mc_k50.csv")

# Colors for methods
COLORS = {
    "naive": "#E74C3C",      # Red
    "influence": "#27AE60",  # Green
    "bootstrap": "#3498DB",  # Blue
}

METHOD_LABELS = {
    "naive": "Naive",
    "influence": "Influence Function",
    "bootstrap": "Bootstrap",
}

# Get true mu
mu_true = df["mu_true"].mean()

# =============================================================================
# Plot 1: Histogram of μ̂ by method
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

for idx, method in enumerate(["naive", "influence", "bootstrap"]):
    ax = axes[idx]
    data = df[df["method"] == method]["mu_hat"]

    ax.hist(data, bins=15, color=COLORS[method], alpha=0.7, edgecolor="black")
    ax.axvline(mu_true, color="black", linestyle="--", linewidth=2, label=f"μ*={mu_true:.4f}")
    ax.axvline(data.mean(), color=COLORS[method], linestyle="-", linewidth=2, label=f"μ̂={data.mean():.4f}")

    ax.set_xlabel("μ̂", fontsize=12)
    ax.set_title(METHOD_LABELS[method], fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    # Add stats
    bias = data.mean() - mu_true
    se = data.std()
    ax.text(0.05, 0.95, f"Bias: {bias:.4f}\nSD: {se:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

axes[0].set_ylabel("Frequency", fontsize=12)
fig.suptitle("Distribution of μ̂ Estimates (M=30, N=1000, K=50 folds)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("mc_all_methods_hist.png", dpi=150, bbox_inches="tight")
print("Saved: mc_all_methods_hist.png")
plt.close()

# =============================================================================
# Plot 2: CI Coverage Visualization
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

for idx, method in enumerate(["naive", "influence", "bootstrap"]):
    ax = axes[idx]
    method_df = df[df["method"] == method].reset_index(drop=True)

    n_sims = len(method_df)
    coverage = method_df["covered"].mean()

    for i, row in method_df.iterrows():
        ci_lo = row["mu_hat"] - 1.96 * row["se"]
        ci_hi = row["mu_hat"] + 1.96 * row["se"]
        color = COLORS[method] if row["covered"] else "#95A5A6"
        alpha = 0.8 if row["covered"] else 0.4
        ax.plot([ci_lo, ci_hi], [i, i], color=color, alpha=alpha, linewidth=1.5)
        ax.plot(row["mu_hat"], i, "o", color=color, markersize=3, alpha=alpha)

    ax.axvline(mu_true, color="black", linestyle="--", linewidth=2)
    ax.set_xlabel("μ", fontsize=12)
    ax.set_title(f"{METHOD_LABELS[method]}\nCoverage: {coverage:.1%}", fontsize=14, fontweight="bold")
    ax.set_xlim(-2, 2)

axes[0].set_ylabel("Simulation", fontsize=12)
fig.suptitle("95% Confidence Interval Coverage (Target: 95%)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("mc_all_methods_ci.png", dpi=150, bbox_inches="tight")
print("Saved: mc_all_methods_ci.png")
plt.close()

# =============================================================================
# Plot 3: SE Comparison
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

methods = ["naive", "influence", "bootstrap"]
x = np.arange(len(methods))
width = 0.35

se_emp = []
se_est = []
for method in methods:
    method_df = df[df["method"] == method]
    se_emp.append(method_df["mu_hat"].std())
    se_est.append(method_df["se"].mean())

bars1 = ax.bar(x - width/2, se_emp, width, label="Empirical SE (true)", color="#34495E", alpha=0.8)
bars2 = ax.bar(x + width/2, se_est, width, label="Estimated SE (mean)", color="#1ABC9C", alpha=0.8)

ax.set_ylabel("Standard Error", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
ax.legend()
ax.set_title("SE Calibration: Estimated vs Empirical", fontsize=14, fontweight="bold")

# Add ratio annotations
for i, (emp, est) in enumerate(zip(se_emp, se_est)):
    ratio = est / emp
    ax.annotate(f"Ratio: {ratio:.2f}", xy=(i, max(emp, est) + 0.02),
                ha="center", fontsize=10, fontweight="bold",
                color="green" if 0.8 < ratio < 1.2 else "red")

plt.tight_layout()
plt.savefig("mc_se_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: mc_se_comparison.png")
plt.close()

# =============================================================================
# Plot 4: Summary Metrics Bar Chart
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Coverage
ax = axes[0]
coverages = [df[df["method"] == m]["covered"].mean() * 100 for m in methods]
bars = ax.bar(range(len(methods)), coverages, color=[COLORS[m] for m in methods], alpha=0.8)
ax.axhline(95, color="black", linestyle="--", linewidth=2, label="Target (95%)")
ax.set_ylabel("Coverage (%)")
ax.set_xticks(range(len(methods)))
ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=15, ha="right")
ax.set_ylim(0, 105)
ax.set_title("Coverage", fontweight="bold")
ax.legend()

# SE Ratio
ax = axes[1]
ratios = [se_est[i] / se_emp[i] for i in range(len(methods))]
bars = ax.bar(range(len(methods)), ratios, color=[COLORS[m] for m in methods], alpha=0.8)
ax.axhline(1.0, color="black", linestyle="--", linewidth=2, label="Target (1.0)")
ax.set_ylabel("SE Ratio (est/emp)")
ax.set_xticks(range(len(methods)))
ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=15, ha="right")
ax.set_ylim(0, 1.5)
ax.set_title("SE Ratio", fontweight="bold")
ax.legend()

# Bias
ax = axes[2]
biases = [abs(df[df["method"] == m]["mu_hat"].mean() - mu_true) for m in methods]
bars = ax.bar(range(len(methods)), biases, color=[COLORS[m] for m in methods], alpha=0.8)
ax.set_ylabel("|Bias|")
ax.set_xticks(range(len(methods)))
ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=15, ha="right")
ax.set_title("Absolute Bias", fontweight="bold")

fig.suptitle("Method Comparison Summary (M=30, K=50)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("mc_summary.png", dpi=150, bbox_inches="tight")
print("Saved: mc_summary.png")
plt.close()

# =============================================================================
# Plot 5: KDE Overlay - All Methods
# =============================================================================
from scipy.stats import gaussian_kde

fig, ax = plt.subplots(figsize=(10, 6))

# Create x range for plotting
x_min = df["mu_hat"].min() - 0.2
x_max = df["mu_hat"].max() + 0.2
x_range = np.linspace(x_min, x_max, 500)

for method in ["naive", "influence", "bootstrap"]:
    data = df[df["method"] == method]["mu_hat"].values
    coverage = df[df["method"] == method]["covered"].mean()

    # Compute KDE
    kde = gaussian_kde(data, bw_method='scott')
    density = kde(x_range)

    # Plot KDE line and fill
    label = f"{METHOD_LABELS[method]} (Cov: {coverage:.1%})"
    ax.plot(x_range, density, color=COLORS[method], linewidth=2.5, label=label)
    ax.fill_between(x_range, density, alpha=0.25, color=COLORS[method])

# True value
ax.axvline(mu_true, color="black", linestyle="--", linewidth=2.5, label=f"μ* = {mu_true:.4f}")

ax.set_xlabel("μ̂ (Estimated ATE)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Distribution of ATE Estimates by Method (KDE Overlay)", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.set_xlim(x_min, x_max)

# Add annotation
ax.text(0.02, 0.98, f"M={len(df)//3} simulations\nN=1000, K=50 folds",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig("mc_kde_overlay.png", dpi=150, bbox_inches="tight")
print("Saved: mc_kde_overlay.png")
plt.close()

print("\nAll plots generated successfully!")

# Print summary table
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Method':<15} {'Coverage':>10} {'SE Ratio':>10} {'|Bias|':>10}")
print("-"*70)
for i, m in enumerate(methods):
    cov = df[df["method"] == m]["covered"].mean() * 100
    ratio = se_est[i] / se_emp[i]
    bias = abs(df[df["method"] == m]["mu_hat"].mean() - mu_true)
    print(f"{METHOD_LABELS[m]:<15} {cov:>9.1f}% {ratio:>10.2f} {bias:>10.4f}")
print("="*70)
