"""Metrics computation for Monte Carlo results."""

import numpy as np
import pandas as pd


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary metrics by model and method.

    Metrics:
    - μ* = True parameter value
    - Bias = E[μ̂] - μ*
    - Var = Var(μ̂)
    - RMSE = √(Bias² + Var)
    - SE(emp) = √Var (true SE from MC)
    - SE(est) = mean(estimated SE)
    - Ratio = SE(est)/SE(emp) (calibration, target=1.0)
    - CI Width = 2×1.96×SE(est)
    - Coverage = P(μ* ∈ CI) (target=95%)
    """
    df_clean = df.dropna(subset=["mu_hat", "se", "bias"])

    summary = df_clean.groupby(["model", "method"]).agg({
        "mu_true": "first",
        "bias": ["mean", "std"],
        "mu_hat": "std",
        "se": "mean",
        "covered": "mean",
        "sim_id": "count",
    }).reset_index()

    summary.columns = [
        "model", "method", "mu_true",
        "bias_mean", "bias_std",
        "empirical_se", "se_mean", "coverage",
        "n_sims",
    ]

    # RMSE
    def rmse_func(group):
        return np.sqrt(np.mean(group["bias"]**2))

    rmse_by_group = df_clean.groupby(["model", "method"]).apply(
        rmse_func
    ).reset_index(name="rmse")

    summary = summary.merge(rmse_by_group, on=["model", "method"])

    # Derived metrics
    summary["se_ratio"] = summary["se_mean"] / summary["empirical_se"]
    summary["variance"] = summary["empirical_se"] ** 2
    summary["ci_width"] = 2 * 1.96 * summary["se_mean"]

    return summary[[
        "model", "method", "mu_true",
        "bias_mean", "variance", "rmse",
        "empirical_se", "se_mean", "se_ratio",
        "ci_width", "coverage", "n_sims",
    ]]


def print_table(metrics_df: pd.DataFrame) -> str:
    """Print formatted summary table."""
    lines = []
    lines.append("=" * 130)
    lines.append("MONTE CARLO RESULTS")
    lines.append("=" * 130)
    lines.append("")

    header = (
        f"{'Model':<10} {'Method':<12} {'μ*':>8} "
        f"{'Bias':>8} {'Var':>8} {'RMSE':>8} "
        f"{'SE(emp)':>8} {'SE(est)':>8} {'Ratio':>6} "
        f"{'CI Width':>9} {'Coverage':>9}"
    )
    lines.append(header)
    lines.append("-" * 130)

    for _, row in metrics_df.iterrows():
        line = (
            f"{row['model']:<10} "
            f"{row['method']:<12} "
            f"{row['mu_true']:>8.4f} "
            f"{row['bias_mean']:>8.4f} "
            f"{row['variance']:>8.4f} "
            f"{row['rmse']:>8.4f} "
            f"{row['empirical_se']:>8.4f} "
            f"{row['se_mean']:>8.4f} "
            f"{row['se_ratio']:>6.2f} "
            f"{row['ci_width']:>9.4f} "
            f"{row['coverage']:>9.2%}"
        )
        lines.append(line)

    lines.append("-" * 130)
    lines.append("")
    lines.append("Ratio = SE(est)/SE(emp), target=1.0 | Coverage target=95%")
    lines.append("Naive underestimates SE -> narrow CI -> poor coverage!")
    lines.append("=" * 130)

    output = "\n".join(lines)
    print(output)
    return output
