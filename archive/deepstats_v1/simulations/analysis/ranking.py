"""
Automated ranking analysis for simulation results.

Compares DeepHTE vs CausalForest vs LinearDML across all DGPs and metrics.
Produces clear winner tables showing which method dominates.

Usage:
    python simulations/analysis/ranking.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load all CSV results into a single DataFrame."""
    all_data = []

    for modality_dir in results_dir.iterdir():
        if not modality_dir.is_dir():
            continue
        modality = modality_dir.name

        for csv_file in modality_dir.glob("*.csv"):
            dgp = csv_file.stem
            df = pd.read_csv(csv_file)
            df["modality"] = modality
            df["dgp"] = dgp
            all_data.append(df)

    if not all_data:
        raise ValueError(f"No CSV files found in {results_dir}")

    return pd.concat(all_data, ignore_index=True)


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean statistics per modality/dgp/method."""
    summary = df.groupby(["modality", "dgp", "method"]).agg({
        "ate_bias": lambda x: np.abs(x).mean(),  # Mean absolute bias
        "ate_covered": "mean",                    # Coverage rate
        "ite_rmse": "mean",                       # Mean ITE RMSE
        "ite_correlation": "mean",                # Mean ITE correlation
        "ate_se": "mean",                         # Mean SE
    }).reset_index()

    summary.columns = ["modality", "dgp", "method", "abs_bias", "coverage", "ite_rmse", "ite_corr", "ate_se"]
    return summary


def determine_winners(summary: pd.DataFrame) -> pd.DataFrame:
    """Determine winner for each metric in each modality/dgp."""
    results = []

    for (modality, dgp), group in summary.groupby(["modality", "dgp"]):
        row = {"modality": modality, "dgp": dgp}

        # ATE Bias: lower is better
        best_bias = group.loc[group["abs_bias"].idxmin()]
        row["bias_winner"] = best_bias["method"]
        row["bias_best"] = best_bias["abs_bias"]

        # Coverage: closer to 0.95 is better
        group = group.copy()
        group["cov_diff"] = np.abs(group["coverage"] - 0.95)
        best_cov = group.loc[group["cov_diff"].idxmin()]
        row["coverage_winner"] = best_cov["method"]
        row["coverage_best"] = best_cov["coverage"]

        # ITE RMSE: lower is better
        best_rmse = group.loc[group["ite_rmse"].idxmin()]
        row["ite_rmse_winner"] = best_rmse["method"]
        row["ite_rmse_best"] = best_rmse["ite_rmse"]

        # ITE Correlation: higher is better
        best_corr = group.loc[group["ite_corr"].idxmax()]
        row["ite_corr_winner"] = best_corr["method"]
        row["ite_corr_best"] = best_corr["ite_corr"]

        # Store all values for comparison
        for _, r in group.iterrows():
            method = r["method"]
            row[f"{method}_bias"] = r["abs_bias"]
            row[f"{method}_coverage"] = r["coverage"]
            row[f"{method}_ite_rmse"] = r["ite_rmse"]
            row[f"{method}_ite_corr"] = r["ite_corr"]

        results.append(row)

    return pd.DataFrame(results)


def print_detailed_table(winners: pd.DataFrame):
    """Print detailed comparison table."""
    print("\n" + "=" * 100)
    print("DETAILED RESULTS BY DGP")
    print("=" * 100)

    # Sort by modality then dgp
    winners = winners.sort_values(["modality", "dgp"])

    for modality in winners["modality"].unique():
        mod_data = winners[winners["modality"] == modality]
        print(f"\n### {modality.upper()} ###\n")
        print(f"{'DGP':<20} {'ATE Bias':<15} {'Coverage':<15} {'ITE RMSE':<15} {'ITE Corr':<15}")
        print("-" * 80)

        for _, row in mod_data.iterrows():
            dgp = row["dgp"]

            # Format each metric with winner highlighted
            def fmt_metric(winner_col, best_col):
                winner = row[winner_col]
                best_val = row[best_col]
                symbol = {"deephte": "D", "causal_forest": "C", "linear_dml": "L"}[winner]
                return f"{symbol}: {best_val:.3f}"

            bias_str = fmt_metric("bias_winner", "bias_best")
            cov_str = fmt_metric("coverage_winner", "coverage_best")
            rmse_str = fmt_metric("ite_rmse_winner", "ite_rmse_best")
            corr_str = fmt_metric("ite_corr_winner", "ite_corr_best")

            print(f"{dgp:<20} {bias_str:<15} {cov_str:<15} {rmse_str:<15} {corr_str:<15}")


def print_win_counts(winners: pd.DataFrame):
    """Print win counts by method and metric."""
    print("\n" + "=" * 100)
    print("WIN COUNTS (D=DeepHTE, C=CausalForest, L=LinearDML)")
    print("=" * 100)

    metrics = [
        ("bias_winner", "ATE Bias (lower=better)"),
        ("coverage_winner", "Coverage (closer to 95%=better)"),
        ("ite_rmse_winner", "ITE RMSE (lower=better)"),
        ("ite_corr_winner", "ITE Correlation (higher=better)"),
    ]

    print(f"\n{'Metric':<35} {'DeepHTE':>10} {'CausalForest':>12} {'LinearDML':>12} {'Total':>8}")
    print("-" * 80)

    total_wins = {"deephte": 0, "causal_forest": 0, "linear_dml": 0}
    total_dgps = 0

    for col, name in metrics:
        counts = winners[col].value_counts()
        d = counts.get("deephte", 0)
        c = counts.get("causal_forest", 0)
        l = counts.get("linear_dml", 0)
        total = d + c + l

        total_wins["deephte"] += d
        total_wins["causal_forest"] += c
        total_wins["linear_dml"] += l
        total_dgps += total

        print(f"{name:<35} {d:>10} {c:>12} {l:>12} {total:>8}")

    print("-" * 80)
    print(f"{'TOTAL':<35} {total_wins['deephte']:>10} {total_wins['causal_forest']:>12} {total_wins['linear_dml']:>12} {total_dgps:>8}")

    # Percentages
    print(f"{'PERCENTAGE':<35} {100*total_wins['deephte']/total_dgps:>9.1f}% {100*total_wins['causal_forest']/total_dgps:>11.1f}% {100*total_wins['linear_dml']/total_dgps:>11.1f}%")


def print_wins_by_modality(winners: pd.DataFrame):
    """Print win breakdown by modality."""
    print("\n" + "=" * 100)
    print("WINS BY MODALITY")
    print("=" * 100)

    for modality in sorted(winners["modality"].unique()):
        mod_data = winners[winners["modality"] == modality]
        n_dgps = len(mod_data)

        print(f"\n### {modality.upper()} ({n_dgps} DGPs) ###")
        print(f"{'Metric':<25} {'DeepHTE':>10} {'CausalForest':>12} {'LinearDML':>12}")
        print("-" * 60)

        for col, name in [("bias_winner", "ATE Bias"), ("coverage_winner", "Coverage"),
                          ("ite_rmse_winner", "ITE RMSE"), ("ite_corr_winner", "ITE Corr")]:
            counts = mod_data[col].value_counts()
            d = counts.get("deephte", 0)
            c = counts.get("causal_forest", 0)
            l = counts.get("linear_dml", 0)
            print(f"{name:<25} {d:>10} {c:>12} {l:>12}")


def print_head_to_head(winners: pd.DataFrame):
    """Print head-to-head comparison: DeepHTE vs CausalForest."""
    print("\n" + "=" * 100)
    print("HEAD-TO-HEAD: DeepHTE vs CausalForest (excluding LinearDML wins)")
    print("=" * 100)

    metrics = ["bias_winner", "coverage_winner", "ite_rmse_winner", "ite_corr_winner"]
    metric_names = ["ATE Bias", "Coverage", "ITE RMSE", "ITE Correlation"]

    print(f"\n{'Metric':<25} {'DeepHTE':>12} {'CausalForest':>14} {'Winner':>12}")
    print("-" * 65)

    for col, name in zip(metrics, metric_names):
        d = (winners[col] == "deephte").sum()
        c = (winners[col] == "causal_forest").sum()

        if d > c:
            winner = "DeepHTE"
        elif c > d:
            winner = "CausalForest"
        else:
            winner = "TIE"

        print(f"{name:<25} {d:>12} {c:>14} {winner:>12}")


def print_coverage_warnings(summary: pd.DataFrame):
    """Print warnings about poor coverage."""
    print("\n" + "=" * 100)
    print("COVERAGE WARNINGS (Methods with coverage < 80% or > 99%)")
    print("=" * 100)

    deephte = summary[summary["method"] == "deephte"]
    poor_coverage = deephte[(deephte["coverage"] < 0.80) | (deephte["coverage"] > 0.99)]

    if len(poor_coverage) > 0:
        print(f"\n{'Modality':<15} {'DGP':<20} {'Coverage':>12} {'Status':<15}")
        print("-" * 65)

        for _, row in poor_coverage.sort_values("coverage").iterrows():
            cov = row["coverage"]
            status = "UNDERCOVERAGE" if cov < 0.80 else "OVERCOVERAGE"
            print(f"{row['modality']:<15} {row['dgp']:<20} {cov:>11.1%} {status:<15}")
    else:
        print("\nNo severe coverage issues found.")


def create_latex_table(winners: pd.DataFrame) -> str:
    """Create LaTeX table for paper."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Method Comparison: Winner by DGP and Metric}",
        r"\label{tab:rankings}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Modality & DGP & ATE Bias & Coverage & ITE RMSE & ITE Corr \\",
        r"\midrule",
    ]

    winners = winners.sort_values(["modality", "dgp"])
    current_modality = None

    symbol_map = {"deephte": r"\textbf{D}", "causal_forest": "C", "linear_dml": "L"}

    for _, row in winners.iterrows():
        modality = row["modality"]
        if modality != current_modality:
            if current_modality is not None:
                lines.append(r"\midrule")
            current_modality = modality

        dgp = row["dgp"].replace("_", r"\_")
        bias = symbol_map[row["bias_winner"]]
        cov = symbol_map[row["coverage_winner"]]
        rmse = symbol_map[row["ite_rmse_winner"]]
        corr = symbol_map[row["ite_corr_winner"]]

        lines.append(f"{modality} & {dgp} & {bias} & {cov} & {rmse} & {corr} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    # Find results directory
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    print("Loading simulation results...")
    df = load_all_results(results_dir)
    print(f"Loaded {len(df)} rows from {df['dgp'].nunique()} DGPs across {df['modality'].nunique()} modalities")

    # Compute summary statistics
    summary = compute_summary_stats(df)

    # Determine winners
    winners = determine_winners(summary)

    # Print reports
    print_win_counts(winners)
    print_head_to_head(winners)
    print_wins_by_modality(winners)
    print_detailed_table(winners)
    print_coverage_warnings(summary)

    # Save results
    output_file = results_dir / "rankings.csv"
    winners.to_csv(output_file, index=False)
    print(f"\n\nRankings saved to: {output_file}")

    # Save LaTeX table
    latex_file = results_dir / "rankings_table.tex"
    latex_table = create_latex_table(winners)
    with open(latex_file, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {latex_file}")

    # Print summary
    print("\n" + "=" * 100)
    print("EXECUTIVE SUMMARY")
    print("=" * 100)

    total = len(winners) * 4  # 4 metrics
    d_wins = sum((winners[col] == "deephte").sum() for col in ["bias_winner", "coverage_winner", "ite_rmse_winner", "ite_corr_winner"])
    c_wins = sum((winners[col] == "causal_forest").sum() for col in ["bias_winner", "coverage_winner", "ite_rmse_winner", "ite_corr_winner"])
    l_wins = sum((winners[col] == "linear_dml").sum() for col in ["bias_winner", "coverage_winner", "ite_rmse_winner", "ite_corr_winner"])

    print(f"""
Overall Win Rate:
  - DeepHTE:      {d_wins:>3}/{total} ({100*d_wins/total:.1f}%)
  - CausalForest: {c_wins:>3}/{total} ({100*c_wins/total:.1f}%)
  - LinearDML:    {l_wins:>3}/{total} ({100*l_wins/total:.1f}%)

Key Findings:
  - DeepHTE dominates on: {', '.join(winners[winners['ite_rmse_winner'] == 'deephte']['modality'].unique())} modalities (ITE RMSE)
  - CausalForest dominates on: {', '.join(winners[winners['ite_rmse_winner'] == 'causal_forest']['modality'].unique())} modalities (ITE RMSE)
""")


if __name__ == "__main__":
    main()
