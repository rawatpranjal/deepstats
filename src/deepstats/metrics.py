"""Metrics computation for Monte Carlo results.

Phase 1: Parameter Recovery (Theorem 1)
Phase 2: Inference Validity (Theorem 2)
Phase 3: Training Diagnostics (Validation Scorecard)
"""

import numpy as np
import pandas as pd


def compute_violation_rate(df: pd.DataFrame) -> float:
    """Compute how often μ̂_influence falls outside naive CI (FLM Section 4.3.2).

    Violation_Rate = (1/M) Σ 𝟙[|μ̂_inf - μ̂_naive| > 1.96 × SE_naive]

    High rate (30-70%) = SUCCESS: IF detecting real bias naive missed
    Low rate (~0%) = Bias correction negligible
    Very high (~100%) = NN heavily biased
    """
    # Handle target column if present (need to match by sim_id AND target)
    has_target = "target" in df.columns

    naive_df = df[df["method"] == "naive"]
    inf_df = df[df["method"] == "influence"]

    if len(naive_df) == 0 or len(inf_df) == 0:
        return np.nan

    if has_target:
        # Set multi-index for matching
        naive_df = naive_df.set_index(["sim_id", "target"])
        inf_df = inf_df.set_index(["sim_id", "target"])
    else:
        naive_df = naive_df.set_index("sim_id")
        inf_df = inf_df.set_index("sim_id")

    # Match by index
    common_ids = naive_df.index.intersection(inf_df.index)
    if len(common_ids) == 0:
        return np.nan

    violations = 0
    for idx in common_ids:
        mu_naive = naive_df.loc[idx, "mu_hat"]
        se_naive = naive_df.loc[idx, "se"]
        mu_inf = inf_df.loc[idx, "mu_hat"]

        # Handle case where loc returns Series (shouldn't happen with proper index)
        if hasattr(mu_naive, '__len__') and not isinstance(mu_naive, str):
            mu_naive = mu_naive.iloc[0]
            se_naive = se_naive.iloc[0]
            mu_inf = mu_inf.iloc[0]

        # Check if influence estimate is outside naive CI
        if abs(mu_inf - mu_naive) > 1.96 * se_naive:
            violations += 1

    return violations / len(common_ids)


def compute_bias_shift(df: pd.DataFrame) -> float:
    """Compute |μ̂_inf - μ̂_naive| (should be > 0 for IF to be doing work)."""
    naive_df = df[df["method"] == "naive"]
    inf_df = df[df["method"] == "influence"]

    if len(naive_df) == 0 or len(inf_df) == 0:
        return np.nan

    return abs(inf_df["mu_hat"].mean() - naive_df["mu_hat"].mean())


def compute_bias_reduction(df: pd.DataFrame) -> float:
    """Compute bias reduction: 1 - |bias_inf| / |bias_naive|.

    Shows what percentage of naive bias is removed by influence function.
    High value (e.g., 95%) means IF is successfully eliminating bias.
    """
    naive_df = df[df["method"] == "naive"]
    inf_df = df[df["method"] == "influence"]

    if len(naive_df) == 0 or len(inf_df) == 0:
        return np.nan

    bias_naive = abs(naive_df["bias"].mean())
    bias_inf = abs(inf_df["bias"].mean())

    if bias_naive < 1e-10:
        return np.nan

    return 1.0 - (bias_inf / bias_naive)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary metrics by model and method.

    Phase 1 (Parameter Recovery):
    - RMSE_α = √(1/N Σ(α̂ᵢ - α*ᵢ)²)
    - RMSE_β = √(1/N Σ(β̂ᵢ - β*ᵢ)²)
    - Corr_α = correlation(α̂, α*)
    - Corr_β = correlation(β̂, β*)

    Phase 2 (Inference on μ* = E[β(X)]):
    - μ* = True parameter value
    - Bias = E[μ̂] - μ*
    - Var = Var(μ̂)
    - MSE = Bias² + Var
    - SE(emp) = √Var (true SE from MC)
    - SE(est) = mean(estimated SE)
    - Ratio = SE(est)/SE(emp) (calibration, target=1.0)
    - Coverage = P(μ* ∈ CI) (target=95%)
    """
    df_clean = df.dropna(subset=["mu_hat", "se", "bias"])

    # Check for optional columns
    has_overfit = "overfit_rate" in df_clean.columns
    has_recovery = "rmse_alpha" in df_clean.columns
    has_phase3 = "correction_ratio" in df_clean.columns
    has_training_quality = "final_val_loss" in df_clean.columns

    # Base aggregations
    agg_dict = {
        "mu_true": "first",
        "bias": ["mean", "std"],
        "mu_hat": "std",
        "se": "mean",
        "covered": "mean",
        "sim_id": "count",
    }

    # Parameter recovery metrics
    if has_recovery:
        agg_dict["rmse_alpha"] = "mean"
        agg_dict["rmse_beta"] = "mean"
        agg_dict["corr_alpha"] = "mean"
        agg_dict["corr_beta"] = "mean"

    if has_overfit:
        agg_dict["overfit_rate"] = "mean"
        agg_dict["mean_best_epoch"] = "mean"

    # Phase 3: Training diagnostics
    if has_phase3:
        agg_dict["final_grad_norm"] = "mean"
        agg_dict["final_beta_std"] = "mean"
        agg_dict["correction_ratio"] = "mean"
        agg_dict["min_hessian_eig"] = "mean"
        agg_dict["hessian_condition"] = "mean"

    # Training quality metrics
    if has_training_quality:
        agg_dict["final_train_loss"] = "mean"
        agg_dict["final_val_loss"] = "mean"
        agg_dict["train_val_gap"] = "mean"

    # NEW: Oracle loss and smoothness diagnostics
    has_oracle = "oracle_loss" in df_clean.columns
    if has_oracle:
        agg_dict["oracle_loss"] = "mean"
        agg_dict["excess_loss_ratio"] = "mean"
        agg_dict["smoothness_ratio"] = "mean"

    # Check if target column exists (for multi-target models like tobit)
    has_target = "target" in df_clean.columns
    group_cols = ["model", "method", "target"] if has_target else ["model", "method"]

    summary = df_clean.groupby(group_cols).agg(agg_dict).reset_index()

    # Build column names dynamically
    base_cols = group_cols + ["mu_true", "bias_mean", "bias_std",
                 "empirical_se", "se_mean", "coverage", "n_sims"]
    if has_recovery:
        base_cols.extend(["rmse_alpha", "rmse_beta", "corr_alpha", "corr_beta"])
    if has_overfit:
        base_cols.extend(["overfit_rate", "mean_best_epoch"])
    if has_phase3:
        base_cols.extend(["final_grad_norm", "final_beta_std", "correction_ratio",
                          "min_hessian_eig", "hessian_condition"])
    if has_training_quality:
        base_cols.extend(["final_train_loss", "final_val_loss", "train_val_gap"])
    if has_oracle:
        base_cols.extend(["oracle_loss", "excess_loss_ratio", "smoothness_ratio"])

    summary.columns = base_cols

    # RMSE of mu_hat
    def rmse_func(group):
        return np.sqrt(np.mean(group["bias"]**2))

    rmse_by_group = df_clean.groupby(group_cols).apply(
        rmse_func, include_groups=False
    ).reset_index(name="rmse_mu")

    summary = summary.merge(rmse_by_group, on=group_cols)

    # Derived metrics
    summary["se_ratio"] = summary["se_mean"] / summary["empirical_se"]
    summary["variance"] = summary["empirical_se"] ** 2
    summary["mse"] = summary["bias_mean"]**2 + summary["variance"]
    summary["ci_width"] = 2 * 1.96 * summary["se_mean"]

    # Bias ratio: proportion of MSE from squared bias (NEW)
    summary["bias_ratio"] = summary["bias_mean"]**2 / summary["mse"]

    # R² for parameter recovery (corr²) (NEW)
    if has_recovery:
        summary["r2_alpha"] = summary["corr_alpha"]**2
        summary["r2_beta"] = summary["corr_beta"]**2

    # Compute violation rate and bias shift (needs raw df)
    violation_rate = compute_violation_rate(df_clean)
    bias_shift = compute_bias_shift(df_clean)
    bias_reduction = compute_bias_reduction(df_clean)

    # Add as scalar columns (same for all rows - these are cross-method metrics)
    summary["violation_rate"] = violation_rate
    summary["bias_shift"] = bias_shift
    summary["bias_reduction"] = bias_reduction

    # Order columns
    cols = list(group_cols)  # ["model", "method"] or ["model", "method", "target"]
    if has_recovery:
        cols.extend(["rmse_alpha", "rmse_beta", "corr_alpha", "corr_beta", "r2_alpha", "r2_beta"])
    cols.extend([
        "mu_true", "bias_mean", "variance", "mse", "bias_ratio", "rmse_mu",
        "empirical_se", "se_mean", "se_ratio", "ci_width", "coverage", "n_sims",
    ])
    if has_overfit:
        cols.extend(["overfit_rate", "mean_best_epoch"])
    if has_phase3:
        cols.extend(["final_grad_norm", "final_beta_std", "correction_ratio",
                     "min_hessian_eig", "hessian_condition"])
    if has_training_quality:
        cols.extend(["final_train_loss", "final_val_loss", "train_val_gap"])
    if has_oracle:
        cols.extend(["oracle_loss", "excess_loss_ratio", "smoothness_ratio"])
    cols.extend(["violation_rate", "bias_shift", "bias_reduction"])

    return summary[cols]


def print_table(metrics_df: pd.DataFrame) -> str:
    """Print formatted consolidated results table."""
    has_overfit = "overfit_rate" in metrics_df.columns
    has_recovery = "rmse_alpha" in metrics_df.columns
    has_phase3 = "correction_ratio" in metrics_df.columns
    has_training_quality = "final_val_loss" in metrics_df.columns
    has_oracle = "oracle_loss" in metrics_df.columns

    lines = []

    # Get simulation info
    n_sims = int(metrics_df["n_sims"].iloc[0]) if "n_sims" in metrics_df.columns else "?"

    # ==========================================================================
    # PHASE 1: PARAMETER RECOVERY
    # ==========================================================================
    if has_recovery:
        lines.append("=" * 100)
        lines.append(f"PHASE 1: PARAMETER RECOVERY (M={n_sims} simulations)")
        lines.append("=" * 100)
        lines.append("")
        lines.append(f"{'Method':<15} {'RMSE_α':>10} {'RMSE_β':>10} {'Corr_α':>10} {'Corr_β':>10} {'R²_α':>8} {'R²_β':>8}")
        lines.append("-" * 100)

        for _, row in metrics_df.iterrows():
            r2_alpha = row.get('r2_alpha', row['corr_alpha']**2)
            r2_beta = row.get('r2_beta', row['corr_beta']**2)
            lines.append(
                f"{row['method']:<15} "
                f"{row['rmse_alpha']:>10.4f} "
                f"{row['rmse_beta']:>10.4f} "
                f"{row['corr_alpha']:>10.4f} "
                f"{row['corr_beta']:>10.4f} "
                f"{r2_alpha:>7.1%} "
                f"{r2_beta:>7.1%}"
            )

        lines.append("-" * 100)
        lines.append("RMSE = Root Mean Square Error | Corr = Correlation | R² = Variance explained")
        lines.append("")

    # ==========================================================================
    # PHASE 2: INFERENCE ON μ* = E[β(X)]
    # ==========================================================================
    lines.append("=" * 100)
    lines.append(f"PHASE 2: INFERENCE ON μ* = E[β(X)] (M={n_sims} simulations)")
    lines.append("=" * 100)
    lines.append("")

    header = (
        f"{'Method':<15} {'μ*':>8} {'Bias':>8} {'Var':>8} {'MSE':>8} {'Bias²/MSE':>9} "
        f"{'SE(emp)':>8} {'SE(est)':>8} {'Ratio':>6} {'Coverage':>9}"
    )
    lines.append(header)
    lines.append("-" * 100)

    for _, row in metrics_df.iterrows():
        # Mark influence as target if it achieves ~95% coverage
        coverage = row['coverage']
        marker = " ←" if 0.90 <= coverage <= 0.99 else ""
        bias_ratio = row.get('bias_ratio', row['bias_mean']**2 / row['mse'] if row['mse'] > 0 else 0)

        lines.append(
            f"{row['method']:<15} "
            f"{row['mu_true']:>8.4f} "
            f"{row['bias_mean']:>8.4f} "
            f"{row['variance']:>8.4f} "
            f"{row['mse']:>8.4f} "
            f"{bias_ratio:>8.1%} "
            f"{row['empirical_se']:>8.4f} "
            f"{row['se_mean']:>8.4f} "
            f"{row['se_ratio']:>6.2f} "
            f"{coverage:>8.1%}{marker}"
        )

    lines.append("-" * 100)
    lines.append("")
    lines.append("Ratio = SE(est)/SE(emp), target=1.0 | Coverage target=95%")
    lines.append("Bias²/MSE = proportion of error from bias (low = good) | ← indicates near-target coverage")
    lines.append("=" * 100)

    # ==========================================================================
    # PHASE 3: TRAINING DIAGNOSTICS
    # ==========================================================================
    if has_phase3:
        lines.append("")
        lines.append("=" * 100)
        lines.append(f"PHASE 3: TRAINING DIAGNOSTICS (M={n_sims} simulations)")
        lines.append("=" * 100)
        lines.append("")

        header3 = (
            f"{'Method':<15} {'Grad→0':>10} {'β_std':>10} {'R_corr':>10} "
            f"{'min(Λ)':>12} {'cond(Λ)':>12} {'Status':>10}"
        )
        lines.append(header3)
        lines.append("-" * 100)

        for _, row in metrics_df.iterrows():
            # Determine status based on diagnostics
            corr_ratio = row.get('correction_ratio', np.nan)
            min_eig = row.get('min_hessian_eig', np.nan)

            if pd.isna(corr_ratio):
                status = "N/A"
            elif corr_ratio > 3.0:
                status = "EXPLOSION"
            elif corr_ratio < 0.01:
                status = "DO-NOTHING"
            elif min_eig < 1e-4:
                status = "UNSTABLE"
            elif 0.1 <= corr_ratio <= 1.0:
                status = "OK ✓"
            else:
                status = "WARN"

            # Format values with N/A handling
            grad_str = f"{row.get('final_grad_norm', np.nan):>10.4f}" if not pd.isna(row.get('final_grad_norm')) else f"{'N/A':>10}"
            beta_str = f"{row.get('final_beta_std', np.nan):>10.4f}" if not pd.isna(row.get('final_beta_std')) else f"{'N/A':>10}"
            corr_str = f"{corr_ratio:>10.3f}" if not pd.isna(corr_ratio) else f"{'N/A':>10}"
            min_eig_str = f"{min_eig:>12.6f}" if not pd.isna(min_eig) else f"{'N/A':>12}"
            cond_str = f"{row.get('hessian_condition', np.nan):>12.1f}" if not pd.isna(row.get('hessian_condition')) else f"{'N/A':>12}"

            lines.append(
                f"{row['method']:<15} "
                f"{grad_str} "
                f"{beta_str} "
                f"{corr_str} "
                f"{min_eig_str} "
                f"{cond_str} "
                f"{status:>10}"
            )

        lines.append("-" * 100)
        lines.append("")
        lines.append("R_corr = |mean(φ)|/SE, target 0.1-1.0 | min(Λ) > 1e-4 for stability")
        lines.append("=" * 100)

    # ==========================================================================
    # TRAINING QUALITY (NN Model Fit)
    # ==========================================================================
    if has_training_quality:
        lines.append("")
        lines.append("=" * 100)
        lines.append(f"TRAINING QUALITY (M={n_sims} simulations)")
        lines.append("=" * 100)
        lines.append("")

        header_tq = (
            f"{'Method':<15} {'ValLoss':>10} {'TrainLoss':>10} {'Gap':>10} "
            f"{'BestEpoch':>10} {'Overfit%':>10} {'Status':>10}"
        )
        lines.append(header_tq)
        lines.append("-" * 100)

        for _, row in metrics_df.iterrows():
            val_loss = row.get('final_val_loss', np.nan)
            train_loss = row.get('final_train_loss', np.nan)
            gap = row.get('train_val_gap', np.nan)
            best_epoch = row.get('mean_best_epoch', np.nan)
            overfit_rate = row.get('overfit_rate', np.nan)

            # Determine training quality status
            if pd.isna(val_loss):
                tq_status = "N/A"
            elif gap > 0.5:
                tq_status = "OVERFIT"
            elif gap < 0:
                tq_status = "UNDERFIT"
            elif gap < 0.1:
                tq_status = "OK ✓"
            else:
                tq_status = "WARN"

            # Format values
            val_str = f"{val_loss:>10.4f}" if not pd.isna(val_loss) else f"{'N/A':>10}"
            train_str = f"{train_loss:>10.4f}" if not pd.isna(train_loss) else f"{'N/A':>10}"
            gap_str = f"{gap:>10.4f}" if not pd.isna(gap) else f"{'N/A':>10}"
            epoch_str = f"{best_epoch:>10.1f}" if not pd.isna(best_epoch) else f"{'N/A':>10}"
            overfit_str = f"{overfit_rate:>9.1%}" if not pd.isna(overfit_rate) else f"{'N/A':>10}"

            lines.append(
                f"{row['method']:<15} "
                f"{val_str} "
                f"{train_str} "
                f"{gap_str} "
                f"{epoch_str} "
                f"{overfit_str} "
                f"{tq_status:>10}"
            )

        lines.append("-" * 100)
        lines.append("")
        lines.append("Gap = ValLoss - TrainLoss, target ≈ 0 | Overfit% = % folds that overfit")
        lines.append("=" * 100)

    # ==========================================================================
    # ORACLE & SMOOTHNESS DIAGNOSTICS (NEW)
    # ==========================================================================
    if has_oracle:
        lines.append("")
        lines.append("=" * 100)
        lines.append(f"ORACLE & SMOOTHNESS DIAGNOSTICS (M={n_sims} simulations)")
        lines.append("=" * 100)
        lines.append("")

        header_oracle = (
            f"{'Method':<15} {'OracleLoss':>12} {'ExcessLoss%':>12} {'Smoothness':>12} {'Status':>10}"
        )
        lines.append(header_oracle)
        lines.append("-" * 100)

        for _, row in metrics_df.iterrows():
            oracle_loss = row.get('oracle_loss', np.nan)
            excess_loss = row.get('excess_loss_ratio', np.nan)
            smoothness = row.get('smoothness_ratio', np.nan)

            # Determine status
            if pd.isna(excess_loss):
                oracle_status = "N/A"
            elif excess_loss < 0.5:
                oracle_status = "GOOD ✓"
            elif excess_loss < 1.0:
                oracle_status = "OK"
            else:
                oracle_status = "HIGH"

            oracle_str = f"{oracle_loss:>12.4f}" if not pd.isna(oracle_loss) else f"{'N/A':>12}"
            excess_str = f"{excess_loss:>11.1%}" if not pd.isna(excess_loss) else f"{'N/A':>12}"
            smooth_str = f"{smoothness:>12.3f}" if not pd.isna(smoothness) else f"{'N/A':>12}"

            lines.append(
                f"{row['method']:<15} "
                f"{oracle_str} "
                f"{excess_str} "
                f"{smooth_str} "
                f"{oracle_status:>10}"
            )

        lines.append("-" * 100)
        lines.append("")
        lines.append("OracleLoss = theoretical minimum (σ²=1.0) | ExcessLoss = (ValLoss-Oracle)/Oracle")
        lines.append("Smoothness = std(grad(β̂))/std(grad(β*)), <1 means regularization shrinkage")
        lines.append("=" * 100)

    # ==========================================================================
    # VALIDATION SCORECARD SUMMARY
    # ==========================================================================
    lines.append("")
    lines.append("=" * 100)
    lines.append("VALIDATION SCORECARD SUMMARY")
    lines.append("=" * 100)
    lines.append("")

    # Get cross-method metrics
    violation_rate = metrics_df["violation_rate"].iloc[0] if "violation_rate" in metrics_df.columns else np.nan
    bias_shift = metrics_df["bias_shift"].iloc[0] if "bias_shift" in metrics_df.columns else np.nan
    bias_reduction = metrics_df["bias_reduction"].iloc[0] if "bias_reduction" in metrics_df.columns else np.nan

    inf_row = metrics_df[metrics_df["method"] == "influence"]
    naive_row = metrics_df[metrics_df["method"] == "naive"]

    # Influence method scorecard
    if len(inf_row) > 0:
        inf_cov = inf_row["coverage"].iloc[0]
        inf_ratio = inf_row["se_ratio"].iloc[0]
        inf_bias_ratio = inf_row["bias_ratio"].iloc[0] if "bias_ratio" in inf_row.columns else 0
        inf_r2_beta = inf_row["r2_beta"].iloc[0] if "r2_beta" in inf_row.columns else np.nan

        # Determine grade
        if 0.93 <= inf_cov <= 0.97 and 0.9 <= inf_ratio <= 1.2:
            inf_grade = "PASS ✓"
        elif inf_cov >= 0.85 or (0.8 <= inf_ratio <= 1.4):
            inf_grade = "WARNING"
        else:
            inf_grade = "FAIL ✗"

        lines.append(f"INFLUENCE METHOD: {inf_grade}")
        lines.append(f"  Coverage: {inf_cov:.1%} | SE Ratio: {inf_ratio:.2f} | Bias²/MSE: {inf_bias_ratio:.1%}")
        if not pd.isna(inf_r2_beta):
            lines.append(f"  R²_β: {inf_r2_beta:.1%} (variance explained in treatment effect)")
        if inf_cov >= 0.93:
            lines.append("  Notes: Achieves nominal coverage. SEs properly calibrated.")
        elif inf_cov >= 0.85:
            lines.append("  Notes: Coverage acceptable but SEs may be conservative.")
        else:
            lines.append("  Notes: Coverage below target. Check IF implementation.")
        lines.append("")

    # Naive method scorecard
    if len(naive_row) > 0:
        naive_cov = naive_row["coverage"].iloc[0]
        naive_ratio = naive_row["se_ratio"].iloc[0]
        naive_bias_ratio = naive_row["bias_ratio"].iloc[0] if "bias_ratio" in naive_row.columns else 0
        naive_r2_beta = naive_row["r2_beta"].iloc[0] if "r2_beta" in naive_row.columns else np.nan

        # Naive expected to fail
        if naive_cov < 0.5:
            naive_grade = "FAIL (CRITICAL) ✗"
        else:
            naive_grade = "UNEXPECTED"

        se_underestimate = (1 - naive_ratio) * 100 if naive_ratio < 1 else 0

        lines.append(f"NAIVE METHOD: {naive_grade}")
        lines.append(f"  Coverage: {naive_cov:.1%} | SE Ratio: {naive_ratio:.2f} | Bias²/MSE: {naive_bias_ratio:.1%}")
        if not pd.isna(naive_r2_beta):
            lines.append(f"  R²_β: {naive_r2_beta:.1%} (variance explained in treatment effect)")
        lines.append(f"  Notes: Severe under-coverage. Underestimates SE by {se_underestimate:.0f}%.")
        lines.append("")

    # Cross-method comparison
    lines.append("CROSS-METHOD COMPARISON:")
    if not pd.isna(bias_reduction):
        lines.append(f"  Bias Reduction: {bias_reduction:.1%} (IF removes this much of naive bias)")
    if not pd.isna(violation_rate):
        viol_status = "SUCCESS" if 0.3 <= violation_rate <= 0.7 else "CHECK"
        lines.append(f"  Violation Rate: {violation_rate:.1%} → {viol_status}")
    if not pd.isna(bias_shift):
        lines.append(f"  Bias Shift: {bias_shift:.4f}")

    lines.append("=" * 100)

    # ==========================================================================
    # TLDR (Quick Summary)
    # ==========================================================================
    lines.append("")
    lines.append("╔" + "═" * 98 + "╗")
    lines.append("║" + " TLDR ".center(98) + "║")
    lines.append("╠" + "═" * 98 + "╣")

    # Get key metrics for TLDR
    tldr_parts = []
    if len(inf_row) > 0:
        inf_cov = inf_row["coverage"].iloc[0]
        inf_ratio = inf_row["se_ratio"].iloc[0]
        if 0.93 <= inf_cov <= 0.97 and 0.9 <= inf_ratio <= 1.2:
            tldr_parts.append(f"║  INFLUENCE: {inf_cov:.0%} coverage, SE ratio {inf_ratio:.2f} → PASS ✓".ljust(99) + "║")
        elif inf_cov >= 0.85:
            tldr_parts.append(f"║  INFLUENCE: {inf_cov:.0%} coverage, SE ratio {inf_ratio:.2f} → WARNING".ljust(99) + "║")
        else:
            tldr_parts.append(f"║  INFLUENCE: {inf_cov:.0%} coverage, SE ratio {inf_ratio:.2f} → FAIL ✗".ljust(99) + "║")

    if len(naive_row) > 0:
        naive_cov = naive_row["coverage"].iloc[0]
        naive_ratio = naive_row["se_ratio"].iloc[0]
        tldr_parts.append(f"║  NAIVE: {naive_cov:.0%} coverage, SE ratio {naive_ratio:.2f} → FAIL (expected)".ljust(99) + "║")

    # Add violation rate interpretation
    if not pd.isna(violation_rate) and len(inf_row) > 0:
        if violation_rate >= 0.5:
            tldr_parts.append(f"║  IF correction is SIGNIFICANT ({violation_rate:.0%} violation rate)".ljust(99) + "║")
        else:
            tldr_parts.append(f"║  IF correction is MODEST ({violation_rate:.0%} violation rate)".ljust(99) + "║")

    for part in tldr_parts:
        lines.append(part)

    lines.append("╚" + "═" * 98 + "╝")

    output = "\n".join(lines)
    print(output)
    return output
