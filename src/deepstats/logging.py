"""Comprehensive logging for FLM validation reports.

Creates machine-readable JSON logs containing ALL statistics, metrics,
and data for later AI analysis.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_float(val: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(val, (np.floating, np.integer)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if pd.isna(val):
        return None
    return val


def extract_phase1(metrics_df: pd.DataFrame) -> dict:
    """Extract Phase 1: Parameter Recovery metrics."""
    result = {}
    for _, row in metrics_df.iterrows():
        method = row["method"]
        corr_alpha = row.get("corr_alpha")
        corr_beta = row.get("corr_beta")
        result[method] = {
            "rmse_alpha": _safe_float(row.get("rmse_alpha")),
            "rmse_beta": _safe_float(row.get("rmse_beta")),
            "corr_alpha": _safe_float(corr_alpha),
            "corr_beta": _safe_float(corr_beta),
            # R² = correlation squared (NEW)
            "r2_alpha": _safe_float(corr_alpha**2 if corr_alpha is not None and not pd.isna(corr_alpha) else None),
            "r2_beta": _safe_float(corr_beta**2 if corr_beta is not None and not pd.isna(corr_beta) else None),
        }
    return result


def extract_phase2(metrics_df: pd.DataFrame) -> dict:
    """Extract Phase 2: Inference Validity metrics."""
    result = {}
    for _, row in metrics_df.iterrows():
        method = row["method"]
        result[method] = {
            "mu_true": _safe_float(row.get("mu_true")),
            "mu_hat_mean": _safe_float(row.get("mu_true", 0) + row.get("bias_mean", 0)),
            "bias": _safe_float(row.get("bias_mean")),
            "variance": _safe_float(row.get("variance")),
            "mse": _safe_float(row.get("mse")),
            "bias_ratio": _safe_float(row.get("bias_ratio")),  # NEW: Bias²/MSE
            "rmse_mu": _safe_float(row.get("rmse_mu")),
            "empirical_se": _safe_float(row.get("empirical_se")),
            "estimated_se": _safe_float(row.get("se_mean")),
            "se_ratio": _safe_float(row.get("se_ratio")),
            "ci_width": _safe_float(row.get("ci_width")),
            "coverage": _safe_float(row.get("coverage")),
            "n_sims": int(row.get("n_sims", 0)),
        }
    return result


def extract_phase3(metrics_df: pd.DataFrame) -> dict:
    """Extract Phase 3: Training Diagnostics."""
    result = {}
    for _, row in metrics_df.iterrows():
        method = row["method"]
        result[method] = {
            "final_grad_norm": _safe_float(row.get("final_grad_norm")),
            "final_beta_std": _safe_float(row.get("final_beta_std")),
            "overfit_rate": _safe_float(row.get("overfit_rate")),
            "mean_best_epoch": _safe_float(row.get("mean_best_epoch")),
        }
        # Influence-only metrics
        if method == "influence":
            result[method].update({
                "correction_ratio": _safe_float(row.get("correction_ratio")),
                "min_hessian_eig": _safe_float(row.get("min_hessian_eig")),
                "hessian_condition": _safe_float(row.get("hessian_condition")),
            })
    return result


def extract_training_quality(metrics_df: pd.DataFrame) -> dict:
    """Extract training quality metrics for NN model fit assessment."""
    result = {}
    for _, row in metrics_df.iterrows():
        method = row["method"]
        result[method] = {
            "final_train_loss": _safe_float(row.get("final_train_loss")),
            "final_val_loss": _safe_float(row.get("final_val_loss")),
            "train_val_gap": _safe_float(row.get("train_val_gap")),
            "mean_best_epoch": _safe_float(row.get("mean_best_epoch")),
            "overfit_rate": _safe_float(row.get("overfit_rate")),
        }
    return result


def compute_scorecard(metrics_df: pd.DataFrame) -> dict:
    """Compute validation scorecard with per-method grades."""
    scorecard = {
        "influence_method": None,
        "naive_method": None,
        "cross_method": {},
        "overall_grade": "FAIL",
    }

    # Get influence row
    inf_row = metrics_df[metrics_df["method"] == "influence"]
    naive_row = metrics_df[metrics_df["method"] == "naive"]

    # Influence method scorecard
    if len(inf_row) > 0:
        inf = inf_row.iloc[0]
        coverage = inf.get("coverage", 0)
        se_ratio = inf.get("se_ratio", 0)
        bias_ratio = inf.get("bias_ratio", 0)
        r2_beta = inf.get("r2_beta")

        # Determine grade
        if 0.93 <= coverage <= 0.97 and 0.9 <= se_ratio <= 1.2:
            grade = "PASS"
            notes = "Achieves nominal coverage. SEs properly calibrated."
        elif coverage >= 0.85 or (0.8 <= se_ratio <= 1.4):
            grade = "WARNING"
            notes = "Coverage acceptable but SEs may be conservative."
        else:
            grade = "FAIL"
            notes = "Coverage below target. Check IF implementation."

        scorecard["influence_method"] = {
            "grade": grade,
            "coverage": _safe_float(coverage),
            "se_ratio": _safe_float(se_ratio),
            "bias_ratio": _safe_float(bias_ratio),
            "r2_beta": _safe_float(r2_beta),
            "notes": notes,
        }

    # Naive method scorecard
    if len(naive_row) > 0:
        naive = naive_row.iloc[0]
        coverage = naive.get("coverage", 0)
        se_ratio = naive.get("se_ratio", 0)
        bias_ratio = naive.get("bias_ratio", 0)
        r2_beta = naive.get("r2_beta")

        # Naive expected to fail
        if coverage < 0.5:
            grade = "FAIL (CRITICAL)"
            se_underestimate = (1 - se_ratio) * 100 if se_ratio < 1 else 0
            notes = f"Severe under-coverage. Underestimates SE by {se_underestimate:.0f}%."
        else:
            grade = "UNEXPECTED"
            notes = "Naive coverage unexpectedly high. Check DGP."

        scorecard["naive_method"] = {
            "grade": grade,
            "coverage": _safe_float(coverage),
            "se_ratio": _safe_float(se_ratio),
            "bias_ratio": _safe_float(bias_ratio),
            "r2_beta": _safe_float(r2_beta),
            "notes": notes,
        }

    # Cross-method comparison
    if len(inf_row) > 0:
        inf = inf_row.iloc[0]
        scorecard["cross_method"] = {
            "violation_rate": _safe_float(inf.get("violation_rate")),
            "bias_shift": _safe_float(inf.get("bias_shift")),
            "bias_reduction": _safe_float(inf.get("bias_reduction")),
        }

    # Overall grade (based on influence method)
    if scorecard["influence_method"]:
        scorecard["overall_grade"] = scorecard["influence_method"]["grade"]

    return scorecard


def create_full_report(
    config: dict,
    dgp_spec: dict,
    raw_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    models: list = None,
    methods: list = None,
    timing: dict = None,
) -> str:
    """Generate comprehensive log file for AI analysis.

    Args:
        config: Simulation configuration dict
        dgp_spec: DGP specification with alpha_star, beta_star, mu_true
        raw_df: Raw per-simulation results DataFrame
        metrics_df: Aggregated metrics DataFrame
        models: List of model names
        methods: List of method names
        timing: Timing information dict

    Returns:
        JSON string containing the full report
    """
    # Convert raw_df to records, handling NaN values
    raw_records = []
    for _, row in raw_df.iterrows():
        record = {}
        for col in raw_df.columns:
            record[col] = _safe_float(row[col])
        raw_records.append(record)

    report = {
        "meta": {
            "generated": datetime.now().isoformat(),
            "version": "1.0",
            "framework": "FLM Influence Function Validation",
        },
        "config": {
            "M": config.get("M"),
            "N": config.get("N"),
            "n_folds": config.get("n_folds"),
            "epochs": config.get("epochs"),
            "hidden_dims": config.get("hidden_dims"),
            "lr": config.get("lr"),
            "batch_size": config.get("batch_size"),
            "dropout": config.get("dropout"),
            "weight_decay": config.get("weight_decay"),
            "seed": config.get("seed"),
            "early_stopping": config.get("early_stopping"),
            "patience": config.get("patience"),
            "val_split": config.get("val_split"),
        },
        "models": models or [],
        "methods": methods or [],
        "dgp": dgp_spec,
        "phase1_recovery": extract_phase1(metrics_df),
        "phase2_inference": extract_phase2(metrics_df),
        "phase3_diagnostics": extract_phase3(metrics_df),
        "training_quality": extract_training_quality(metrics_df),
        "validation_scorecard": compute_scorecard(metrics_df),
        "raw_data": raw_records,
        "timing": timing or {},
    }

    return json.dumps(report, indent=2, default=str)


def save_report(report: str, output_dir: str = "logs") -> str:
    """Save report to timestamped log file.

    Args:
        report: JSON string report
        output_dir: Directory to save report

    Returns:
        Path to saved report file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{output_dir}/mc_run_{timestamp}.log"
    with open(path, "w") as f:
        f.write(report)
    return path


def format_human_readable(report_json: str) -> str:
    """Format report as human-readable text with JSON sections."""
    report = json.loads(report_json)

    lines = []
    lines.append("=" * 80)
    lines.append("FLM VALIDATION REPORT")
    lines.append(f"Generated: {report['meta']['generated']}")
    lines.append("=" * 80)
    lines.append("")

    # Configuration
    lines.append("## CONFIGURATION")
    lines.append(json.dumps(report["config"], indent=2))
    lines.append("")

    # DGP
    lines.append("## DGP SPECIFICATION")
    lines.append(json.dumps(report["dgp"], indent=2))
    lines.append("")

    # Phase 1
    lines.append("## PHASE 1: PARAMETER RECOVERY")
    lines.append(json.dumps(report["phase1_recovery"], indent=2))
    lines.append("")

    # Phase 2
    lines.append("## PHASE 2: INFERENCE VALIDITY")
    lines.append(json.dumps(report["phase2_inference"], indent=2))
    lines.append("")

    # Phase 3
    lines.append("## PHASE 3: TRAINING DIAGNOSTICS")
    lines.append(json.dumps(report["phase3_diagnostics"], indent=2))
    lines.append("")

    # Training Quality
    if "training_quality" in report:
        lines.append("## TRAINING QUALITY (NN Model Fit)")
        lines.append(json.dumps(report["training_quality"], indent=2))
        lines.append("")

    # Scorecard
    lines.append("## VALIDATION SCORECARD")
    lines.append(json.dumps(report["validation_scorecard"], indent=2))
    lines.append("")

    # Timing
    if report.get("timing"):
        lines.append("## TIMING")
        lines.append(json.dumps(report["timing"], indent=2))
        lines.append("")

    # Raw data count
    lines.append(f"## RAW DATA: {len(report['raw_data'])} records")
    lines.append("")

    lines.append("=" * 80)
    lines.append("END REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)
