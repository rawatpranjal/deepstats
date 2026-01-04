"""Quantile estimation analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class QuantileMetrics:
    """Metrics for quantile estimation.

    Attributes
    ----------
    quantiles : list[float]
        Quantiles evaluated.
    biases : dict[float, float]
        Bias by quantile.
    rmses : dict[float, float]
        RMSE by quantile.
    coverages : dict[float, float]
        Coverage by quantile.
    mean_ses : dict[float, float]
        Mean estimated SE by quantile.
    empirical_ses : dict[float, float]
        Empirical SE by quantile.
    se_ratios : dict[float, float]
        SE ratio by quantile (calibration).
    relative_biases : dict[float, float]
        Relative bias by quantile.
    ci_widths : dict[float, float]
        CI width by quantile.
    maes : dict[float, float]
        MAE by quantile.
    """

    quantiles: list[float]
    biases: dict[float, float] = field(default_factory=dict)
    rmses: dict[float, float] = field(default_factory=dict)
    coverages: dict[float, float] = field(default_factory=dict)

    # New metrics
    mean_ses: dict[float, float] = field(default_factory=dict)
    empirical_ses: dict[float, float] = field(default_factory=dict)
    se_ratios: dict[float, float] = field(default_factory=dict)
    relative_biases: dict[float, float] = field(default_factory=dict)
    ci_widths: dict[float, float] = field(default_factory=dict)
    maes: dict[float, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate text summary."""
        lines = ["Quantile Estimation Metrics", "=" * 60]

        for q in self.quantiles:
            bias = self.biases.get(q, np.nan)
            rmse = self.rmses.get(q, np.nan)
            mae = self.maes.get(q, np.nan)
            rel_bias = self.relative_biases.get(q, np.nan)
            emp_se = self.empirical_ses.get(q, np.nan)
            lines.append(
                f"Q{int(q*100):02d}: "
                f"bias={bias:>7.3f}, "
                f"rel_bias={rel_bias:>7.3f}, "
                f"rmse={rmse:>7.3f}, "
                f"mae={mae:>7.3f}, "
                f"emp_se={emp_se:>7.3f}"
            )

        return "\n".join(lines)


def compute_quantile_metrics(
    results_df: pd.DataFrame,
    true_quantiles: dict[float, float],
    method: str,
) -> QuantileMetrics:
    """Compute quantile estimation metrics.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame with quantile columns.
    true_quantiles : dict[float, float]
        True quantile values.
    method : str
        Method name.

    Returns
    -------
    QuantileMetrics
        Quantile analysis.
    """
    method_df = results_df[results_df["method"] == method]

    quantiles = list(true_quantiles.keys())
    biases = {}
    rmses = {}
    maes = {}
    relative_biases = {}
    empirical_ses = {}

    for q in quantiles:
        col = f"q{int(q*100):02d}"
        if col in method_df.columns:
            estimates = method_df[col].values
            true_val = true_quantiles[q]

            bias = float(np.mean(estimates) - true_val)
            biases[q] = bias
            rmses[q] = float(np.sqrt(np.mean((estimates - true_val) ** 2)))
            maes[q] = float(np.mean(np.abs(estimates - true_val)))
            relative_biases[q] = bias / abs(true_val) if abs(true_val) > 1e-10 else np.nan
            empirical_ses[q] = float(np.std(estimates, ddof=1)) if len(estimates) > 1 else np.nan

    return QuantileMetrics(
        quantiles=quantiles,
        biases=biases,
        rmses=rmses,
        maes=maes,
        relative_biases=relative_biases,
        empirical_ses=empirical_ses,
    )


def compare_quantile_methods(
    results_df: pd.DataFrame,
    true_quantiles: dict[float, float],
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """Compare quantile estimation across methods.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame.
    true_quantiles : dict[float, float]
        True quantile values.
    methods : list[str], optional
        Methods to compare.

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    if methods is None:
        methods = results_df["method"].unique().tolist()

    records = []
    for method in methods:
        metrics = compute_quantile_metrics(results_df, true_quantiles, method)
        for q in metrics.quantiles:
            records.append({
                "method": method,
                "quantile": q,
                "bias": metrics.biases.get(q, np.nan),
                "relative_bias": metrics.relative_biases.get(q, np.nan),
                "rmse": metrics.rmses.get(q, np.nan),
                "mae": metrics.maes.get(q, np.nan),
                "empirical_se": metrics.empirical_ses.get(q, np.nan),
            })

    return pd.DataFrame(records)
