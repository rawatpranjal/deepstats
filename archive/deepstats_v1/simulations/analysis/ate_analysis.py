"""ATE distribution analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ATEDistribution:
    """Distribution of ATE estimates.

    Attributes
    ----------
    estimates : np.ndarray
        ATE estimates across replications.
    true_ate : float
        True ATE value.
    bias : float
        Mean bias of estimates.
    rmse : float
        Root mean squared error.
    empirical_se : float
        Standard deviation of estimates.
    mean_se : float
        Mean of estimated standard errors.
    se_ratio : float
        Ratio of mean_se to empirical_se (calibration).
    relative_bias : float
        Bias divided by |true_ate| (scale-free).
    mae : float
        Mean Absolute Error.
    ci_width : float
        Average confidence interval width.
    power : float
        Rejection rate when true effect is non-zero.
    type1_error : float
        Rejection rate when true effect is approximately zero.
    """

    estimates: np.ndarray
    true_ate: float
    bias: float
    rmse: float
    empirical_se: float
    mean_se: float
    se_ratio: float

    # New metrics (with defaults for backward compatibility)
    relative_bias: float = np.nan
    mae: float = np.nan
    ci_width: float = np.nan
    power: float = np.nan
    type1_error: float = np.nan

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "ATE Distribution Summary",
            "------------------------",
            f"True ATE:      {self.true_ate:>8.4f}",
            f"Mean Estimate: {np.mean(self.estimates):>8.4f}",
            f"Bias:          {self.bias:>8.4f}",
            f"Rel. Bias:     {self.relative_bias:>8.4f}",
            f"RMSE:          {self.rmse:>8.4f}",
            f"MAE:           {self.mae:>8.4f}",
            f"Empirical SE:  {self.empirical_se:>8.4f}",
            f"Mean SE:       {self.mean_se:>8.4f}",
            f"SE Ratio:      {self.se_ratio:>8.2f}",
            f"CI Width:      {self.ci_width:>8.4f}",
        ]

        if not np.isnan(self.power):
            lines.append(f"Power:         {self.power:>8.1%}")
        if not np.isnan(self.type1_error):
            lines.append(f"Type I Error:  {self.type1_error:>8.1%}")

        return "\n".join(lines) + "\n"


def compute_ate_distribution(
    results_df: pd.DataFrame,
    method: str,
    null_threshold: float = 0.01,
) -> ATEDistribution:
    """Compute ATE distribution for a method.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame from SimulationResults.to_dataframe().
    method : str
        Method name.
    null_threshold : float, default=0.01
        If |true_ate| < null_threshold, compute type1_error instead of power.

    Returns
    -------
    ATEDistribution
        Distribution analysis.
    """
    method_df = results_df[results_df["method"] == method]

    estimates = method_df["ate_estimate"].values
    true_ate = method_df["true_ate"].iloc[0]
    ses = method_df["ate_se"].values

    bias = float(np.mean(estimates) - true_ate)
    rmse = float(np.sqrt(np.mean((estimates - true_ate) ** 2)))
    empirical_se = float(np.std(estimates))
    mean_se = float(np.mean(ses))
    se_ratio = mean_se / empirical_se if empirical_se > 0 else np.nan

    # NEW: Relative bias
    relative_bias = bias / abs(true_ate) if abs(true_ate) > 1e-10 else np.nan

    # NEW: MAE
    mae = float(np.mean(np.abs(estimates - true_ate)))

    # NEW: CI width
    z_crit = 1.96
    ci_widths = 2 * z_crit * ses
    ci_width = float(np.mean(ci_widths))

    # NEW: Power / Type I error
    ci_lower = estimates - z_crit * ses
    ci_upper = estimates + z_crit * ses
    rejects_null = (ci_lower > 0) | (ci_upper < 0)

    if abs(true_ate) < null_threshold:
        type1_error = float(np.mean(rejects_null))
        power = np.nan
    else:
        power = float(np.mean(rejects_null))
        type1_error = np.nan

    return ATEDistribution(
        estimates=estimates,
        true_ate=true_ate,
        bias=bias,
        rmse=rmse,
        empirical_se=empirical_se,
        mean_se=mean_se,
        se_ratio=se_ratio,
        relative_bias=relative_bias,
        mae=mae,
        ci_width=ci_width,
        power=power,
        type1_error=type1_error,
    )


def compare_ate_distributions(
    results_df: pd.DataFrame, methods: list[str] | None = None
) -> pd.DataFrame:
    """Compare ATE distributions across methods.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame.
    methods : list[str], optional
        Methods to compare. Default: all methods.

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    if methods is None:
        methods = results_df["method"].unique().tolist()

    records = []
    for method in methods:
        dist = compute_ate_distribution(results_df, method)
        records.append({
            "method": method,
            "bias": dist.bias,
            "relative_bias": dist.relative_bias,
            "rmse": dist.rmse,
            "mae": dist.mae,
            "empirical_se": dist.empirical_se,
            "mean_se": dist.mean_se,
            "se_ratio": dist.se_ratio,
            "ci_width": dist.ci_width,
            "power": dist.power,
            "type1_error": dist.type1_error,
        })

    return pd.DataFrame(records).set_index("method")
