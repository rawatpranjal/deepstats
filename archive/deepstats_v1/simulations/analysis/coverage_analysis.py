"""Coverage analysis for simulation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CoverageResults:
    """Coverage analysis results.

    Attributes
    ----------
    ate_coverage : float
        Coverage rate for ATE (should be ~0.95).
    ate_coverage_by_method : dict[str, float]
        Coverage by method.
    quantile_coverage : dict[float, float]
        Coverage by quantile.
    n_replications : int
        Number of replications.
    """

    ate_coverage: float
    ate_coverage_by_method: dict[str, float] = field(default_factory=dict)
    quantile_coverage: dict[float, float] = field(default_factory=dict)
    n_replications: int = 0

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "Coverage Analysis",
            "=" * 40,
            f"Replications: {self.n_replications}",
            f"Overall ATE Coverage: {self.ate_coverage:.1%}",
            "",
            "ATE Coverage by Method:",
        ]

        for method, cov in self.ate_coverage_by_method.items():
            status = "OK" if 0.90 <= cov <= 0.98 else "WARN"
            lines.append(f"  {method:20s}: {cov:>6.1%} [{status}]")

        if self.quantile_coverage:
            lines.extend(["", "Quantile Coverage:"])
            for q, cov in sorted(self.quantile_coverage.items()):
                lines.append(f"  Q{int(q*100):02d}: {cov:>6.1%}")

        return "\n".join(lines)


def compute_coverage(
    results_df: pd.DataFrame, alpha: float = 0.05
) -> CoverageResults:
    """Compute coverage rates from simulation results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame with ate_covered column.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    CoverageResults
        Coverage analysis.
    """
    # Overall coverage
    ate_coverage = float(results_df["ate_covered"].mean())

    # Coverage by method
    coverage_by_method = (
        results_df.groupby("method")["ate_covered"].mean().to_dict()
    )

    # Quantile coverage (if quantile columns exist)
    quantile_cols = [c for c in results_df.columns if c.startswith("q")]
    quantile_coverage = {}

    # Note: True quantile coverage requires true quantile values
    # which would need to be added to the results

    return CoverageResults(
        ate_coverage=ate_coverage,
        ate_coverage_by_method=coverage_by_method,
        quantile_coverage=quantile_coverage,
        n_replications=len(results_df) // len(coverage_by_method),
    )


def compute_coverage_intervals(
    results_df: pd.DataFrame, method: str, alpha: float = 0.05
) -> pd.DataFrame:
    """Compute confidence intervals for coverage rates.

    Uses Wilson score interval for proportions.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame.
    method : str
        Method name.
    alpha : float
        Significance level.

    Returns
    -------
    pd.DataFrame
        Coverage with confidence intervals.
    """
    from scipy import stats

    method_df = results_df[results_df["method"] == method]
    n = len(method_df)
    coverage = method_df["ate_covered"].mean()

    # Wilson score interval
    z = stats.norm.ppf(1 - alpha / 2)
    denominator = 1 + z**2 / n
    center = (coverage + z**2 / (2 * n)) / denominator
    spread = z * np.sqrt((coverage * (1 - coverage) + z**2 / (4 * n)) / n) / denominator

    return pd.DataFrame({
        "method": [method],
        "coverage": [coverage],
        "ci_lower": [center - spread],
        "ci_upper": [center + spread],
        "n": [n],
    })
