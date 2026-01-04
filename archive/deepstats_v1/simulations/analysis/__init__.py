"""Analysis modules for simulation results."""

from .ate_analysis import compute_ate_distribution, ATEDistribution
from .coverage_analysis import compute_coverage, CoverageResults
from .quantile_analysis import compute_quantile_metrics, QuantileMetrics
from .figures import generate_all_figures, FigureGenerator

__all__ = [
    "compute_ate_distribution",
    "ATEDistribution",
    "compute_coverage",
    "CoverageResults",
    "compute_quantile_metrics",
    "QuantileMetrics",
    "generate_all_figures",
    "FigureGenerator",
]
