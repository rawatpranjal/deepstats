"""Utility functions."""

from .linalg import safe_inverse, batch_inverse
from .formatting import (
    compute_z_and_pvalue,
    format_pvalue,
    format_summary_header,
    format_coefficient_table,
    format_diagnostics_footer,
    format_short_repr,
    format_full_summary,
)
from .result_mixin import PredictVisualizeMixin

__all__ = [
    "safe_inverse",
    "batch_inverse",
    "compute_z_and_pvalue",
    "format_pvalue",
    "format_summary_header",
    "format_coefficient_table",
    "format_diagnostics_footer",
    "format_short_repr",
    "format_full_summary",
    "PredictVisualizeMixin",
]
