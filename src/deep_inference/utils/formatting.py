"""Summary formatting utilities for statsmodels-style output."""

from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from scipy import stats


def compute_z_and_pvalue(estimate: float, se: float) -> Tuple[float, float]:
    """
    Compute z-statistic and two-sided p-value.

    Args:
        estimate: Point estimate
        se: Standard error

    Returns:
        (z_stat, p_value) tuple
    """
    if se <= 0 or np.isnan(se):
        return np.nan, np.nan

    z_stat = estimate / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return z_stat, p_value


def format_pvalue(p: float) -> str:
    """
    Format p-value for display.

    Args:
        p: p-value

    Returns:
        Formatted string (e.g., "0.000", "0.042", "<0.001")
    """
    if np.isnan(p):
        return "   nan"
    if p < 0.001:
        return "  0.000"
    return f"  {p:.3f}"


def format_summary_header(
    title: str,
    family: Optional[str] = None,
    target: Optional[str] = None,
    n_obs: Optional[int] = None,
    n_folds: Optional[int] = None,
    width: int = 78,
) -> str:
    """
    Format statsmodels-style header block.

    Args:
        title: Main title (e.g., "Structural DML Results")
        family: Model family name
        target: Target functional name
        n_obs: Number of observations
        n_folds: Number of cross-fitting folds
        width: Total width of output

    Returns:
        Formatted header string
    """
    lines = []
    sep = "=" * width

    lines.append(sep)
    lines.append(f"{title:^{width}}")
    lines.append(sep)

    # Get current date/time
    now = datetime.now()
    date_str = now.strftime("%a, %d %b %Y")
    time_str = now.strftime("%H:%M:%S")

    # Build two-column layout
    left_col = []
    right_col = []

    if family is not None:
        left_col.append(("Family:", family.capitalize()))
    if target is not None:
        right_col.append(("Target:", target))

    if n_obs is not None:
        left_col.append(("No. Observations:", str(n_obs)))
    if n_folds is not None:
        right_col.append(("No. Folds:", str(n_folds)))

    left_col.append(("Date:", date_str))
    right_col.append(("Time:", time_str))

    # Format columns
    half_width = width // 2
    for left, right in zip(left_col, right_col):
        left_str = f"{left[0]:<18}{left[1]:<{half_width - 18}}"
        right_str = f"{right[0]:<18}{right[1]}"
        lines.append(f"{left_str}{right_str}")

    # Handle unequal lengths
    if len(left_col) > len(right_col):
        for item in left_col[len(right_col):]:
            lines.append(f"{item[0]:<18}{item[1]}")
    elif len(right_col) > len(left_col):
        for item in right_col[len(left_col):]:
            lines.append(" " * half_width + f"{item[0]:<18}{item[1]}")

    lines.append(sep)

    return "\n".join(lines)


def format_coefficient_table(
    coef_name: str,
    estimate: float,
    se: float,
    ci_lower: float,
    ci_upper: float,
    width: int = 78,
) -> str:
    """
    Format coefficient table row with header.

    Args:
        coef_name: Name of coefficient (e.g., "E[beta]")
        estimate: Point estimate
        se: Standard error
        ci_lower: Lower confidence interval bound
        ci_upper: Upper confidence interval bound
        width: Total width of output

    Returns:
        Formatted table string
    """
    z_stat, p_value = compute_z_and_pvalue(estimate, se)

    lines = []
    sep = "-" * width

    # Header row
    header = f"{'':>12}{'coef':>10}{'std err':>12}{'z':>10}{'P>|z|':>10}{'[0.025':>12}{'0.975]':>10}"
    lines.append(header)
    lines.append(sep)

    # Data row
    p_str = format_pvalue(p_value)
    row = f"{coef_name:>12}{estimate:>10.4f}{se:>12.4f}{z_stat:>10.3f}{p_str}{ci_lower:>12.4f}{ci_upper:>10.4f}"
    lines.append(row)

    lines.append("=" * width)

    return "\n".join(lines)


def format_diagnostics_footer(
    diagnostics: Dict[str, Any],
    width: int = 78,
) -> str:
    """
    Format diagnostics section.

    Args:
        diagnostics: Dictionary of diagnostic values
        width: Total width of output

    Returns:
        Formatted diagnostics string
    """
    lines = []
    lines.append("Diagnostics:")

    # Select key diagnostics to display
    key_diagnostics = [
        ("min_lambda_eigenvalue", "Min Lambda eigenvalue", "{:.6f}"),
        ("mean_cond_number", "Mean condition number", "{:.2f}"),
        ("correction_ratio", "Correction ratio", "{:.4f}"),
        ("pct_regularized", "Pct regularized", "{:.1f}%"),
    ]

    for key, label, fmt in key_diagnostics:
        if key in diagnostics:
            value = diagnostics[key]
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                formatted_value = fmt.format(value) if isinstance(value, (int, float)) else str(value)
                lines.append(f"  {label + ':':<25} {formatted_value}")

    lines.append("-" * width)

    return "\n".join(lines)


def format_short_repr(
    class_name: str,
    estimate: float,
    se: float,
    ci_lower: float,
    ci_upper: float,
) -> str:
    """
    Format short __repr__ string.

    Args:
        class_name: Name of result class
        estimate: Point estimate
        se: Standard error
        ci_lower: Lower CI bound
        ci_upper: Upper CI bound

    Returns:
        Short repr string
    """
    return (
        f"<{class_name}: mu_hat={estimate:.4f}, se={se:.4f}, "
        f"95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]>"
    )


def format_full_summary(
    title: str,
    coef_name: str,
    estimate: float,
    se: float,
    ci_lower: float,
    ci_upper: float,
    diagnostics: Optional[Dict[str, Any]] = None,
    family: Optional[str] = None,
    target: Optional[str] = None,
    n_obs: Optional[int] = None,
    n_folds: Optional[int] = None,
    width: int = 78,
) -> str:
    """
    Format complete summary output.

    Args:
        title: Main title
        coef_name: Coefficient name for table
        estimate: Point estimate
        se: Standard error
        ci_lower: Lower CI bound
        ci_upper: Upper CI bound
        diagnostics: Optional diagnostics dict
        family: Model family name
        target: Target functional name
        n_obs: Number of observations
        n_folds: Number of folds
        width: Output width

    Returns:
        Complete formatted summary string
    """
    parts = []

    # Header
    parts.append(format_summary_header(
        title=title,
        family=family,
        target=target,
        n_obs=n_obs,
        n_folds=n_folds,
        width=width,
    ))

    # Coefficient table
    parts.append(format_coefficient_table(
        coef_name=coef_name,
        estimate=estimate,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        width=width,
    ))

    # Diagnostics footer (if provided)
    if diagnostics:
        parts.append(format_diagnostics_footer(diagnostics, width=width))

    return "\n".join(parts)
