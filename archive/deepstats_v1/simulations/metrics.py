"""Evaluation metrics for simulation studies.

This module provides metrics for evaluating HTE estimators:
- ATE bias, RMSE, and coverage
- ITE RMSE and correlation
- Quantile accuracy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass
class ATEMetrics:
    """Metrics for Average Treatment Effect estimation."""

    bias: float
    rmse: float
    coverage: float  # 95% CI coverage rate
    mean_se: float  # Mean estimated standard error
    empirical_se: float  # Empirical standard deviation of estimates
    se_ratio: float  # mean_se / empirical_se (should be ~1)

    # New metrics (with defaults for backward compatibility)
    relative_bias: float = np.nan  # bias / |true_ate| (scale-free)
    mae: float = np.nan  # Mean Absolute Error across replications
    ci_width: float = np.nan  # Average 95% CI width
    power: float = np.nan  # Rejection rate of H0:ATE=0 when effect is non-zero
    type1_error: float = np.nan  # Rejection rate when true_ate ≈ 0


@dataclass
class ITEMetrics:
    """Metrics for Individual Treatment Effect estimation."""

    rmse: float
    mae: float
    correlation: float
    rank_correlation: float  # Spearman correlation

    # New metrics (with defaults for backward compatibility)
    bias: float = np.nan  # Mean bias of ITE estimates
    relative_rmse: float = np.nan  # RMSE / |mean(true_ite)|
    relative_bias: float = np.nan  # bias / |mean(true_ite)|
    coverage: float = np.nan  # Proportion of true ITEs within CIs (requires SEs)
    mean_se: float = np.nan  # Mean estimated SE across ITEs
    empirical_se: float = np.nan  # Empirical SE of ITE errors
    calibration_ratio: float = np.nan  # mean_se / empirical_se
    ks_statistic: float = np.nan  # Kolmogorov-Smirnov statistic
    ks_pvalue: float = np.nan  # KS test p-value
    wasserstein_distance: float = np.nan  # Wasserstein-1 distance


@dataclass
class QuantileMetrics:
    """Metrics for quantile estimation accuracy."""

    quantiles: list[float]
    biases: dict[float, float]
    rmses: dict[float, float]
    coverages: dict[float, float]

    # New metrics (with defaults for backward compatibility)
    mean_ses: dict[float, float] = field(default_factory=dict)  # Mean estimated SEs
    empirical_ses: dict[float, float] = field(default_factory=dict)  # Empirical SEs
    se_ratios: dict[float, float] = field(default_factory=dict)  # SE calibration
    relative_biases: dict[float, float] = field(default_factory=dict)  # Relative bias
    ci_widths: dict[float, float] = field(default_factory=dict)  # CI widths
    maes: dict[float, float] = field(default_factory=dict)  # MAE per quantile


def compute_ate_metrics(
    true_ate: float,
    estimated_ates: Sequence[float],
    estimated_ses: Sequence[float],
    alpha: float = 0.05,
    null_threshold: float = 0.01,
) -> ATEMetrics:
    """Compute ATE estimation metrics.

    Parameters
    ----------
    true_ate : float
        True average treatment effect.
    estimated_ates : array-like
        Estimated ATEs from each simulation.
    estimated_ses : array-like
        Estimated standard errors from each simulation.
    alpha : float, default=0.05
        Significance level for coverage calculation.
    null_threshold : float, default=0.01
        If |true_ate| < null_threshold, compute type1_error instead of power.

    Returns
    -------
    ATEMetrics
        Container with ATE evaluation metrics.
    """
    ates = np.asarray(estimated_ates)
    ses = np.asarray(estimated_ses)

    # Bias
    bias = float(np.mean(ates) - true_ate)

    # RMSE
    rmse = float(np.sqrt(np.mean((ates - true_ate) ** 2)))

    # Coverage: proportion of CIs containing true ATE
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ates - z_crit * ses
    ci_upper = ates + z_crit * ses
    covered = (ci_lower <= true_ate) & (true_ate <= ci_upper)
    coverage = float(np.mean(covered))

    # Standard error analysis
    mean_se = float(np.mean(ses))
    empirical_se = float(np.std(ates, ddof=1))
    se_ratio = mean_se / empirical_se if empirical_se > 0 else np.nan

    # Relative bias (scale-free)
    relative_bias = bias / abs(true_ate) if abs(true_ate) > 1e-10 else np.nan

    # MAE across replications
    mae = float(np.mean(np.abs(ates - true_ate)))

    # Average CI width
    ci_width = float(np.mean(ci_upper - ci_lower))

    # Power or Type I Error
    # Reject H0 if CI does not contain 0
    rejects_null = (ci_lower > 0) | (ci_upper < 0)

    if abs(true_ate) < null_threshold:
        # True effect is approximately zero - this is Type I error
        type1_error = float(np.mean(rejects_null))
        power = np.nan
    else:
        # True effect is non-zero - this is power
        power = float(np.mean(rejects_null))
        type1_error = np.nan

    return ATEMetrics(
        bias=bias,
        rmse=rmse,
        coverage=coverage,
        mean_se=mean_se,
        empirical_se=empirical_se,
        se_ratio=se_ratio,
        relative_bias=relative_bias,
        mae=mae,
        ci_width=ci_width,
        power=power,
        type1_error=type1_error,
    )


def compute_ite_metrics(
    true_ites: Sequence[np.ndarray],
    estimated_ites: Sequence[np.ndarray],
    estimated_ses: Sequence[np.ndarray] | None = None,
    alpha: float = 0.05,
) -> ITEMetrics:
    """Compute ITE estimation metrics.

    Parameters
    ----------
    true_ites : list of arrays
        True ITEs from each simulation.
    estimated_ites : list of arrays
        Estimated ITEs from each simulation.
    estimated_ses : list of arrays, optional
        Estimated standard errors of ITEs for each simulation.
        Required for coverage and calibration metrics.
    alpha : float, default=0.05
        Significance level for coverage calculation.

    Returns
    -------
    ITEMetrics
        Container with ITE evaluation metrics.
    """
    rmses = []
    maes = []
    biases = []
    correlations = []
    rank_correlations = []
    ks_statistics = []
    ks_pvalues = []
    wasserstein_distances = []

    # For coverage and calibration
    all_errors = []
    all_ses = []
    coverage_counts = []
    z_crit = stats.norm.ppf(1 - alpha / 2)

    for i, (true, est) in enumerate(zip(true_ites, estimated_ites)):
        true = np.asarray(true)
        est = np.asarray(est)

        # RMSE
        rmses.append(np.sqrt(np.mean((true - est) ** 2)))

        # MAE
        maes.append(np.mean(np.abs(true - est)))

        # Bias
        biases.append(np.mean(est - true))

        # Pearson correlation
        if np.std(true) > 0 and np.std(est) > 0:
            correlations.append(np.corrcoef(true, est)[0, 1])
        else:
            correlations.append(np.nan)

        # Spearman rank correlation
        if len(true) > 2:
            rho, _ = stats.spearmanr(true, est)
            rank_correlations.append(rho)
        else:
            rank_correlations.append(np.nan)

        # KS statistic (distribution match)
        if len(true) > 1 and len(est) > 1:
            ks_stat, ks_pval = stats.ks_2samp(true, est)
            ks_statistics.append(ks_stat)
            ks_pvalues.append(ks_pval)
        else:
            ks_statistics.append(np.nan)
            ks_pvalues.append(np.nan)

        # Wasserstein distance
        if len(true) > 0 and len(est) > 0:
            w_dist = stats.wasserstein_distance(true, est)
            wasserstein_distances.append(w_dist)
        else:
            wasserstein_distances.append(np.nan)

        # Collect errors for empirical SE
        all_errors.extend(est - true)

        # Coverage if SEs provided
        if estimated_ses is not None and i < len(estimated_ses):
            ses = np.asarray(estimated_ses[i])
            if len(ses) == len(est):
                all_ses.extend(ses)
                ci_lower = est - z_crit * ses
                ci_upper = est + z_crit * ses
                covered = (ci_lower <= true) & (true <= ci_upper)
                coverage_counts.extend(covered)

    # Aggregate base metrics
    rmse = float(np.nanmean(rmses))
    mae = float(np.nanmean(maes))
    bias = float(np.nanmean(biases))
    correlation = float(np.nanmean(correlations))
    rank_correlation = float(np.nanmean(rank_correlations))

    # Distribution metrics
    ks_statistic = float(np.nanmean(ks_statistics))
    ks_pvalue = float(np.nanmean(ks_pvalues))
    wasserstein_distance = float(np.nanmean(wasserstein_distances))

    # Relative metrics (scale-free)
    all_true = np.concatenate([np.asarray(t) for t in true_ites])
    mean_true = np.mean(all_true)
    relative_rmse = rmse / abs(mean_true) if abs(mean_true) > 1e-10 else np.nan
    relative_bias = bias / abs(mean_true) if abs(mean_true) > 1e-10 else np.nan

    # Empirical SE of errors
    empirical_se = float(np.std(all_errors, ddof=1)) if len(all_errors) > 1 else np.nan

    # Coverage and calibration (only if SEs provided)
    if estimated_ses is not None and len(coverage_counts) > 0:
        coverage = float(np.mean(coverage_counts))
        mean_se = float(np.mean(all_ses))
        calibration_ratio = mean_se / empirical_se if empirical_se > 0 else np.nan
    else:
        coverage = np.nan
        mean_se = np.nan
        calibration_ratio = np.nan

    return ITEMetrics(
        rmse=rmse,
        mae=mae,
        correlation=correlation,
        rank_correlation=rank_correlation,
        bias=bias,
        relative_rmse=relative_rmse,
        relative_bias=relative_bias,
        coverage=coverage,
        mean_se=mean_se,
        empirical_se=empirical_se,
        calibration_ratio=calibration_ratio,
        ks_statistic=ks_statistic,
        ks_pvalue=ks_pvalue,
        wasserstein_distance=wasserstein_distance,
    )


def compute_quantile_metrics(
    true_ites: Sequence[np.ndarray],
    estimated_ites: Sequence[np.ndarray],
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
    bootstrap_ses: Sequence[dict[float, float]] | None = None,
    alpha: float = 0.05,
) -> QuantileMetrics:
    """Compute quantile estimation metrics.

    Parameters
    ----------
    true_ites : list of arrays
        True ITEs from each simulation.
    estimated_ites : list of arrays
        Estimated ITEs from each simulation.
    quantiles : tuple of floats
        Quantiles to evaluate.
    bootstrap_ses : list of dicts, optional
        Bootstrap standard errors for each quantile, per simulation.
    alpha : float, default=0.05
        Significance level for coverage.

    Returns
    -------
    QuantileMetrics
        Container with quantile evaluation metrics.
    """
    quantiles = list(quantiles)
    biases = {q: [] for q in quantiles}
    errors_sq = {q: [] for q in quantiles}
    abs_errors = {q: [] for q in quantiles}  # For MAE
    coverages = {q: [] for q in quantiles}
    ci_widths_list = {q: [] for q in quantiles}  # For CI width
    estimated_quantiles = {q: [] for q in quantiles}  # For empirical SE
    true_quantiles_list = {q: [] for q in quantiles}  # For relative bias

    z_crit = stats.norm.ppf(1 - alpha / 2)

    for i, (true, est) in enumerate(zip(true_ites, estimated_ites)):
        true = np.asarray(true)
        est = np.asarray(est)

        for q in quantiles:
            true_q = np.quantile(true, q)
            est_q = np.quantile(est, q)

            biases[q].append(est_q - true_q)
            errors_sq[q].append((est_q - true_q) ** 2)
            abs_errors[q].append(abs(est_q - true_q))
            estimated_quantiles[q].append(est_q)
            true_quantiles_list[q].append(true_q)

            # Coverage and CI width if SEs provided
            if bootstrap_ses is not None and i < len(bootstrap_ses):
                se = bootstrap_ses[i].get(q, np.nan)
                if not np.isnan(se):
                    ci_lower = est_q - z_crit * se
                    ci_upper = est_q + z_crit * se
                    coverages[q].append(int(ci_lower <= true_q <= ci_upper))
                    ci_widths_list[q].append(ci_upper - ci_lower)

    # Aggregate existing metrics
    bias_dict = {q: float(np.mean(biases[q])) for q in quantiles}
    rmse_dict = {q: float(np.sqrt(np.mean(errors_sq[q]))) for q in quantiles}
    coverage_dict = {
        q: float(np.mean(coverages[q])) if coverages[q] else np.nan for q in quantiles
    }

    # NEW: MAE per quantile
    mae_dict = {q: float(np.mean(abs_errors[q])) for q in quantiles}

    # NEW: Relative bias (bias / |mean true quantile|)
    relative_bias_dict = {}
    for q in quantiles:
        mean_true_q = np.mean(true_quantiles_list[q])
        relative_bias_dict[q] = (
            bias_dict[q] / abs(mean_true_q) if abs(mean_true_q) > 1e-10 else np.nan
        )

    # NEW: Empirical SE of quantile estimates
    empirical_se_dict = {
        q: float(np.std(estimated_quantiles[q], ddof=1))
        if len(estimated_quantiles[q]) > 1
        else np.nan
        for q in quantiles
    }

    # NEW: Mean SE, SE ratio, and CI width (only if bootstrap_ses provided)
    mean_se_dict = {}
    se_ratio_dict = {}
    ci_width_dict = {}

    if bootstrap_ses is not None:
        for q in quantiles:
            ses_for_q = [bs.get(q, np.nan) for bs in bootstrap_ses if q in bs]
            valid_ses = [s for s in ses_for_q if not np.isnan(s)]
            mean_se_dict[q] = float(np.mean(valid_ses)) if valid_ses else np.nan

            if empirical_se_dict[q] > 0 and not np.isnan(mean_se_dict.get(q, np.nan)):
                se_ratio_dict[q] = mean_se_dict[q] / empirical_se_dict[q]
            else:
                se_ratio_dict[q] = np.nan

            ci_width_dict[q] = (
                float(np.mean(ci_widths_list[q])) if ci_widths_list[q] else np.nan
            )

    return QuantileMetrics(
        quantiles=quantiles,
        biases=bias_dict,
        rmses=rmse_dict,
        coverages=coverage_dict,
        mean_ses=mean_se_dict,
        empirical_ses=empirical_se_dict,
        se_ratios=se_ratio_dict,
        relative_biases=relative_bias_dict,
        ci_widths=ci_width_dict,
        maes=mae_dict,
    )


def compute_simulation_metrics(
    true_ate: float,
    estimated_ates: Sequence[float],
    estimated_ses: Sequence[float],
    true_ites: Sequence[np.ndarray],
    estimated_ites: Sequence[np.ndarray],
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
    bootstrap_ses: Sequence[dict[float, float]] | None = None,
) -> dict:
    """Compute all simulation metrics.

    Parameters
    ----------
    true_ate : float
        True average treatment effect.
    estimated_ates : array-like
        Estimated ATEs from each simulation.
    estimated_ses : array-like
        Estimated standard errors from each simulation.
    true_ites : list of arrays
        True ITEs from each simulation.
    estimated_ites : list of arrays
        Estimated ITEs from each simulation.
    quantiles : tuple of floats
        Quantiles to evaluate.
    bootstrap_ses : list of dicts, optional
        Bootstrap standard errors for quantiles.

    Returns
    -------
    dict
        Dictionary with all metrics.
    """
    ate_metrics = compute_ate_metrics(true_ate, estimated_ates, estimated_ses)
    ite_metrics = compute_ite_metrics(true_ites, estimated_ites)
    quantile_metrics = compute_quantile_metrics(
        true_ites, estimated_ites, quantiles, bootstrap_ses
    )

    return {
        "ate": ate_metrics,
        "ite": ite_metrics,
        "quantile": quantile_metrics,
    }
