"""Standard error validation utilities.

This module provides tools for comparing and validating standard error
estimates across different methods (analytical, bootstrap, influence function,
jackknife).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from .._typing import Float64Array

if TYPE_CHECKING:
    from ..results.deep_results import DeepResults


@dataclass
class SEValidationResult:
    """Results from SE validation across methods.

    Attributes
    ----------
    analytical_se : ndarray
        Analytical (sandwich) standard errors.
    bootstrap_se : ndarray, optional
        Bootstrap standard errors.
    influence_se : ndarray, optional
        Influence function standard errors.
    jackknife_se : ndarray, optional
        Jackknife standard errors.
    bootstrap_ratio : ndarray, optional
        Ratio of bootstrap to analytical SE.
    influence_ratio : ndarray, optional
        Ratio of influence to analytical SE.
    jackknife_ratio : ndarray, optional
        Ratio of jackknife to analytical SE.
    max_ratio_deviation : float
        Maximum deviation of any ratio from 1.
    agreement_status : str
        Quality of agreement: "excellent", "good", "acceptable", "poor".
    """

    analytical_se: Float64Array
    bootstrap_se: Float64Array | None = None
    influence_se: Float64Array | None = None
    jackknife_se: Float64Array | None = None
    bootstrap_ratio: Float64Array | None = None
    influence_ratio: Float64Array | None = None
    jackknife_ratio: Float64Array | None = None
    max_ratio_deviation: float = 0.0
    agreement_status: Literal["excellent", "good", "acceptable", "poor"] = "good"

    def summary(self) -> str:
        """Generate comparison summary table.

        Returns
        -------
        str
            Formatted summary.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("        Standard Error Validation Results")
        lines.append("=" * 60)
        lines.append(f"Agreement Status: {self.agreement_status.upper()}")
        lines.append(f"Max Ratio Deviation: {self.max_ratio_deviation:.3f}")
        lines.append("-" * 60)
        lines.append("")

        # Table header
        headers = ["Param", "Analytical"]
        if self.bootstrap_se is not None:
            headers.append("Bootstrap")
        if self.influence_se is not None:
            headers.append("Influence")
        if self.jackknife_se is not None:
            headers.append("Jackknife")

        # Format header
        header_line = "  ".join(f"{h:>12}" for h in headers)
        lines.append(header_line)
        lines.append("-" * 60)

        # Table rows
        for i in range(len(self.analytical_se)):
            row = [f"x{i}", f"{self.analytical_se[i]:.4f}"]
            if self.bootstrap_se is not None:
                row.append(f"{self.bootstrap_se[i]:.4f}")
            if self.influence_se is not None:
                row.append(f"{self.influence_se[i]:.4f}")
            if self.jackknife_se is not None:
                row.append(f"{self.jackknife_se[i]:.4f}")
            lines.append("  ".join(f"{r:>12}" for r in row))

        lines.append("-" * 60)

        # Ratios section
        lines.append("")
        lines.append("Ratios (method / analytical):")
        if self.bootstrap_ratio is not None:
            lines.append(f"  Bootstrap: {np.mean(self.bootstrap_ratio):.3f} (mean), "
                        f"[{np.min(self.bootstrap_ratio):.3f}, {np.max(self.bootstrap_ratio):.3f}]")
        if self.influence_ratio is not None:
            lines.append(f"  Influence: {np.mean(self.influence_ratio):.3f} (mean), "
                        f"[{np.min(self.influence_ratio):.3f}, {np.max(self.influence_ratio):.3f}]")
        if self.jackknife_ratio is not None:
            lines.append(f"  Jackknife: {np.mean(self.jackknife_ratio):.3f} (mean), "
                        f"[{np.min(self.jackknife_ratio):.3f}, {np.max(self.jackknife_ratio):.3f}]")

        lines.append("=" * 60)

        return "\n".join(lines)


def validate_standard_errors(
    results: "DeepResults",
    methods: list[str] | None = None,
    bootstrap_type: str = "pairs",
    n_bootstrap: int = 500,
    random_state: int | None = None,
) -> SEValidationResult:
    """Compare standard errors across multiple methods.

    Parameters
    ----------
    results : DeepResults
        Fitted model results containing all necessary data.
    methods : list, optional
        Which methods to compare. Default is ["bootstrap", "influence"].
        Options: "bootstrap", "influence", "jackknife".
    bootstrap_type : str, default="pairs"
        Type of bootstrap if "bootstrap" in methods.
    n_bootstrap : int, default=500
        Number of bootstrap replications.
    random_state : int, optional
        Random seed.

    Returns
    -------
    SEValidationResult
        Comparison results with diagnostics.

    Notes
    -----
    Agreement thresholds:
    - excellent: all ratios within [0.9, 1.1]
    - good: all ratios within [0.8, 1.2]
    - acceptable: all ratios within [0.7, 1.3]
    - poor: ratios outside [0.7, 1.3]
    """
    if methods is None:
        methods = ["bootstrap", "influence"]

    analytical_se = results.std_errors.copy()

    bootstrap_se = None
    influence_se = None
    jackknife_se = None

    if "bootstrap" in methods and results.network_ is not None:
        from .bootstrap import bootstrap_pairs, bootstrap_wild, create_nn_fit_function

        fit_fn = create_nn_fit_function(
            results.network_,
            epochs=50,
            compute_marginal_effects=True,
        )

        if bootstrap_type == "pairs":
            boot_result = bootstrap_pairs(
                results.X_,
                results.y_,
                fit_fn,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
        else:
            boot_result = bootstrap_wild(
                results.X_,
                results.y_,
                results.fitted_values,
                results.residuals,
                fit_fn,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
                distribution=bootstrap_type.replace("wild_", ""),
            )

        bootstrap_se = boot_result.se

    if "influence" in methods and results.network_ is not None:
        from .influence import compute_influence_function_se

        inf_result = compute_influence_function_se(
            results.X_,
            results.y_,
            results.network_,
            results.fitted_values,
            results.residuals,
            cross_fit=True,
            n_folds=5,
            random_state=random_state,
        )
        influence_se = inf_result.se

    if "jackknife" in methods:
        from .jackknife import delete_d_jackknife_se

        # Use delete-d for efficiency
        from .bootstrap import create_nn_fit_function

        if results.network_ is not None:
            fit_fn = create_nn_fit_function(
                results.network_,
                epochs=30,
                compute_marginal_effects=True,
            )

            jk_result = delete_d_jackknife_se(
                results.X_,
                results.y_,
                fit_fn,
                results.params,
                d=max(10, len(results.y_) // 50),
                random_state=random_state,
            )
            jackknife_se = jk_result.se

    # Compute ratios
    bootstrap_ratio = bootstrap_se / analytical_se if bootstrap_se is not None else None
    influence_ratio = influence_se / analytical_se if influence_se is not None else None
    jackknife_ratio = jackknife_se / analytical_se if jackknife_se is not None else None

    # Compute max deviation
    ratios = []
    if bootstrap_ratio is not None:
        ratios.extend(np.abs(bootstrap_ratio - 1))
    if influence_ratio is not None:
        ratios.extend(np.abs(influence_ratio - 1))
    if jackknife_ratio is not None:
        ratios.extend(np.abs(jackknife_ratio - 1))

    max_deviation = max(ratios) if ratios else 0.0

    # Determine agreement status
    if max_deviation <= 0.1:
        status = "excellent"
    elif max_deviation <= 0.2:
        status = "good"
    elif max_deviation <= 0.3:
        status = "acceptable"
    else:
        status = "poor"

    return SEValidationResult(
        analytical_se=analytical_se,
        bootstrap_se=bootstrap_se,
        influence_se=influence_se,
        jackknife_se=jackknife_se,
        bootstrap_ratio=bootstrap_ratio,
        influence_ratio=influence_ratio,
        jackknife_ratio=jackknife_ratio,
        max_ratio_deviation=max_deviation,
        agreement_status=status,
    )


def monte_carlo_se_comparison(
    dgp_fn,
    true_params: Float64Array,
    estimator,
    n_replications: int = 100,
    methods: list[str] | None = None,
    random_state: int | None = None,
) -> dict:
    """Monte Carlo comparison of SE methods.

    Runs multiple replications to compare:
    1. Empirical SD of estimates (gold standard)
    2. Mean of analytical SEs
    3. Mean of bootstrap SEs
    4. Mean of influence function SEs

    Parameters
    ----------
    dgp_fn : callable
        Function that generates (X, y) from the DGP.
        Signature: dgp_fn(seed) -> (X, y)
    true_params : ndarray
        True parameter values.
    estimator : estimator
        Estimator object with fit() method.
    n_replications : int, default=100
        Number of Monte Carlo replications.
    methods : list, optional
        SE methods to compare.
    random_state : int, optional
        Base random seed.

    Returns
    -------
    dict
        Monte Carlo comparison results including:
        - empirical_sd: Empirical standard deviation of estimates
        - mean_analytical_se: Average analytical SE across replications
        - coverage_rates: CI coverage rates for each method
        - bias: Mean bias for each coefficient
    """
    if methods is None:
        methods = ["analytical"]

    rng = np.random.default_rng(random_state)
    p = len(true_params)

    estimates = []
    analytical_ses = []
    bootstrap_ses = [] if "bootstrap" in methods else None
    influence_ses = [] if "influence" in methods else None

    for rep in range(n_replications):
        seed = rng.integers(0, 2**31)
        X, y = dgp_fn(seed)

        result = estimator.fit(X, y)
        estimates.append(result.params)
        analytical_ses.append(result.std_errors)

        if "bootstrap" in methods:
            from .bootstrap import bootstrap_pairs, create_nn_fit_function

            fit_fn = create_nn_fit_function(result.network_, epochs=30)
            boot_result = bootstrap_pairs(X, y, fit_fn, n_bootstrap=100, random_state=seed)
            bootstrap_ses.append(boot_result.se)

        if "influence" in methods:
            from .influence import compute_influence_function_se

            inf_result = compute_influence_function_se(
                X, y, result.network_, result.fitted_values, result.residuals
            )
            influence_ses.append(inf_result.se)

    estimates = np.array(estimates)
    analytical_ses = np.array(analytical_ses)

    # Empirical SD (gold standard)
    empirical_sd = np.std(estimates, axis=0, ddof=1)

    # Mean SEs
    mean_analytical = np.mean(analytical_ses, axis=0)

    # Bias
    mean_estimate = np.mean(estimates, axis=0)
    bias = mean_estimate - true_params

    # Coverage rates
    def compute_coverage(ses, alpha=0.05):
        from scipy import stats
        z = stats.norm.ppf(1 - alpha / 2)
        lower = estimates - z * ses
        upper = estimates + z * ses
        covered = (lower <= true_params) & (true_params <= upper)
        return np.mean(covered, axis=0)

    results = {
        "empirical_sd": empirical_sd,
        "mean_analytical_se": mean_analytical,
        "bias": bias,
        "mean_estimate": mean_estimate,
        "true_params": true_params,
        "analytical_coverage": compute_coverage(analytical_ses),
    }

    if bootstrap_ses:
        bootstrap_ses = np.array(bootstrap_ses)
        results["mean_bootstrap_se"] = np.mean(bootstrap_ses, axis=0)
        results["bootstrap_coverage"] = compute_coverage(bootstrap_ses)

    if influence_ses:
        influence_ses = np.array(influence_ses)
        results["mean_influence_se"] = np.mean(influence_ses, axis=0)
        results["influence_coverage"] = compute_coverage(influence_ses)

    return results
