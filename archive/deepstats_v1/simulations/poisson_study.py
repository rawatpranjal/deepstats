"""Monte Carlo simulation study for DeepPoisson estimator.

This module provides tools for evaluating DeepPoisson inference via
Monte Carlo simulations, measuring:
- Bias and RMSE for E[lambda], Var[lambda], quantiles
- Coverage rates for 95% confidence intervals
- SE calibration (estimated vs empirical SE ratio)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy import stats


@dataclass
class PoissonSimulationResult:
    """Result from a single Poisson simulation run.

    Attributes
    ----------
    true_mean_lambda : float
        True E[lambda(X)].
    true_var_lambda : float
        True Var[lambda(X)].
    true_quantiles : dict[float, float]
        True quantiles.
    est_mean_lambda : float
        Estimated E[lambda(X)].
    est_mean_lambda_se : float
        Estimated SE for mean_lambda.
    est_var_lambda : float
        Estimated Var[lambda(X)].
    est_var_lambda_se : float
        Estimated SE for var_lambda.
    est_quantiles : dict[float, float]
        Estimated quantiles.
    est_quantile_se : dict[float, float]
        Estimated SE for quantiles.
    seed : int
        Random seed used.
    """

    # True values
    true_mean_lambda: float
    true_var_lambda: float
    true_quantiles: dict[float, float]

    # Estimates
    est_mean_lambda: float
    est_mean_lambda_se: float
    est_var_lambda: float
    est_var_lambda_se: float
    est_quantiles: dict[float, float]
    est_quantile_se: dict[float, float]

    # Diagnostics
    loss_history: list[float] = field(default_factory=list)
    seed: int = 0

    @property
    def mean_lambda_covered(self) -> bool:
        """Whether 95% CI for mean_lambda contains true value."""
        z = 1.96
        ci_lower = self.est_mean_lambda - z * self.est_mean_lambda_se
        ci_upper = self.est_mean_lambda + z * self.est_mean_lambda_se
        return ci_lower <= self.true_mean_lambda <= ci_upper

    @property
    def var_lambda_covered(self) -> bool:
        """Whether 95% CI for var_lambda contains true value."""
        z = 1.96
        ci_lower = self.est_var_lambda - z * self.est_var_lambda_se
        ci_upper = self.est_var_lambda + z * self.est_var_lambda_se
        return ci_lower <= self.true_var_lambda <= ci_upper

    def quantile_covered(self, q: float) -> bool:
        """Whether 95% CI for quantile q contains true value."""
        if q not in self.true_quantiles or q not in self.est_quantiles:
            return False
        z = 1.96
        est = self.est_quantiles[q]
        se = self.est_quantile_se.get(q, np.nan)
        if np.isnan(se):
            return False
        ci_lower = est - z * se
        ci_upper = est + z * se
        return ci_lower <= self.true_quantiles[q] <= ci_upper


@dataclass
class PoissonSimulationSummary:
    """Aggregated results from Poisson simulation study.

    Attributes
    ----------
    n_simulations : int
        Number of successful simulations.
    mean_lambda_bias : float
        Bias in E[lambda(X)] estimates.
    mean_lambda_rmse : float
        RMSE in E[lambda(X)] estimates.
    mean_lambda_coverage : float
        95% CI coverage rate for E[lambda(X)].
    mean_lambda_mean_se : float
        Average estimated SE.
    mean_lambda_empirical_se : float
        Empirical SD of estimates.
    mean_lambda_se_ratio : float
        mean_se / empirical_se (should be ~1.0).
    var_lambda_* : float
        Same metrics for Var[lambda(X)].
    quantile_* : dict[float, float]
        Same metrics per quantile.
    results : list[PoissonSimulationResult]
        Individual simulation results.
    """

    n_simulations: int

    # Mean lambda metrics
    mean_lambda_bias: float
    mean_lambda_rmse: float
    mean_lambda_coverage: float
    mean_lambda_mean_se: float
    mean_lambda_empirical_se: float
    mean_lambda_se_ratio: float

    # Var lambda metrics
    var_lambda_bias: float
    var_lambda_rmse: float
    var_lambda_coverage: float
    var_lambda_mean_se: float
    var_lambda_empirical_se: float
    var_lambda_se_ratio: float

    # Quantile metrics
    quantile_biases: dict[float, float]
    quantile_rmses: dict[float, float]
    quantile_coverages: dict[float, float]

    # All results
    results: list[PoissonSimulationResult]

    def summary(self) -> str:
        """Generate text summary of simulation results."""
        lines = []
        lines.append("=" * 70)
        lines.append("         Poisson Simulation Study Summary")
        lines.append("=" * 70)
        lines.append(f"Number of simulations: {self.n_simulations}")
        lines.append("")

        # Mean lambda
        lines.append("E[lambda(X)] - Average Rate Parameter")
        lines.append("-" * 50)
        lines.append(f"  Bias:            {self.mean_lambda_bias:>12.4f}")
        lines.append(f"  RMSE:            {self.mean_lambda_rmse:>12.4f}")
        lines.append(f"  Coverage (95%):  {self.mean_lambda_coverage:>12.1%}")
        lines.append(f"  Mean SE:         {self.mean_lambda_mean_se:>12.4f}")
        lines.append(f"  Empirical SE:    {self.mean_lambda_empirical_se:>12.4f}")
        lines.append(f"  SE Ratio:        {self.mean_lambda_se_ratio:>12.2f}")
        lines.append("")

        # Var lambda
        lines.append("Var[lambda(X)] - Rate Heterogeneity")
        lines.append("-" * 50)
        lines.append(f"  Bias:            {self.var_lambda_bias:>12.4f}")
        lines.append(f"  RMSE:            {self.var_lambda_rmse:>12.4f}")
        lines.append(f"  Coverage (95%):  {self.var_lambda_coverage:>12.1%}")
        lines.append(f"  Mean SE:         {self.var_lambda_mean_se:>12.4f}")
        lines.append(f"  Empirical SE:    {self.var_lambda_empirical_se:>12.4f}")
        lines.append(f"  SE Ratio:        {self.var_lambda_se_ratio:>12.2f}")
        lines.append("")

        # Quantiles
        if self.quantile_biases:
            lines.append("Quantiles of lambda(X)")
            lines.append("-" * 50)
            lines.append(f"  {'Quantile':>10}  {'Bias':>10}  {'RMSE':>10}  {'Coverage':>10}")
            for q in sorted(self.quantile_biases.keys()):
                lines.append(
                    f"  {q:>10.2f}  "
                    f"{self.quantile_biases[q]:>10.4f}  "
                    f"{self.quantile_rmses[q]:>10.4f}  "
                    f"{self.quantile_coverages[q]:>10.1%}"
                )
            lines.append("")

        lines.append("=" * 70)
        lines.append("Note: Coverage should be ~95%. SE Ratio should be ~1.0.")
        lines.append("=" * 70)

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PoissonSimulationSummary(n={self.n_simulations}, "
            f"mean_coverage={self.mean_lambda_coverage:.1%}, "
            f"var_coverage={self.var_lambda_coverage:.1%})"
        )


class PoissonSimulationStudy:
    """Run Monte Carlo simulations for DeepPoisson estimator.

    This class orchestrates simulation studies to evaluate the quality
    of inference for E[lambda(X)], Var[lambda(X)], and quantiles.

    Parameters
    ----------
    dgp : callable
        Data generating process. Takes seed as argument, returns
        PoissonSimulationData or dict with required fields.
    estimator_factory : callable
        Factory function that creates a new DeepPoisson instance.
    n_simulations : int, default=100
        Number of Monte Carlo simulations.
    random_state : int, optional
        Base random seed.
    verbose : int, default=1
        Verbosity level.

    Examples
    --------
    >>> from deepstats.simulations import PoissonSimulationStudy
    >>> from deepstats.simulations.dgp_poisson import make_poisson_dgp_lowdim
    >>> from deepstats.estimators import DeepPoisson
    >>>
    >>> def dgp(seed):
    ...     return make_poisson_dgp_lowdim(seed, n=2000, p=5)
    >>>
    >>> def make_estimator():
    ...     return DeepPoisson(
    ...         hidden_dims=[64, 32],
    ...         epochs=300,
    ...         cross_fit_folds=5,
    ...         verbose=0,
    ...     )
    >>>
    >>> study = PoissonSimulationStudy(dgp, make_estimator, n_simulations=100)
    >>> summary = study.run()
    >>> print(summary.summary())
    """

    def __init__(
        self,
        dgp: Callable[[int], Any],
        estimator_factory: Callable[[], Any],
        n_simulations: int = 100,
        random_state: int | None = None,
        verbose: int = 1,
    ):
        self.dgp = dgp
        self.estimator_factory = estimator_factory
        self.n_simulations = n_simulations
        self.random_state = random_state
        self.verbose = verbose

    def run(self) -> PoissonSimulationSummary:
        """Run all simulations and aggregate results.

        Returns
        -------
        PoissonSimulationSummary
            Aggregated results with bias, RMSE, coverage metrics.
        """
        results: list[PoissonSimulationResult] = []

        for i in range(self.n_simulations):
            seed = (self.random_state or 0) + i

            if self.verbose >= 1:
                print(f"\rSimulation {i+1}/{self.n_simulations}", end="", flush=True)

            try:
                result = self._run_single(seed)
                results.append(result)
            except Exception as e:
                if self.verbose >= 2:
                    print(f"\nSimulation {i+1} failed: {e}")

        if self.verbose >= 1:
            print()

        return self._aggregate_results(results)

    def _run_single(self, seed: int) -> PoissonSimulationResult:
        """Run a single simulation.

        Parameters
        ----------
        seed : int
            Random seed.

        Returns
        -------
        PoissonSimulationResult
            Result from this simulation.
        """
        # Generate data
        dgp_data = self.dgp(seed)

        # Handle dataclass or dict
        if hasattr(dgp_data, "X"):
            X = dgp_data.X
            y = dgp_data.y
            true_mean_lambda = dgp_data.true_mean_lambda
            true_var_lambda = dgp_data.true_var_lambda
            true_quantiles = dgp_data.true_quantiles
        else:
            X = dgp_data["X"]
            y = dgp_data["y"]
            true_mean_lambda = dgp_data["true_mean_lambda"]
            true_var_lambda = dgp_data["true_var_lambda"]
            true_quantiles = dgp_data["true_quantiles"]

        # Create and fit estimator
        estimator = self.estimator_factory()
        result = estimator.fit(X, y)

        return PoissonSimulationResult(
            true_mean_lambda=true_mean_lambda,
            true_var_lambda=true_var_lambda,
            true_quantiles=true_quantiles,
            est_mean_lambda=result.mean_lambda,
            est_mean_lambda_se=result.mean_lambda_se,
            est_var_lambda=result.var_lambda,
            est_var_lambda_se=result.var_lambda_se,
            est_quantiles=result.quantiles,
            est_quantile_se=result.quantile_se,
            loss_history=result.loss_history_,
            seed=seed,
        )

    def _aggregate_results(
        self,
        results: list[PoissonSimulationResult],
    ) -> PoissonSimulationSummary:
        """Aggregate individual simulation results.

        Parameters
        ----------
        results : list[PoissonSimulationResult]
            Individual results.

        Returns
        -------
        PoissonSimulationSummary
            Aggregated metrics.
        """
        n = len(results)
        if n == 0:
            raise ValueError("No successful simulations to aggregate")

        # Mean lambda metrics
        true_means = np.array([r.true_mean_lambda for r in results])
        est_means = np.array([r.est_mean_lambda for r in results])
        est_mean_ses = np.array([r.est_mean_lambda_se for r in results])

        mean_lambda_errors = est_means - true_means
        mean_lambda_bias = float(np.mean(mean_lambda_errors))
        mean_lambda_rmse = float(np.sqrt(np.mean(mean_lambda_errors**2)))
        mean_lambda_coverage = float(np.mean([r.mean_lambda_covered for r in results]))
        mean_lambda_mean_se = float(np.mean(est_mean_ses))
        mean_lambda_empirical_se = float(np.std(est_means, ddof=1))
        mean_lambda_se_ratio = (
            mean_lambda_mean_se / mean_lambda_empirical_se
            if mean_lambda_empirical_se > 1e-12
            else np.nan
        )

        # Var lambda metrics
        true_vars = np.array([r.true_var_lambda for r in results])
        est_vars = np.array([r.est_var_lambda for r in results])
        est_var_ses = np.array([r.est_var_lambda_se for r in results])

        var_lambda_errors = est_vars - true_vars
        var_lambda_bias = float(np.mean(var_lambda_errors))
        var_lambda_rmse = float(np.sqrt(np.mean(var_lambda_errors**2)))
        var_lambda_coverage = float(np.mean([r.var_lambda_covered for r in results]))
        var_lambda_mean_se = float(np.mean(est_var_ses))
        var_lambda_empirical_se = float(np.std(est_vars, ddof=1))
        var_lambda_se_ratio = (
            var_lambda_mean_se / var_lambda_empirical_se
            if var_lambda_empirical_se > 1e-12
            else np.nan
        )

        # Quantile metrics
        quantile_biases: dict[float, float] = {}
        quantile_rmses: dict[float, float] = {}
        quantile_coverages: dict[float, float] = {}

        if results[0].true_quantiles:
            quantiles = sorted(results[0].true_quantiles.keys())
            for q in quantiles:
                true_qs = np.array([r.true_quantiles[q] for r in results])
                est_qs = np.array([r.est_quantiles.get(q, np.nan) for r in results])
                valid = ~np.isnan(est_qs)
                if valid.sum() > 0:
                    errors = est_qs[valid] - true_qs[valid]
                    quantile_biases[q] = float(np.mean(errors))
                    quantile_rmses[q] = float(np.sqrt(np.mean(errors**2)))
                    coverages = [r.quantile_covered(q) for r in results]
                    quantile_coverages[q] = float(np.mean(coverages))

        return PoissonSimulationSummary(
            n_simulations=n,
            mean_lambda_bias=mean_lambda_bias,
            mean_lambda_rmse=mean_lambda_rmse,
            mean_lambda_coverage=mean_lambda_coverage,
            mean_lambda_mean_se=mean_lambda_mean_se,
            mean_lambda_empirical_se=mean_lambda_empirical_se,
            mean_lambda_se_ratio=mean_lambda_se_ratio,
            var_lambda_bias=var_lambda_bias,
            var_lambda_rmse=var_lambda_rmse,
            var_lambda_coverage=var_lambda_coverage,
            var_lambda_mean_se=var_lambda_mean_se,
            var_lambda_empirical_se=var_lambda_empirical_se,
            var_lambda_se_ratio=var_lambda_se_ratio,
            quantile_biases=quantile_biases,
            quantile_rmses=quantile_rmses,
            quantile_coverages=quantile_coverages,
            results=results,
        )
