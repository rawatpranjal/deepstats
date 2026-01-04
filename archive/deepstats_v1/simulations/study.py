"""Simulation study framework for HTE estimators.

This module provides the main SimulationStudy class for running
Monte Carlo simulations to evaluate heterogeneous treatment effect
estimators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any
import warnings

import numpy as np
import pandas as pd

from .metrics import (
    ATEMetrics,
    ITEMetrics,
    QuantileMetrics,
    compute_ate_metrics,
    compute_ite_metrics,
    compute_quantile_metrics,
)
from .diagnostics import (
    FittingDiagnosis,
    analyze_loss_curves,
    LossCurveAnalysis,
)


@dataclass
class SimulationResult:
    """Result from a single simulation run.

    Attributes
    ----------
    ate_true : float
        True average treatment effect.
    ate_estimate : float
        Estimated ATE.
    ate_se : float
        Estimated standard error of ATE.
    ite_true : np.ndarray
        True individual treatment effects.
    ite_estimate : np.ndarray
        Estimated ITEs.
    train_loss : list[float]
        Training loss at each epoch.
    val_loss : list[float]
        Validation loss at each epoch.
    fitting_diagnosis : FittingDiagnosis
        Diagnosis of fitting behavior.
    loss_analysis : LossCurveAnalysis
        Detailed loss curve analysis.
    quantile_estimates : dict[float, float]
        Estimated quantiles of ITE distribution.
    seed : int
        Random seed used for this simulation.
    """

    ate_true: float
    ate_estimate: float
    ate_se: float
    ite_true: np.ndarray
    ite_estimate: np.ndarray
    train_loss: list[float]
    val_loss: list[float]
    fitting_diagnosis: FittingDiagnosis
    loss_analysis: LossCurveAnalysis
    quantile_estimates: dict[float, float] = field(default_factory=dict)
    seed: int = 0

    @property
    def ate_bias(self) -> float:
        """Bias of ATE estimate."""
        return self.ate_estimate - self.ate_true

    @property
    def ate_covered(self) -> bool:
        """Whether 95% CI contains true ATE."""
        z = 1.96
        ci_lower = self.ate_estimate - z * self.ate_se
        ci_upper = self.ate_estimate + z * self.ate_se
        return ci_lower <= self.ate_true <= ci_upper

    @property
    def ite_rmse(self) -> float:
        """RMSE of ITE estimates."""
        return float(np.sqrt(np.mean((self.ite_true - self.ite_estimate) ** 2)))

    @property
    def ite_correlation(self) -> float:
        """Correlation between true and estimated ITEs."""
        if np.std(self.ite_true) == 0 or np.std(self.ite_estimate) == 0:
            return np.nan
        return float(np.corrcoef(self.ite_true, self.ite_estimate)[0, 1])


@dataclass
class SimulationSummary:
    """Aggregated results from simulation study.

    Attributes
    ----------
    n_simulations : int
        Number of simulations run.
    ate_metrics : ATEMetrics
        Aggregate ATE metrics.
    ite_metrics : ITEMetrics
        Aggregate ITE metrics.
    quantile_metrics : QuantileMetrics
        Aggregate quantile metrics.
    fitting_distribution : dict[FittingDiagnosis, int]
        Count of each fitting diagnosis.
    results : list[SimulationResult]
        Individual simulation results.
    """

    n_simulations: int
    ate_metrics: ATEMetrics
    ite_metrics: ITEMetrics
    quantile_metrics: QuantileMetrics
    fitting_distribution: dict[FittingDiagnosis, int]
    results: list[SimulationResult]

    def summary(self) -> str:
        """Generate text summary of simulation results."""
        lines = [
            "=" * 60,
            "Simulation Study Summary",
            "=" * 60,
            f"Number of simulations: {self.n_simulations}",
            "",
            "ATE Estimation",
            "-" * 40,
            f"  Bias:         {self.ate_metrics.bias:>10.4f}",
            f"  Rel. Bias:    {self.ate_metrics.relative_bias:>10.4f}",
            f"  RMSE:         {self.ate_metrics.rmse:>10.4f}",
            f"  MAE:          {self.ate_metrics.mae:>10.4f}",
            f"  Coverage:     {self.ate_metrics.coverage:>10.1%}",
            f"  CI Width:     {self.ate_metrics.ci_width:>10.4f}",
            f"  Mean SE:      {self.ate_metrics.mean_se:>10.4f}",
            f"  Empirical SE: {self.ate_metrics.empirical_se:>10.4f}",
            f"  SE Ratio:     {self.ate_metrics.se_ratio:>10.2f}",
        ]

        # Add power or type1_error based on which is available
        if not np.isnan(self.ate_metrics.power):
            lines.append(f"  Power:        {self.ate_metrics.power:>10.1%}")
        if not np.isnan(self.ate_metrics.type1_error):
            lines.append(f"  Type I Err:   {self.ate_metrics.type1_error:>10.1%}")

        lines.extend([
            "",
            "ITE Estimation",
            "-" * 40,
            f"  RMSE:         {self.ite_metrics.rmse:>10.4f}",
            f"  MAE:          {self.ite_metrics.mae:>10.4f}",
            f"  Bias:         {self.ite_metrics.bias:>10.4f}",
            f"  Rel. RMSE:    {self.ite_metrics.relative_rmse:>10.4f}",
            f"  Rel. Bias:    {self.ite_metrics.relative_bias:>10.4f}",
            f"  Correlation:  {self.ite_metrics.correlation:>10.4f}",
            f"  Rank Corr:    {self.ite_metrics.rank_correlation:>10.4f}",
            f"  Empirical SE: {self.ite_metrics.empirical_se:>10.4f}",
            f"  KS Statistic: {self.ite_metrics.ks_statistic:>10.4f}",
            f"  Wasserstein:  {self.ite_metrics.wasserstein_distance:>10.4f}",
        ])

        # Add coverage and calibration if available
        if not np.isnan(self.ite_metrics.coverage):
            lines.append(f"  Coverage:     {self.ite_metrics.coverage:>10.1%}")
        if not np.isnan(self.ite_metrics.mean_se):
            lines.append(f"  Mean SE:      {self.ite_metrics.mean_se:>10.4f}")
        if not np.isnan(self.ite_metrics.calibration_ratio):
            lines.append(f"  Calib. Ratio: {self.ite_metrics.calibration_ratio:>10.2f}")

        lines.extend([
            "",
            "Quantile Estimation",
            "-" * 40,
        ])

        for q in self.quantile_metrics.quantiles:
            bias = self.quantile_metrics.biases[q]
            rmse = self.quantile_metrics.rmses[q]
            mae = self.quantile_metrics.maes.get(q, np.nan)
            rel_bias = self.quantile_metrics.relative_biases.get(q, np.nan)
            lines.append(
                f"  Q{int(q*100):02d}:  bias={bias:>8.4f}  rmse={rmse:>8.4f}  "
                f"mae={mae:>8.4f}  rel_bias={rel_bias:>8.4f}"
            )

        lines.extend([
            "",
            "Fitting Diagnosis Distribution",
            "-" * 40,
        ])

        for diagnosis, count in self.fitting_distribution.items():
            pct = count / self.n_simulations * 100
            lines.append(f"  {diagnosis.value:<25} {count:>4} ({pct:>5.1f}%)")

        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        records = []
        for r in self.results:
            # Compute CI bounds
            z = 1.96
            ci_lower = r.ate_estimate - z * r.ate_se
            ci_upper = r.ate_estimate + z * r.ate_se
            ci_width = ci_upper - ci_lower
            rejects_null = (ci_lower > 0) or (ci_upper < 0)

            # Compute relative bias
            ate_relative_bias = (
                r.ate_bias / abs(r.ate_true) if abs(r.ate_true) > 1e-10 else np.nan
            )

            records.append({
                "ate_true": r.ate_true,
                "ate_estimate": r.ate_estimate,
                "ate_se": r.ate_se,
                "ate_bias": r.ate_bias,
                "ate_relative_bias": ate_relative_bias,
                "ate_covered": r.ate_covered,
                "ci_width": ci_width,
                "rejects_null": rejects_null,
                "ite_rmse": r.ite_rmse,
                "ite_correlation": r.ite_correlation,
                "fitting_diagnosis": r.fitting_diagnosis.value,
                "train_loss_final": r.train_loss[-1] if r.train_loss else np.nan,
                "val_loss_final": r.val_loss[-1] if r.val_loss else np.nan,
                "seed": r.seed,
            })
        return pd.DataFrame(records)


class SimulationStudy:
    """Run Monte Carlo simulations for HTE estimators.

    Parameters
    ----------
    dgp : callable
        Data generating process. Should return a dict or object with:
        - data: pd.DataFrame with Y, T, X columns
        - true_ate: float
        - true_ite: np.ndarray
    estimator_factory : callable
        Factory function that creates a new estimator instance.
    n_simulations : int, default=100
        Number of Monte Carlo simulations.
    validation_split : float, default=0.2
        Proportion of data for validation.
    quantiles : tuple, default=(0.1, 0.25, 0.5, 0.75, 0.9)
        Quantiles to evaluate.
    random_state : int, optional
        Base random seed.
    verbose : int, default=1
        Verbosity level (0=silent, 1=progress, 2=detailed).

    Examples
    --------
    >>> from deepstats.simulations import SimulationStudy
    >>> from deepstats.datasets.ab_test import make_ab_test
    >>> from deepstats import DeepHTE
    >>>
    >>> def dgp(seed):
    ...     return make_ab_test(n=1000, seed=seed)
    >>>
    >>> def make_estimator():
    ...     return DeepHTE(
    ...         formula="Y ~ a(X1+X2+X3) + b(X1+X2+X3) * T",
    ...         epochs=100,
    ...         verbose=0,
    ...     )
    >>>
    >>> study = SimulationStudy(dgp, make_estimator, n_simulations=50)
    >>> summary = study.run()
    >>> print(summary.summary())
    """

    def __init__(
        self,
        dgp: Callable[[int], Any],
        estimator_factory: Callable[[], Any],
        n_simulations: int = 100,
        validation_split: float = 0.2,
        quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
        random_state: int | None = None,
        verbose: int = 1,
    ):
        self.dgp = dgp
        self.estimator_factory = estimator_factory
        self.n_simulations = n_simulations
        self.validation_split = validation_split
        self.quantiles = quantiles
        self.random_state = random_state
        self.verbose = verbose

    def run(self) -> SimulationSummary:
        """Run all simulations and aggregate results.

        Returns
        -------
        SimulationSummary
            Aggregated simulation results.
        """
        results = []

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
                warnings.warn(f"Simulation {i+1} failed: {e}")

        if self.verbose >= 1:
            print()  # New line after progress

        return self._aggregate_results(results)

    def _run_single(self, seed: int) -> SimulationResult:
        """Run a single simulation.

        Parameters
        ----------
        seed : int
            Random seed for this simulation.

        Returns
        -------
        SimulationResult
            Result from this simulation.
        """
        # Generate data
        dgp_result = self.dgp(seed)

        # Handle different DGP return types
        if hasattr(dgp_result, "data"):
            data = dgp_result.data
            true_ate = dgp_result.true_ate
            true_ite = dgp_result.true_ite
        else:
            data = dgp_result["data"]
            true_ate = dgp_result["true_ate"]
            true_ite = dgp_result["true_ite"]

        # Split data for validation
        n = len(data)
        rng = np.random.default_rng(seed)
        val_idx = rng.random(n) < self.validation_split
        train_data = data[~val_idx].reset_index(drop=True)
        val_data = data[val_idx].reset_index(drop=True)
        true_ite_train = true_ite[~val_idx]

        # Create and fit estimator
        estimator = self.estimator_factory()

        # Check if estimator supports validation data
        if hasattr(estimator, "fit") and "validation_data" in estimator.fit.__code__.co_varnames:
            result = estimator.fit(train_data, validation_data=val_data)
        else:
            result = estimator.fit(train_data)

        # Extract results
        ate_estimate = result.ate
        ate_se = result.ate_se
        ite_estimate = result.ite

        # Get loss histories if available
        train_loss = getattr(result, "train_loss_history", [])
        val_loss = getattr(result, "val_loss_history", [])

        # If no validation loss, use training loss as proxy
        if not val_loss and train_loss:
            val_loss = train_loss.copy()

        # Analyze fitting
        if train_loss and val_loss:
            loss_analysis = analyze_loss_curves(train_loss, val_loss)
            fitting_diagnosis = loss_analysis.diagnosis
        else:
            loss_analysis = LossCurveAnalysis(
                diagnosis=FittingDiagnosis.INCONCLUSIVE,
                train_final=np.nan,
                val_final=np.nan,
                train_min=np.nan,
                val_min=np.nan,
                generalization_gap=np.nan,
                relative_gap=np.nan,
                val_increasing=False,
                optimal_epoch=0,
                convergence_epoch=0,
                explanation="No loss history available.",
            )
            fitting_diagnosis = FittingDiagnosis.INCONCLUSIVE

        # Quantile estimates
        quantile_estimates = {q: float(np.quantile(ite_estimate, q)) for q in self.quantiles}

        return SimulationResult(
            ate_true=true_ate,
            ate_estimate=ate_estimate,
            ate_se=ate_se,
            ite_true=true_ite_train,
            ite_estimate=ite_estimate,
            train_loss=train_loss,
            val_loss=val_loss,
            fitting_diagnosis=fitting_diagnosis,
            loss_analysis=loss_analysis,
            quantile_estimates=quantile_estimates,
            seed=seed,
        )

    def _aggregate_results(self, results: list[SimulationResult]) -> SimulationSummary:
        """Aggregate individual simulation results.

        Parameters
        ----------
        results : list[SimulationResult]
            Individual simulation results.

        Returns
        -------
        SimulationSummary
            Aggregated results.
        """
        if not results:
            raise ValueError("No successful simulations to aggregate.")

        # Extract arrays
        true_ate = results[0].ate_true  # Assume same DGP
        estimated_ates = [r.ate_estimate for r in results]
        estimated_ses = [r.ate_se for r in results]
        true_ites = [r.ite_true for r in results]
        estimated_ites = [r.ite_estimate for r in results]

        # Compute metrics
        ate_metrics = compute_ate_metrics(true_ate, estimated_ates, estimated_ses)
        ite_metrics = compute_ite_metrics(true_ites, estimated_ites)
        quantile_metrics = compute_quantile_metrics(
            true_ites, estimated_ites, self.quantiles
        )

        # Fitting distribution
        fitting_distribution = {}
        for diagnosis in FittingDiagnosis:
            fitting_distribution[diagnosis] = sum(
                1 for r in results if r.fitting_diagnosis == diagnosis
            )

        return SimulationSummary(
            n_simulations=len(results),
            ate_metrics=ate_metrics,
            ite_metrics=ite_metrics,
            quantile_metrics=quantile_metrics,
            fitting_distribution=fitting_distribution,
            results=results,
        )
