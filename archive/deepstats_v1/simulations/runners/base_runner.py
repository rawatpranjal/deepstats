"""Base simulation runner class.

Provides common functionality for running simulation studies
across different data modalities.
"""

from __future__ import annotations

import json
import pickle
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class SimulationConfig:
    """Configuration for simulation study.

    Attributes
    ----------
    n_reps : int
        Number of Monte Carlo replications.
    n_samples : int
        Number of samples per replication.
    seed : int
        Base random seed.
    methods : list[str]
        Methods to compare.
    quantiles : list[float]
        Quantiles to estimate.
    validation_split : float
        Proportion of data for validation.
    output_dir : str
        Directory for saving results.
    save_results : bool
        Whether to save results to disk.
    verbose : int
        Verbosity level.
    """

    n_reps: int = 20
    n_samples: int = 2000
    seed: int = 42
    methods: list[str] = field(
        default_factory=lambda: ["deephte", "causal_forest", "linear_dml"]
    )
    quantiles: list[float] = field(
        default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9]
    )
    validation_split: float = 0.2
    output_dir: str = "simulations/results"
    save_results: bool = True
    verbose: int = 1


@dataclass
class MethodResult:
    """Results from a single method on a single replication.

    Attributes
    ----------
    method : str
        Method name.
    ate_estimate : float
        Estimated ATE.
    ate_se : float
        Standard error of ATE.
    ite_estimates : np.ndarray
        Estimated ITEs.
    quantile_estimates : dict[float, float]
        Estimated quantiles.
    fit_time : float
        Time to fit model in seconds.
    """

    method: str
    ate_estimate: float
    ate_se: float
    ite_estimates: np.ndarray
    quantile_estimates: dict[float, float] = field(default_factory=dict)
    fit_time: float = 0.0


@dataclass
class ReplicationResult:
    """Results from a single replication.

    Attributes
    ----------
    rep_id : int
        Replication ID.
    seed : int
        Random seed used.
    true_ate : float
        True ATE.
    true_ite : np.ndarray
        True ITEs.
    method_results : dict[str, MethodResult]
        Results by method.
    """

    rep_id: int
    seed: int
    true_ate: float
    true_ite: np.ndarray
    method_results: dict[str, MethodResult] = field(default_factory=dict)


@dataclass
class SimulationResults:
    """Aggregated results from simulation study.

    Attributes
    ----------
    config : SimulationConfig
        Configuration used.
    modality : str
        Data modality.
    dgp_name : str
        DGP scenario name.
    replications : list[ReplicationResult]
        Results per replication.
    start_time : str
        Start time of simulation.
    end_time : str
        End time of simulation.
    """

    config: SimulationConfig
    modality: str
    dgp_name: str
    replications: list[ReplicationResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        records = []
        for rep in self.replications:
            for method, result in rep.method_results.items():
                ate_bias = result.ate_estimate - rep.true_ate

                # CI calculations
                z_crit = 1.96
                ci_lower = result.ate_estimate - z_crit * result.ate_se
                ci_upper = result.ate_estimate + z_crit * result.ate_se
                ate_covered = (
                    (ci_lower <= rep.true_ate <= ci_upper)
                    if result.ate_se > 0
                    else False
                )
                ci_width = ci_upper - ci_lower if result.ate_se > 0 else np.nan
                rejects_null = (ci_lower > 0) or (ci_upper < 0)

                # Relative bias
                ate_relative_bias = (
                    ate_bias / abs(rep.true_ate) if abs(rep.true_ate) > 1e-10 else np.nan
                )

                # ITE metrics
                ite_rmse = float(
                    np.sqrt(np.mean((rep.true_ite - result.ite_estimates) ** 2))
                )
                ite_mae = float(np.mean(np.abs(rep.true_ite - result.ite_estimates)))
                ite_corr = float(
                    np.corrcoef(rep.true_ite, result.ite_estimates)[0, 1]
                ) if len(rep.true_ite) == len(result.ite_estimates) else np.nan

                record = {
                    "rep_id": rep.rep_id,
                    "seed": rep.seed,
                    "method": method,
                    "true_ate": rep.true_ate,
                    "ate_estimate": result.ate_estimate,
                    "ate_se": result.ate_se,
                    "ate_bias": ate_bias,
                    "ate_relative_bias": ate_relative_bias,
                    "ate_covered": ate_covered,
                    "ci_width": ci_width,
                    "rejects_null": rejects_null,
                    "ite_rmse": ite_rmse,
                    "ite_mae": ite_mae,
                    "ite_correlation": ite_corr,
                    "fit_time": result.fit_time,
                }

                # Add quantile estimates and per-quantile metrics
                for q, val in result.quantile_estimates.items():
                    record[f"q{int(q*100):02d}"] = val
                    # Compute true quantile for this replication
                    true_q = float(np.quantile(rep.true_ite, q))
                    record[f"q{int(q*100):02d}_true"] = true_q
                    record[f"q{int(q*100):02d}_bias"] = val - true_q

                records.append(record)

        return pd.DataFrame(records)

    def summary_by_method(self) -> pd.DataFrame:
        """Compute summary statistics by method."""
        df = self.to_dataframe()
        summary = df.groupby("method").agg({
            "ate_bias": ["mean", "std"],
            "ate_relative_bias": "mean",
            "ate_se": "mean",
            "ate_covered": "mean",
            "ci_width": "mean",
            "rejects_null": "mean",
            "ite_rmse": ["mean", "std"],
            "ite_mae": "mean",
            "ite_correlation": "mean",
            "fit_time": "mean",
        })
        summary.columns = [
            "bias_mean", "bias_std",
            "relative_bias",
            "se_mean",
            "coverage",
            "ci_width",
            "rejection_rate",
            "ite_rmse_mean", "ite_rmse_std",
            "ite_mae",
            "ite_corr",
            "time_mean",
        ]

        # Add empirical SE and SE ratio
        empirical_se = df.groupby("method")["ate_estimate"].std()
        summary["empirical_se"] = empirical_se
        summary["se_ratio"] = summary["se_mean"] / summary["empirical_se"]

        # Compute MAE from bias values
        summary["mae"] = df.groupby("method")["ate_bias"].apply(
            lambda x: np.mean(np.abs(x))
        )

        return summary

    def save(self, path: str | Path) -> None:
        """Save results to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save DataFrame
        df = self.to_dataframe()
        df.to_csv(path.with_suffix(".csv"), index=False)

        # Save full results as pickle
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self, f)

        # Save config as JSON
        config_dict = {
            "modality": self.modality,
            "dgp_name": self.dgp_name,
            "n_reps": self.config.n_reps,
            "n_samples": self.config.n_samples,
            "seed": self.config.seed,
            "methods": self.config.methods,
            "quantiles": self.config.quantiles,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SimulationResults":
        """Load results from disk."""
        path = Path(path)
        with open(path.with_suffix(".pkl"), "rb") as f:
            return pickle.load(f)


class BaseSimulationRunner(ABC):
    """Abstract base class for simulation runners.

    Subclasses must implement:
    - _create_estimator: Create estimator for a given method
    - _run_method: Run a method on data and return results
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    @property
    @abstractmethod
    def modality(self) -> str:
        """Return the data modality (tabular, image, text, graph)."""
        pass

    @abstractmethod
    def _generate_data(self, dgp_name: str, seed: int) -> dict:
        """Generate data for a single replication."""
        pass

    @abstractmethod
    def _run_method(
        self, method: str, data: dict, seed: int
    ) -> MethodResult:
        """Run a single method and return results."""
        pass

    def run(self, dgp_name: str) -> SimulationResults:
        """Run full simulation study for a DGP.

        Parameters
        ----------
        dgp_name : str
            Name of DGP scenario.

        Returns
        -------
        SimulationResults
            Full simulation results.
        """
        results = SimulationResults(
            config=self.config,
            modality=self.modality,
            dgp_name=dgp_name,
            start_time=datetime.now().isoformat(),
        )

        for rep in range(self.config.n_reps):
            seed = self.config.seed + rep

            if self.config.verbose >= 1:
                print(
                    f"\r[{self.modality}/{dgp_name}] "
                    f"Rep {rep + 1}/{self.config.n_reps}",
                    end="",
                    flush=True,
                )

            try:
                rep_result = self._run_replication(dgp_name, rep, seed)
                results.replications.append(rep_result)
            except Exception as e:
                if self.config.verbose >= 2:
                    print(f"\nRep {rep + 1} failed: {e}")
                warnings.warn(f"Replication {rep + 1} failed: {e}")

        if self.config.verbose >= 1:
            print()

        results.end_time = datetime.now().isoformat()

        if self.config.save_results:
            output_path = Path(self.config.output_dir) / self.modality / dgp_name
            results.save(output_path)

        return results

    def _run_replication(
        self, dgp_name: str, rep_id: int, seed: int
    ) -> ReplicationResult:
        """Run a single replication."""
        # Generate data
        data = self._generate_data(dgp_name, seed)

        rep_result = ReplicationResult(
            rep_id=rep_id,
            seed=seed,
            true_ate=data["true_ate"],
            true_ite=data["true_ite"],
        )

        # Run each method
        for method in self.config.methods:
            try:
                method_result = self._run_method(method, data, seed)
                rep_result.method_results[method] = method_result
            except Exception as e:
                if self.config.verbose >= 2:
                    print(f"\n  Method {method} failed: {e}")
                warnings.warn(f"Method {method} failed on rep {rep_id}: {e}")

        return rep_result

    def run_all_dgps(self, dgp_names: list[str]) -> dict[str, SimulationResults]:
        """Run simulations for multiple DGPs.

        Parameters
        ----------
        dgp_names : list[str]
            List of DGP names to run.

        Returns
        -------
        dict[str, SimulationResults]
            Results by DGP name.
        """
        all_results = {}
        for dgp_name in dgp_names:
            if self.config.verbose >= 1:
                print(f"\n=== Running {self.modality}/{dgp_name} ===")
            all_results[dgp_name] = self.run(dgp_name)
        return all_results
