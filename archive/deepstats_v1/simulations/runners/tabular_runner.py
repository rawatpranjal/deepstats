"""Tabular simulation runner."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .base_runner import (
    BaseSimulationRunner,
    SimulationConfig,
    MethodResult,
)
from simulations.dgp import get_tabular_dgp


class TabularSimulationRunner(BaseSimulationRunner):
    """Simulation runner for tabular data.

    Runs DeepHTE, CausalForest, and LinearDML on tabular DGPs.
    """

    @property
    def modality(self) -> str:
        return "tabular"

    def _generate_data(self, dgp_name: str, seed: int) -> dict:
        """Generate tabular data."""
        dgp = get_tabular_dgp(dgp_name)
        return dgp.generate(seed)

    def _run_method(
        self, method: str, data: dict, seed: int
    ) -> MethodResult:
        """Run a method on tabular data."""
        df = data["data"]

        start_time = time.time()

        if method == "deephte":
            result = self._run_deephte(df, seed)
        elif method == "causal_forest":
            result = self._run_causal_forest(df, seed)
        elif method == "linear_dml":
            result = self._run_linear_dml(df, seed)
        elif method == "quantile_forest":
            result = self._run_quantile_forest(df, seed)
        else:
            raise ValueError(f"Unknown method: {method}")

        fit_time = time.time() - start_time

        # Compute quantiles
        quantile_estimates = {
            q: float(np.quantile(result["ite"], q))
            for q in self.config.quantiles
        }

        return MethodResult(
            method=method,
            ate_estimate=result["ate"],
            ate_se=result["ate_se"],
            ite_estimates=result["ite"],
            quantile_estimates=quantile_estimates,
            fit_time=fit_time,
        )

    def _run_deephte(self, df: pd.DataFrame, seed: int) -> dict:
        """Run DeepHTE on tabular data."""
        from deepstats import DeepHTE

        # Build formula from covariates
        covariate_cols = [c for c in df.columns if c not in ("Y", "T")]
        covs = "+".join(covariate_cols)
        formula = f"Y ~ a({covs}) + b({covs}) * T"

        model = DeepHTE(
            formula=formula,
            epochs=200,
            hidden_dims=[64, 32],
            lr=0.001,
            random_state=seed,
            verbose=0,
        )

        result = model.fit(df)

        return {
            "ate": result.ate,
            "ate_se": result.ate_se,
            "ite": result.ite,
        }

    def _run_causal_forest(self, df: pd.DataFrame, seed: int) -> dict:
        """Run CausalForest on tabular data."""
        from deepstats.comparison import CausalForestWrapper

        model = CausalForestWrapper(
            n_estimators=100,
            cv=5,
            random_state=seed,
        )

        result = model.fit(df)

        return {
            "ate": result.ate,
            "ate_se": result.ate_se,
            "ite": result.ite,
        }

    def _run_linear_dml(self, df: pd.DataFrame, seed: int) -> dict:
        """Run LinearDML on tabular data."""
        from deepstats.comparison import EconMLWrapper

        model = EconMLWrapper(cv=5, random_state=seed)
        result = model.fit(df)

        return {
            "ate": result.ate,
            "ate_se": result.ate_se,
            "ite": result.ite,
        }

    def _run_quantile_forest(self, df: pd.DataFrame, seed: int) -> dict:
        """Run QuantileForest on tabular data."""
        from deepstats.comparison import QuantileForestWrapper

        model = QuantileForestWrapper(
            n_estimators=100,
            quantiles=self.config.quantiles,
            random_state=seed,
        )

        result = model.fit(df)

        return {
            "ate": result.ate,
            "ate_se": result.ate_se,
            "ite": result.ite,
        }
