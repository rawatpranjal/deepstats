"""Text simulation runner."""

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
from simulations.dgp import get_text_dgp


class TextSimulationRunner(BaseSimulationRunner):
    """Simulation runner for text data.

    Runs DeepHTE with LSTM backbone on text HTE data.
    Note: Comparison methods require text features or embeddings.
    """

    @property
    def modality(self) -> str:
        return "text"

    def _generate_data(self, dgp_name: str, seed: int) -> dict:
        """Generate text data."""
        dgp = get_text_dgp(dgp_name)
        return dgp.generate(seed, n=self.config.n_samples)

    def _run_method(
        self, method: str, data: dict, seed: int
    ) -> MethodResult:
        """Run a method on text data."""
        start_time = time.time()

        if method == "deephte":
            result = self._run_deephte(data, seed)
        elif method in ("causal_forest", "linear_dml", "quantile_forest"):
            result = self._run_with_features(method, data, seed)
        else:
            raise ValueError(f"Unknown method: {method}")

        fit_time = time.time() - start_time

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

    def _run_deephte(self, data: dict, seed: int) -> dict:
        """Run DeepHTE with text backbone on raw tokens."""
        from deepstats import DeepHTE

        model = DeepHTE(
            backbone="text",
            epochs=100,  # Reduced for faster simulation
            lr=0.001,
            dropout=0.1,
            random_state=seed,
            verbose=0,
        )

        # Use raw tokens instead of extracted features
        result = model.fit_raw(
            X=data["tokens"],  # (n, seq_len) token indices
            y=data["outcome"],
            t=data["treatment"],
            vocab_size=data.get("vocab_size", 1000),
        )

        return {"ate": result.ate, "ate_se": result.ate_se, "ite": result.ite}

    def _run_with_features(
        self, method: str, data: dict, seed: int
    ) -> dict:
        """Run method using extracted text features."""
        features = data["text_features"]
        n_features = features.shape[1]
        df = pd.DataFrame(
            features,
            columns=[f"X{i+1}" for i in range(n_features)],
        )
        df["T"] = data["treatment"]
        df["Y"] = data["outcome"]

        if method == "deephte":
            from deepstats import DeepHTE

            covs = "+".join([f"X{i+1}" for i in range(n_features)])
            formula = f"Y ~ a({covs}) + b({covs}) * T"

            model = DeepHTE(
                formula=formula,
                epochs=100,
                hidden_dims=[128, 64],
                random_state=seed,
                verbose=0,
            )
            result = model.fit(df)
            return {"ate": result.ate, "ate_se": result.ate_se, "ite": result.ite}

        elif method == "causal_forest":
            from deepstats.comparison import CausalForestWrapper

            model = CausalForestWrapper(n_estimators=100, random_state=seed)
            result = model.fit(df)
            return {"ate": result.ate, "ate_se": result.ate_se, "ite": result.ite}

        elif method == "linear_dml":
            from deepstats.comparison import EconMLWrapper

            model = EconMLWrapper(random_state=seed)
            result = model.fit(df)
            return {"ate": result.ate, "ate_se": result.ate_se, "ite": result.ite}

        elif method == "quantile_forest":
            from deepstats.comparison import QuantileForestWrapper

            model = QuantileForestWrapper(random_state=seed)
            result = model.fit(df)
            return {"ate": result.ate, "ate_se": result.ate_se, "ite": result.ite}

        else:
            raise ValueError(f"Unknown method: {method}")
