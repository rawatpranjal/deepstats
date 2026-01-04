"""Time series DGP wrappers for simulation studies.

Wraps the time series HTE generator from deepstats.datasets.multimodal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from deepstats.datasets.multimodal import make_timeseries_hte, TimeSeriesHTEData


@dataclass
class TimeSeriesDGP:
    """Wrapper for time series DGP functions.

    Attributes
    ----------
    name : str
        Name of the DGP scenario.
    heterogeneity : str
        Type of heterogeneity (trend, volatility, seasonality, complex).
    seq_len : int
        Sequence length.
    n_features : int
        Number of features per timestep.
    description : str
        Description of the scenario.
    """

    name: str
    heterogeneity: Literal["trend", "volatility", "seasonality", "complex"]
    seq_len: int = 50
    n_features: int = 3
    description: str = ""

    def generate(
        self, seed: int, n: int = 2000, noise_scale: float = 1.0
    ) -> dict:
        """Generate time series HTE data.

        Parameters
        ----------
        seed : int
            Random seed.
        n : int, default=2000
            Number of observations.
        noise_scale : float, default=1.0
            Noise scale.

        Returns
        -------
        dict
            Dictionary with time series data and ground truth.
        """
        data = make_timeseries_hte(
            n=n,
            seq_len=self.seq_len,
            n_features=self.n_features,
            heterogeneity=self.heterogeneity,
            noise_scale=noise_scale,
            seed=seed,
        )

        return {
            "sequences": data.sequences,
            "treatment": data.treatment,
            "outcome": data.outcome,
            "true_ate": data.true_ate,
            "true_ite": data.true_ite,
            "ts_features": data.ts_features,
        }


# Registry of time series DGPs
TIMESERIES_DGPS: dict[str, TimeSeriesDGP] = {
    "trend": TimeSeriesDGP(
        name="trend",
        heterogeneity="trend",
        description="Treatment effect varies with sequence trend",
    ),
    "volatility": TimeSeriesDGP(
        name="volatility",
        heterogeneity="volatility",
        description="Treatment effect varies with sequence volatility",
    ),
    "seasonality": TimeSeriesDGP(
        name="seasonality",
        heterogeneity="seasonality",
        description="Treatment effect varies with seasonal patterns",
    ),
    "complex": TimeSeriesDGP(
        name="complex",
        heterogeneity="complex",
        description="Complex combination of time series features",
    ),
}


def get_timeseries_dgp(name: str) -> TimeSeriesDGP:
    """Get a time series DGP by name."""
    if name not in TIMESERIES_DGPS:
        raise ValueError(
            f"Unknown time series DGP: {name}. Available: {list(TIMESERIES_DGPS.keys())}"
        )
    return TIMESERIES_DGPS[name]


def list_timeseries_dgps() -> list[str]:
    """List available time series DGP names."""
    return list(TIMESERIES_DGPS.keys())
