"""Tabular DGP wrappers for simulation studies.

Wraps the existing DGP implementations in deepstats.simulations.dgp
and deepstats.simulations.tough_dgp to provide unified interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from deepstats.simulations.dgp import (
    make_overfit_scenario,
    make_underfit_scenario,
    make_balanced_scenario,
    make_high_noise_scenario,
    make_sparse_scenario,
    make_confounded_scenario,
)
from deepstats.simulations.tough_dgp import (
    make_mixed_tough_scenario,
    make_sparse_nonlinear_scenario,
    make_threshold_scenario,
    make_deep_interaction_scenario,
    make_multifreq_scenario,
)


@dataclass
class TabularDGP:
    """Wrapper for tabular DGP functions.

    Attributes
    ----------
    name : str
        Name of the DGP scenario.
    dgp_func : callable
        Function that generates data given a seed.
    description : str
        Description of the scenario.
    """

    name: str
    dgp_func: Callable[[int], Any]
    description: str

    def generate(self, seed: int) -> dict:
        """Generate data for this DGP.

        Parameters
        ----------
        seed : int
            Random seed.

        Returns
        -------
        dict
            Dictionary with 'data', 'true_ate', 'true_ite' keys.
        """
        result = self.dgp_func(seed)
        if hasattr(result, "data"):
            return {
                "data": result.data,
                "true_ate": result.true_ate,
                "true_ite": result.true_ite,
            }
        return result


# Registry of available tabular DGPs
TABULAR_DGPS: dict[str, TabularDGP] = {}


def register_dgp(name: str, dgp_func: Callable, description: str = "") -> None:
    """Register a tabular DGP."""
    TABULAR_DGPS[name] = TabularDGP(name, dgp_func, description)


def get_tabular_dgp(name: str) -> TabularDGP:
    """Get a tabular DGP by name."""
    if name not in TABULAR_DGPS:
        raise ValueError(
            f"Unknown DGP: {name}. Available: {list(TABULAR_DGPS.keys())}"
        )
    return TABULAR_DGPS[name]


# Register standard DGPs
register_dgp(
    "overfit",
    lambda seed: make_overfit_scenario(seed, n=100, p=50),
    "Small n, high p - tests overfitting detection",
)

register_dgp(
    "underfit",
    lambda seed: make_underfit_scenario(seed, n=5000),
    "Large n, complex function - tests model capacity",
)

register_dgp(
    "balanced",
    lambda seed: make_balanced_scenario(seed, n=2000, p=10),
    "Balanced setting with appropriate complexity",
)

register_dgp(
    "high_noise",
    lambda seed: make_high_noise_scenario(seed, n=2000, signal_to_noise=0.5),
    "High noise scenario - tests robustness",
)

register_dgp(
    "sparse",
    lambda seed: make_sparse_scenario(seed, n=2000, p=50, p_relevant=5),
    "Sparse true signal - tests variable selection",
)

register_dgp(
    "confounded",
    lambda seed: make_confounded_scenario(seed, n=2000, confounding_strength=0.7),
    "Confounded treatment assignment",
)

# Register tough DGPs
register_dgp(
    "mixed",
    lambda seed: make_mixed_tough_scenario(seed, n=2000),
    "Mixed linear/nonlinear pattern",
)

register_dgp(
    "sparse_nonlinear",
    lambda seed: make_sparse_nonlinear_scenario(seed, n=2000, p=50),
    "Sparse nonlinear effects in high dimensions",
)

register_dgp(
    "threshold",
    lambda seed: make_threshold_scenario(seed, n=2000),
    "Threshold-based treatment effect",
)

register_dgp(
    "deep_interaction",
    lambda seed: make_deep_interaction_scenario(seed, n=2000),
    "Deep 3-way and 4-way interactions",
)

register_dgp(
    "multifreq",
    lambda seed: make_multifreq_scenario(seed, n=2000),
    "Multi-frequency periodic patterns",
)


def list_tabular_dgps() -> list[str]:
    """List available tabular DGP names."""
    return list(TABULAR_DGPS.keys())
