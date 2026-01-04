"""Comparison module for benchmarking DeepHTE against other methods.

This module provides wrappers for alternative causal inference methods
to facilitate comparison in simulation studies.

Available Wrappers
------------------
EconMLWrapper : Wrapper for EconML LinearDML
    Uses linear CATE assumption, good baseline for comparison.
CausalForestWrapper : Wrapper for EconML CausalForestDML
    Flexible nonparametric CATE estimation with valid inference.
QuantileForestWrapper : Quantile Forest for CATE quantile estimation
    Estimates quantiles of the treatment effect distribution.

Examples
--------
>>> from deepstats.comparison import EconMLWrapper, CausalForestWrapper
>>> from deepstats import DeepHTE
>>> import pandas as pd
>>>
>>> # Load data
>>> data = pd.read_csv("ab_test.csv")
>>>
>>> # Fit multiple methods
>>> deephte = DeepHTE(formula="Y ~ a(X1+X2) + b(X1+X2) * T")
>>> dml = EconMLWrapper()
>>> cf = CausalForestWrapper()
>>>
>>> result_deep = deephte.fit(data)
>>> result_dml = dml.fit(data)
>>> result_cf = cf.fit(data)
>>>
>>> # Compare ATEs
>>> print(f"DeepHTE ATE: {result_deep.ate:.3f}")
>>> print(f"LinearDML ATE: {result_dml.ate:.3f}")
>>> print(f"CausalForest ATE: {result_cf.ate:.3f}")
"""

from .causal_forest import CausalForestWrapper
from .dml_wrapper import EconMLWrapper
from .quantile_forest import QuantileForestWrapper

__all__ = [
    "EconMLWrapper",
    "CausalForestWrapper",
    "QuantileForestWrapper",
]
