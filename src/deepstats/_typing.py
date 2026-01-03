"""Type definitions for deepstats.

This module provides type aliases using numpy.typing for clear,
consistent type annotations throughout the package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd
    import torch.nn as nn

# Core numeric types
Float64Array = NDArray[np.float64]
Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

# Flexible input types (accept both numpy and pandas)
ArrayLike = Union[Float64Array, "pd.Series", "pd.DataFrame", list]
DataFrameOrArray = Union["pd.DataFrame", Float64Array]

# Generic type variable for estimators
T = TypeVar("T")


class Learner(Protocol):
    """Protocol for sklearn-compatible learners.

    Any object with fit() and predict() methods satisfies this protocol.
    This enables the meta-estimator pattern with arbitrary ML models.
    """

    def fit(self, X: Float64Array, y: Float64Array) -> "Learner":
        """Fit the model to data."""
        ...

    def predict(self, X: Float64Array) -> Float64Array:
        """Generate predictions."""
        ...


class Classifier(Learner, Protocol):
    """Protocol for sklearn-compatible classifiers."""

    def predict_proba(self, X: Float64Array) -> Float64Array:
        """Predict class probabilities."""
        ...


# Network type
Network = "nn.Module"

# Standard error types
SEType = Union[str, None]  # "iid", "HC0", "HC1", "HC2", "HC3", "cluster"
