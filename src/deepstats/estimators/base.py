"""Base estimator class for deepstats.

This module provides the base class for all deepstats estimators,
implementing the sklearn estimator interface for pipeline compatibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .._typing import DataFrameOrArray, Float64Array

if TYPE_CHECKING:
    from ..results.deep_results import DeepResults


class DeepEstimatorBase(BaseEstimator, RegressorMixin, ABC):
    """Abstract base class for deep learning estimators.

    This class provides the sklearn-compatible interface with:
    - `fit(X, y)` returning a Results object
    - `predict(X)` for generating predictions
    - Support for `clone()` for cross-validation

    All configuration happens in `__init__` (no side effects).
    All computation happens in `fit()`.

    Subclasses must implement `_fit_impl()`.
    """

    @abstractmethod
    def _fit_impl(
        self,
        X: Float64Array,
        y: Float64Array,
        feature_names: list[str],
    ) -> "DeepResults":
        """Implementation of the fitting procedure.

        Parameters
        ----------
        X : Float64Array
            Feature matrix (n, p).
        y : Float64Array
            Target vector (n,).
        feature_names : list[str]
            Names of features.

        Returns
        -------
        DeepResults
            Results object with estimates and inference.
        """
        pass

    def fit(
        self,
        X: DataFrameOrArray,
        y: Float64Array | pd.Series | None = None,
    ) -> "DeepResults":
        """Fit the model.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix or DataFrame containing both features and target
            (if using formula).
        y : array-like, optional
            Target vector. Required if not using formula.

        Returns
        -------
        DeepResults
            Results object with `summary()`, `vcov()`, `confint()` methods.
        """
        X_array, y_array, feature_names = self._validate_input(X, y)
        self.results_ = self._fit_impl(X_array, y_array, feature_names)
        self.is_fitted_ = True
        return self.results_

    def _validate_input(
        self,
        X: DataFrameOrArray,
        y: Float64Array | pd.Series | None,
    ) -> tuple[Float64Array, Float64Array, list[str]]:
        """Validate and convert input data.

        Handles both DataFrame and array inputs, preserving column names
        for interpretable output.
        """
        # Handle formula if specified
        if hasattr(self, "formula") and self.formula is not None:
            return self._parse_formula(X)

        # Extract feature names
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X_array = X.values.astype(np.float64)
        else:
            X_array = np.asarray(X, dtype=np.float64)
            feature_names = [f"x{i}" for i in range(X_array.shape[1])]

        # Validate y
        if y is None:
            raise ValueError("y is required when not using a formula")

        if isinstance(y, pd.Series):
            y_array = y.values.astype(np.float64)
        else:
            y_array = np.asarray(y, dtype=np.float64).flatten()

        # Basic validation
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError(
                f"X and y have inconsistent samples: {X_array.shape[0]} vs {y_array.shape[0]}"
            )

        return X_array, y_array, feature_names

    def _parse_formula(
        self, data: pd.DataFrame
    ) -> tuple[Float64Array, Float64Array, list[str]]:
        """Parse R-style formula using formulaic."""
        try:
            import formulaic
        except ImportError as e:
            raise ImportError(
                "formulaic is required for formula parsing. "
                "Install with: pip install formulaic"
            ) from e

        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a DataFrame when using formula")

        model_matrix = formulaic.model_matrix(self.formula, data)

        y_array = model_matrix.lhs.to_numpy().flatten().astype(np.float64)
        X_df = model_matrix.rhs
        feature_names = list(X_df.columns)
        X_array = X_df.to_numpy().astype(np.float64)

        # Remove intercept (neural net doesn't need it)
        if "Intercept" in feature_names:
            idx = feature_names.index("Intercept")
            feature_names.pop(idx)
            X_array = np.delete(X_array, idx, axis=1)

        return X_array, y_array, feature_names

    def predict(self, X: DataFrameOrArray) -> Float64Array:
        """Generate predictions.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.

        Returns
        -------
        Float64Array
            Predicted values.
        """
        check_is_fitted(self, "results_")

        if isinstance(X, pd.DataFrame):
            X_array = X.values.astype(np.float64)
        else:
            X_array = np.asarray(X, dtype=np.float64)

        return self.results_.predict(X_array)

    def score(self, X: DataFrameOrArray, y: Float64Array | pd.Series) -> float:
        """Return R^2 score.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.
        y : array-like
            True values.

        Returns
        -------
        float
            R^2 score.
        """
        check_is_fitted(self, "results_")

        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.asarray(y)

        y_pred = self.predict(X)
        ss_res = np.sum((y_array - y_pred) ** 2)
        ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)

        if ss_tot < 1e-12:
            return 1.0
        return float(1.0 - ss_res / ss_tot)
