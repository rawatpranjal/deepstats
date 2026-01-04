"""Linear OLS estimator.

Classical OLS regression with robust standard errors for comparison
with scipy.stats and statsmodels.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import linalg, stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .._typing import Float64Array
from ..inference.standard_errors import compute_vcov
from ..results.deep_results import DeepResults


class LinearOLS(BaseEstimator, RegressorMixin):
    """Classical OLS regression with robust standard errors.

    This estimator fits Y = X @ beta + epsilon using closed-form OLS,
    providing a baseline for comparison with neural network estimators.

    The estimator inherits from sklearn's BaseEstimator for pipeline
    compatibility (clone, GridSearchCV, etc.).

    Parameters
    ----------
    add_intercept : bool, default=True
        Whether to add an intercept term.
    robust_se : str, default="HC1"
        Standard error type: "iid", "HC0", "HC1", "HC2", "HC3".
    formula : str, optional
        R-style formula like "Y ~ X1 + X2". If provided, X should be
        a DataFrame containing all variables.

    Attributes
    ----------
    results_ : DeepResults
        Estimation results after fitting.
    coef_ : ndarray
        Coefficient estimates (sklearn compatibility).
    intercept_ : float
        Intercept term if add_intercept=True.

    Examples
    --------
    >>> from deepstats.estimators import LinearOLS
    >>> import numpy as np
    >>>
    >>> X = np.random.randn(1000, 3)
    >>> y = 2 + 0.5*X[:, 0] - 0.3*X[:, 1] + np.random.randn(1000) * 0.5
    >>>
    >>> model = LinearOLS(robust_se="HC1")
    >>> result = model.fit(X, y)
    >>> print(result.summary())
    """

    def __init__(
        self,
        add_intercept: bool = True,
        robust_se: Literal["iid", "HC0", "HC1", "HC2", "HC3"] = "HC1",
        formula: str | None = None,
    ) -> None:
        self.add_intercept = add_intercept
        self.robust_se = robust_se
        self.formula = formula

    def fit(
        self,
        X: Float64Array | pd.DataFrame,
        y: Float64Array | pd.Series | None = None,
    ) -> DeepResults:
        """Fit OLS model.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix (n, p). If formula is provided, should be
            a DataFrame containing all variables.
        y : array-like, optional
            Target vector (n,). Not needed if formula is provided.

        Returns
        -------
        DeepResults
            Estimation results with inference.
        """
        # Handle formula interface
        if self.formula is not None:
            X_array, y_array, feature_names = self._parse_formula(X)
        else:
            X_array = np.asarray(X, dtype=np.float64)
            y_array = np.asarray(y, dtype=np.float64)
            if isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
            else:
                feature_names = [f"x{i}" for i in range(X_array.shape[1])]

        # Add intercept if requested
        if self.add_intercept:
            n = X_array.shape[0]
            X_array = np.column_stack([np.ones(n), X_array])
            feature_names = ["const"] + feature_names

        # OLS estimation: beta = (X'X)^{-1} X'y
        n, p = X_array.shape
        XtX = X_array.T @ X_array
        Xty = X_array.T @ y_array

        # Use Cholesky for numerical stability
        try:
            L = linalg.cholesky(XtX, lower=True)
            params = linalg.cho_solve((L, True), Xty)
        except linalg.LinAlgError:
            # Fall back to general solver
            params = linalg.solve(XtX, Xty, assume_a="sym")

        # Fitted values and residuals
        fitted_values = X_array @ params
        residuals = y_array - fitted_values

        # Estimate sigma
        df_resid = n - p
        sigma = np.sqrt(np.sum(residuals**2) / df_resid)

        # Compute variance-covariance matrix
        vcov_matrix = compute_vcov(X_array, residuals, se_type=self.robust_se)
        std_errors = np.sqrt(np.diag(vcov_matrix))

        # Store for sklearn compatibility
        if self.add_intercept:
            self.intercept_ = params[0]
            self.coef_ = params[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = params

        self._X_design = X_array  # Store design matrix for predict
        self._feature_names = feature_names

        # Create results object
        self.results_ = DeepResults(
            params=params,
            std_errors=std_errors,
            vcov_matrix=vcov_matrix,
            fitted_values=fitted_values,
            residuals=residuals,
            feature_names=feature_names,
            n_obs=n,
            df_resid=df_resid,
            network_=None,
            loss_history_=[],
            family="normal",
            se_type=self.robust_se,
            sigma_=sigma,
            y_=y_array,
            X_=X_array,
        )

        return self.results_

    def predict(self, X: Float64Array | pd.DataFrame) -> Float64Array:
        """Generate predictions.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix (n_new, p).

        Returns
        -------
        ndarray
            Predictions (n_new,).
        """
        check_is_fitted(self, "results_")

        X_array = np.asarray(X, dtype=np.float64)

        if self.add_intercept:
            n = X_array.shape[0]
            X_array = np.column_stack([np.ones(n), X_array])

        return X_array @ self.results_.params

    def score(self, X: Float64Array, y: Float64Array) -> float:
        """Return R-squared score.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            True target values.

        Returns
        -------
        float
            R-squared score.
        """
        check_is_fitted(self, "results_")
        y_pred = self.predict(X)
        y = np.asarray(y)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1.0 - ss_res / ss_tot

    def _parse_formula(
        self, data: pd.DataFrame
    ) -> tuple[Float64Array, Float64Array, list[str]]:
        """Parse R-style formula.

        Parameters
        ----------
        data : DataFrame
            Data containing all variables.

        Returns
        -------
        tuple
            (X, y, feature_names)
        """
        try:
            from formulaic import Formula
        except ImportError:
            raise ImportError(
                "formulaic is required for formula interface. "
                "Install with: pip install formulaic"
            )

        formula = Formula(self.formula)
        y, X = formula.get_model_matrix(data)

        # Remove intercept if present (we add it ourselves)
        if "Intercept" in X.columns:
            X = X.drop("Intercept", axis=1)

        feature_names = list(X.columns)

        return X.values.astype(np.float64), y.values.flatten().astype(np.float64), feature_names
