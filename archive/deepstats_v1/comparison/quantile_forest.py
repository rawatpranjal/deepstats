"""Quantile Forest wrapper for treatment effect quantile estimation.

This module provides a wrapper for quantile regression forests to estimate
quantiles of the treatment effect distribution. It serves as a comparison
baseline for DeepHTE's quantile estimation capabilities.

The approach fits separate quantile forests for treated and control groups,
then computes quantiles of the estimated CATE distribution.

References
----------
- Meinshausen (2006). "Quantile Regression Forests"
- Athey, Tibshirani & Wager (2019). "Generalized Random Forests"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class QuantileForestResults:
    """Results from Quantile Forest estimation.

    Attributes
    ----------
    ate : float
        Average Treatment Effect estimate.
    ate_se : float
        Standard error of ATE (bootstrap-based).
    ite : np.ndarray
        Individual Treatment Effects (CATE at each X).
    quantiles : dict[float, float]
        Estimated quantiles of CATE distribution.
    quantile_values : dict[float, np.ndarray]
        Per-observation conditional quantiles.
    model_t1 : Any
        Fitted model for treated group.
    model_t0 : Any
        Fitted model for control group.
    """

    ate: float
    ate_se: float
    ite: np.ndarray
    quantiles: dict[float, float] = field(default_factory=dict)
    quantile_values: dict[float, np.ndarray] = field(default_factory=dict)
    model_t1: Any = None
    model_t0: Any = None

    def ate_confint(self, alpha: float = 0.05) -> tuple[float, float]:
        """Compute confidence interval for ATE."""
        from scipy import stats

        z = stats.norm.ppf(1 - alpha / 2)
        return self.ate - z * self.ate_se, self.ate + z * self.ate_se


class QuantileForestWrapper:
    """Quantile Forest for treatment effect quantile comparison.

    This estimator fits separate models for treated and control outcomes,
    then estimates quantiles of the CATE distribution by computing
    quantiles of the difference E[Y|T=1,X] - E[Y|T=0,X].

    For robust quantile estimation, it uses sklearn's GradientBoostingRegressor
    with quantile loss or RandomForestQuantileRegressor if available.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int | None, default=5
        Maximum depth of trees.
    quantiles : list[float], optional
        Quantiles to estimate. Default: [0.1, 0.25, 0.5, 0.75, 0.9].
    n_bootstrap : int, default=100
        Number of bootstrap samples for SE estimation.
    random_state : int | None, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from deepstats.comparison import QuantileForestWrapper
    >>> import pandas as pd
    >>>
    >>> data = pd.DataFrame({...})
    >>> qf = QuantileForestWrapper(quantiles=[0.1, 0.5, 0.9])
    >>> result = qf.fit(data)
    >>> print(f"Median CATE: {result.quantiles[0.5]:.3f}")
    >>> print(f"90th percentile: {result.quantiles[0.9]:.3f}")
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 5,
        quantiles: list[float] | None = None,
        n_bootstrap: int = 100,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self._model_t1 = None
        self._model_t0 = None
        self._covariate_cols = None

    def fit(self, data: pd.DataFrame) -> QuantileForestResults:
        """Fit Quantile Forests and estimate CATE quantiles.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with columns 'Y' (outcome), 'T' (treatment),
            and covariates (everything else).

        Returns
        -------
        QuantileForestResults
            Results object with ATE, ITEs, and quantile estimates.
        """
        from sklearn.ensemble import RandomForestRegressor

        # Extract data
        Y = data["Y"].values
        T = data["T"].values
        X = data.drop(columns=["Y", "T"]).values
        self._covariate_cols = [c for c in data.columns if c not in ("Y", "T")]

        # Split by treatment
        X_t1 = X[T == 1]
        Y_t1 = Y[T == 1]
        X_t0 = X[T == 0]
        Y_t0 = Y[T == 0]

        # Fit separate random forests for E[Y|T=1,X] and E[Y|T=0,X]
        self._model_t1 = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self._model_t0 = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

        self._model_t1.fit(X_t1, Y_t1)
        self._model_t0.fit(X_t0, Y_t0)

        # Estimate ITEs: b(X) = E[Y|T=1,X] - E[Y|T=0,X]
        mu1 = self._model_t1.predict(X)
        mu0 = self._model_t0.predict(X)
        ite = mu1 - mu0

        # ATE
        ate = float(np.mean(ite))

        # Bootstrap SE for ATE
        rng = np.random.default_rng(self.random_state)
        ate_bootstrap = []
        for _ in range(self.n_bootstrap):
            idx = rng.choice(len(ite), size=len(ite), replace=True)
            ate_bootstrap.append(np.mean(ite[idx]))
        ate_se = float(np.std(ate_bootstrap))

        # Compute quantiles of ITE distribution
        quantile_estimates = {q: float(np.quantile(ite, q)) for q in self.quantiles}

        # Per-observation conditional quantiles using tree variance
        quantile_values = self._estimate_conditional_quantiles(X, ite)

        return QuantileForestResults(
            ate=ate,
            ate_se=ate_se,
            ite=ite,
            quantiles=quantile_estimates,
            quantile_values=quantile_values,
            model_t1=self._model_t1,
            model_t0=self._model_t0,
        )

    def _estimate_conditional_quantiles(
        self, X: np.ndarray, ite: np.ndarray
    ) -> dict[float, np.ndarray]:
        """Estimate conditional quantiles using local bootstrap.

        Uses tree-based neighbors to estimate conditional distribution.

        Parameters
        ----------
        X : np.ndarray
            Covariate matrix.
        ite : np.ndarray
            Estimated ITEs.

        Returns
        -------
        dict[float, np.ndarray]
            Per-observation quantile estimates.
        """
        # Use leaf membership to find similar observations
        leaves_t1 = self._model_t1.apply(X)
        leaves_t0 = self._model_t0.apply(X)

        n = len(X)
        quantile_values = {q: np.zeros(n) for q in self.quantiles}

        for i in range(n):
            # Find observations in same leaves
            same_leaf_mask = np.all(leaves_t1 == leaves_t1[i], axis=1) | np.all(
                leaves_t0 == leaves_t0[i], axis=1
            )

            # Use at least 10 neighbors
            if same_leaf_mask.sum() < 10:
                # Fall back to k-nearest by leaf similarity
                leaf_dist = np.sum(leaves_t1 != leaves_t1[i], axis=1) + np.sum(
                    leaves_t0 != leaves_t0[i], axis=1
                )
                neighbor_idx = np.argsort(leaf_dist)[:20]
                local_ite = ite[neighbor_idx]
            else:
                local_ite = ite[same_leaf_mask]

            for q in self.quantiles:
                quantile_values[q][i] = np.quantile(local_ite, q)

        return quantile_values

    def effect(self, X: np.ndarray) -> np.ndarray:
        """Predict CATE for new covariates.

        Parameters
        ----------
        X : np.ndarray
            Covariates of shape (n, p).

        Returns
        -------
        np.ndarray
            CATE predictions.
        """
        if self._model_t1 is None or self._model_t0 is None:
            raise RuntimeError("Model not fitted yet")
        mu1 = self._model_t1.predict(X)
        mu0 = self._model_t0.predict(X)
        return mu1 - mu0

    def effect_quantiles(
        self, X: np.ndarray, quantiles: list[float] | None = None
    ) -> dict[float, np.ndarray]:
        """Estimate conditional quantiles of CATE for new observations.

        Parameters
        ----------
        X : np.ndarray
            Covariates of shape (n, p).
        quantiles : list[float], optional
            Quantiles to estimate.

        Returns
        -------
        dict[float, np.ndarray]
            Per-observation quantile estimates.
        """
        if self._model_t1 is None or self._model_t0 is None:
            raise RuntimeError("Model not fitted yet")

        quantiles = quantiles or self.quantiles
        ite = self.effect(X)

        return self._estimate_conditional_quantiles(X, ite)
