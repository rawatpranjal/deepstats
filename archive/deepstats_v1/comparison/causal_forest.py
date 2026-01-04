"""Wrapper for EconML CausalForestDML.

This module provides a wrapper around EconML's CausalForestDML estimator
for comparison with DeepHTE in simulation studies.

CausalForestDML combines honest random forests with double machine learning
to estimate heterogeneous treatment effects with valid confidence intervals.
Unlike LinearDML, it can capture nonlinear treatment effect heterogeneity.

References
----------
- Athey, Tibshirani & Wager (2019). "Generalized Random Forests"
- Wager & Athey (2018). "Estimation and Inference of HTE using Random Forests"
- Microsoft EconML: https://github.com/microsoft/EconML
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CausalForestResults:
    """Results from CausalForestDML estimation.

    Attributes
    ----------
    ate : float
        Average Treatment Effect estimate.
    ate_se : float
        Standard error of ATE.
    ite : np.ndarray
        Individual Treatment Effects (CATE at each X).
    ate_ci : tuple[float, float]
        95% confidence interval for ATE.
    ite_ci : np.ndarray | None
        Confidence intervals for ITEs, shape (n, 2).
    model : Any
        Fitted CausalForestDML model.
    """

    ate: float
    ate_se: float
    ite: np.ndarray
    ate_ci: tuple[float, float]
    ite_ci: np.ndarray | None = None
    model: Any = None

    def ate_confint(self, alpha: float = 0.05) -> tuple[float, float]:
        """Compute confidence interval for ATE."""
        from scipy import stats

        z = stats.norm.ppf(1 - alpha / 2)
        return self.ate - z * self.ate_se, self.ate + z * self.ate_se

    def effect_quantiles(
        self, quantiles: list[float] | None = None
    ) -> dict[float, float]:
        """Compute quantiles of ITE distribution.

        Parameters
        ----------
        quantiles : list[float], optional
            Quantiles to compute. Default: [0.1, 0.25, 0.5, 0.75, 0.9].

        Returns
        -------
        dict[float, float]
            Mapping from quantile to estimated value.
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        return {q: float(np.quantile(self.ite, q)) for q in quantiles}


class CausalForestWrapper:
    """Wrapper for EconML CausalForestDML to match DeepHTE interface.

    CausalForestDML uses a combination of honest random forests and
    double machine learning for flexible heterogeneous treatment effect
    estimation with valid inference.

    This wrapper matches the DeepHTE interface for easy comparison
    in simulation studies.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int | None, default=None
        Maximum depth of trees. None means unlimited.
    min_samples_leaf : int, default=5
        Minimum samples required at a leaf node.
    cv : int, default=5
        Number of cross-validation folds for cross-fitting.
    random_state : int | None, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from deepstats.comparison import CausalForestWrapper
    >>> import pandas as pd
    >>>
    >>> data = pd.DataFrame({...})
    >>> cf = CausalForestWrapper(n_estimators=200, random_state=42)
    >>> result = cf.fit(data)
    >>> print(f"ATE: {result.ate:.3f} (SE: {result.ate_se:.3f})")
    >>> print(f"ITE quantiles: {result.effect_quantiles()}")

    Notes
    -----
    Requires econml to be installed: pip install econml

    CausalForestDML can capture nonlinear heterogeneity but requires
    more data than LinearDML to achieve stable estimates.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 5,
        cv: int = 5,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.cv = cv
        self.random_state = random_state
        self._model = None
        self._covariate_cols = None

    def fit(self, data: pd.DataFrame) -> CausalForestResults:
        """Fit CausalForestDML and return results.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with columns 'Y' (outcome), 'T' (treatment),
            and covariates (everything else).

        Returns
        -------
        CausalForestResults
            Results object with ATE, ITEs, and inference.
        """
        try:
            from econml.dml import CausalForestDML
        except ImportError as e:
            raise ImportError(
                "econml is required for CausalForestWrapper. "
                "Install it with: pip install econml"
            ) from e

        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        # Extract data
        Y = data["Y"].values
        T = data["T"].values
        X = data.drop(columns=["Y", "T"]).values
        self._covariate_cols = [c for c in data.columns if c not in ("Y", "T")]

        # Nuisance models
        model_y = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=self.random_state,
        )
        model_t = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=self.random_state,
        )

        # Fit CausalForestDML
        self._model = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            cv=self.cv,
            random_state=self.random_state,
            discrete_treatment=True,
        )
        self._model.fit(Y, T, X=X)

        # Extract results
        ate = float(self._model.ate(X))
        ite = self._model.effect(X).flatten()

        # Confidence intervals and SE
        ate_inf = self._model.ate_inference(X)
        ate_se = float(ate_inf.stderr_mean)
        ci = ate_inf.conf_int_mean(alpha=0.05)
        ate_ci = (float(ci[0]), float(ci[1]))

        # ITE confidence intervals
        ite_inf = self._model.effect_inference(X)
        ite_ci_raw = ite_inf.conf_int(alpha=0.05)
        ite_ci = np.column_stack([ite_ci_raw[0].flatten(), ite_ci_raw[1].flatten()])

        return CausalForestResults(
            ate=ate,
            ate_se=ate_se,
            ite=ite,
            ate_ci=ate_ci,
            ite_ci=ite_ci,
            model=self._model,
        )

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
        if self._model is None:
            raise RuntimeError("Model not fitted yet")
        return self._model.effect(X).flatten()

    def effect_interval(
        self, X: np.ndarray, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get confidence intervals for CATE predictions.

        Parameters
        ----------
        X : np.ndarray
            Covariates of shape (n, p).
        alpha : float, default=0.05
            Significance level.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Lower and upper bounds for CATE.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted yet")
        inf = self._model.effect_inference(X)
        ci = inf.conf_int(alpha=alpha)
        return ci[0].flatten(), ci[1].flatten()
