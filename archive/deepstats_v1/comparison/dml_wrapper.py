"""Wrapper for EconML LinearDML to match DeepHTE interface.

This module provides a wrapper around EconML's LinearDML estimator
to facilitate comparison with DeepHTE in simulation studies.

LinearDML assumes a linear CATE model: b(X) = X @ beta, which makes it
a strong baseline when the true effect is linear, but it can struggle
with complex nonlinear heterogeneity patterns that DeepHTE can capture.

References
----------
- Chernozhukov et al. (2018). "Double/Debiased Machine Learning"
- Microsoft EconML: https://github.com/microsoft/EconML
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DMLResults:
    """Results from LinearDML estimation.

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
    model : Any
        Fitted EconML model.
    """

    ate: float
    ate_se: float
    ite: np.ndarray
    ate_ci: tuple[float, float]
    model: Any = None

    def ate_confint(self, alpha: float = 0.05) -> tuple[float, float]:
        """Compute confidence interval for ATE."""
        from scipy import stats

        z = stats.norm.ppf(1 - alpha / 2)
        return self.ate - z * self.ate_se, self.ate + z * self.ate_se


class EconMLWrapper:
    """Wrapper for EconML LinearDML to match DeepHTE interface.

    LinearDML estimates heterogeneous treatment effects using the
    partially linear model:

        Y = theta(X) * T + g(X) + epsilon

    where theta(X) = X @ beta (linear CATE assumption).

    This wrapper matches the DeepHTE interface for easy comparison
    in simulation studies.

    Parameters
    ----------
    model_y : estimator, optional
        Model for outcome regression. Default: RandomForestRegressor.
    model_t : estimator, optional
        Model for treatment propensity. Default: RandomForestClassifier.
    cv : int, default=5
        Number of cross-validation folds for cross-fitting.
    random_state : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from deepstats.comparison import EconMLWrapper
    >>> import pandas as pd
    >>>
    >>> data = pd.DataFrame({...})
    >>> dml = EconMLWrapper(cv=5, random_state=42)
    >>> result = dml.fit(data)
    >>> print(f"ATE: {result.ate:.3f} (SE: {result.ate_se:.3f})")

    Notes
    -----
    Requires econml to be installed: pip install econml

    The key difference from DeepHTE is that LinearDML assumes a LINEAR
    relationship between X and the treatment effect. For complex nonlinear
    heterogeneity (e.g., interactions, thresholds), DeepHTE is expected
    to perform better.
    """

    def __init__(
        self,
        model_y: Any = None,
        model_t: Any = None,
        cv: int = 5,
        random_state: int | None = None,
    ) -> None:
        self.model_y = model_y
        self.model_t = model_t
        self.cv = cv
        self.random_state = random_state
        self._dml = None
        self._covariate_cols = None

    def fit(self, data: pd.DataFrame) -> DMLResults:
        """Fit LinearDML and return results.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with columns 'Y' (outcome), 'T' (treatment),
            and covariates (everything else).

        Returns
        -------
        DMLResults
            Results object with ATE, ITEs, and inference.
        """
        try:
            from econml.dml import LinearDML
        except ImportError as e:
            raise ImportError(
                "econml is required for EconMLWrapper. "
                "Install it with: pip install econml"
            ) from e

        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        # Extract data
        Y = data["Y"].values
        T = data["T"].values
        X = data.drop(columns=["Y", "T"]).values
        self._covariate_cols = [c for c in data.columns if c not in ("Y", "T")]

        # Default models
        model_y = self.model_y or RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=self.random_state,
        )
        model_t = self.model_t or RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=self.random_state,
        )

        # Fit LinearDML
        self._dml = LinearDML(
            model_y=model_y,
            model_t=model_t,
            cv=self.cv,
            random_state=self.random_state,
            discrete_treatment=True,
        )
        self._dml.fit(Y, T, X=X)

        # Extract results
        ate = float(self._dml.ate(X))
        ite = self._dml.effect(X).flatten()

        # Confidence interval and SE
        ate_inf = self._dml.ate_inference(X)
        ate_se = float(ate_inf.stderr_mean)
        ci = ate_inf.conf_int_mean(alpha=0.05)
        ate_ci = (float(ci[0]), float(ci[1]))

        return DMLResults(
            ate=ate,
            ate_se=ate_se,
            ite=ite,
            ate_ci=ate_ci,
            model=self._dml,
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
        if self._dml is None:
            raise RuntimeError("Model not fitted yet")
        return self._dml.effect(X).flatten()
