"""Results container for deep learning estimation.

This module provides the DeepResults class that holds estimation results
and provides methods for statistical inference (summary, vcov, confint).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

from .._typing import Float64Array

if TYPE_CHECKING:
    import torch.nn as nn


@dataclass
class DeepResults:
    """Container for estimation results with statistical inference.

    This class follows the statsmodels pattern: it holds all estimation
    results and provides methods for inference and prediction.

    Attributes
    ----------
    params : Float64Array
        Estimated parameters (average marginal effects for neural nets).
    std_errors : Float64Array
        Standard errors of the estimates.
    vcov_matrix : Float64Array
        Variance-covariance matrix.
    fitted_values : Float64Array
        In-sample predictions.
    residuals : Float64Array
        Residuals (y - fitted_values).
    feature_names : list[str]
        Names of input features.
    n_obs : int
        Number of observations.
    df_resid : int
        Residual degrees of freedom.
    network_ : nn.Module
        Trained neural network (nuisance store).
    loss_history_ : list[float]
        Training loss at each epoch (nuisance store).
    family : str
        Distribution family name.
    se_type : str
        Type of standard errors computed.

    Examples
    --------
    >>> result = DeepOLS().fit(X, y)
    >>> print(result.summary())
    >>> result.confint(alpha=0.05)
    """

    params: Float64Array
    std_errors: Float64Array
    vcov_matrix: Float64Array
    fitted_values: Float64Array
    residuals: Float64Array
    feature_names: list[str]
    n_obs: int
    df_resid: int
    network_: Any = None  # nn.Module, but avoid type import issues
    loss_history_: list[float] = field(default_factory=list)
    family: str = "normal"
    se_type: str = "HC1"
    sigma_: float | None = None
    y_: Float64Array | None = None
    X_: Float64Array | None = None

    @property
    def mse(self) -> float:
        """Mean squared error."""
        return float(np.mean(self.residuals**2))

    @property
    def rmse(self) -> float:
        """Root mean squared error."""
        return float(np.sqrt(self.mse))

    @property
    def r_squared(self) -> float:
        """R-squared (coefficient of determination)."""
        if self.y_ is None:
            ss_tot = np.sum((self.fitted_values - np.mean(self.fitted_values)) ** 2)
        else:
            ss_tot = np.sum((self.y_ - np.mean(self.y_)) ** 2)
        ss_res = np.sum(self.residuals**2)
        if ss_tot < 1e-12:
            return 1.0
        return float(1.0 - ss_res / ss_tot)

    @property
    def adj_r_squared(self) -> float:
        """Adjusted R-squared."""
        n = self.n_obs
        p = len(self.feature_names)
        r2 = self.r_squared
        if n - p - 1 <= 0:
            return r2
        return float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))

    @property
    def tvalues(self) -> Float64Array:
        """t-statistics for the parameters."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                self.std_errors > 1e-12,
                self.params / self.std_errors,
                np.nan,
            )

    @property
    def pvalues(self) -> Float64Array:
        """Two-sided p-values for the parameters."""
        return 2 * (1 - stats.t.cdf(np.abs(self.tvalues), df=self.df_resid))

    def vcov(self, se_type: str | None = None) -> Float64Array:
        """Get variance-covariance matrix.

        Parameters
        ----------
        se_type : str, optional
            Type of standard errors. If None, returns stored vcov.
            Currently stored vcov is returned; future versions will
            support recomputing with different se_type.

        Returns
        -------
        Float64Array
            Variance-covariance matrix.
        """
        if se_type is not None and se_type != self.se_type:
            raise NotImplementedError(
                f"Recomputing vcov with se_type='{se_type}' not yet implemented. "
                f"Current se_type is '{self.se_type}'."
            )
        return self.vcov_matrix

    def confint(self, alpha: float = 0.05) -> pd.DataFrame:
        """Compute confidence intervals.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level. Default gives 95% CI.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['lower', 'upper'] indexed by feature names.
        """
        q = stats.t.ppf(1 - alpha / 2, df=self.df_resid)
        lower = self.params - q * self.std_errors
        upper = self.params + q * self.std_errors
        return pd.DataFrame(
            {"lower": lower, "upper": upper},
            index=self.feature_names,
        )

    def summary(self, alpha: float = 0.05) -> str:
        """Generate a Stata-style summary table.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for confidence intervals.

        Returns
        -------
        str
            Formatted summary table.
        """
        lines = []
        lines.append("=" * 78)
        lines.append("                    Deep Neural Network Regression Results")
        lines.append("=" * 78)
        lines.append(f"Family:           {self.family.capitalize()}")
        lines.append(f"SE Type:          {self.se_type}")
        lines.append(f"No. Observations: {self.n_obs:,}")
        lines.append(f"Df Residuals:     {self.df_resid:,}")
        lines.append(f"Df Model:         {len(self.feature_names)}")
        lines.append("-" * 78)
        lines.append(f"R-squared:        {self.r_squared:.6f}")
        lines.append(f"Adj. R-squared:   {self.adj_r_squared:.6f}")
        lines.append(f"MSE:              {self.mse:.6f}")
        lines.append(f"RMSE:             {self.rmse:.6f}")
        if self.sigma_ is not None:
            lines.append(f"Sigma:            {self.sigma_:.6f}")
        lines.append("=" * 78)
        lines.append("")

        # Coefficient table
        ci = self.confint(alpha)
        ci_level = int((1 - alpha) * 100)

        rows = []
        for i, name in enumerate(self.feature_names):
            rows.append(
                [
                    name,
                    f"{self.params[i]:.6f}",
                    f"{self.std_errors[i]:.6f}",
                    f"{self.tvalues[i]:.3f}",
                    f"{self.pvalues[i]:.4f}" if not np.isnan(self.pvalues[i]) else "-",
                    f"[{ci.iloc[i]['lower']:.4f}, {ci.iloc[i]['upper']:.4f}]",
                ]
            )

        headers = ["", "coef", "std err", "t", "P>|t|", f"[{ci_level}% CI]"]
        lines.append(tabulate(rows, headers=headers, tablefmt="simple"))
        lines.append("-" * 78)

        if self.loss_history_:
            lines.append(f"Epochs trained:   {len(self.loss_history_)}")
            lines.append(f"Final loss:       {self.loss_history_[-1]:.6f}")

        lines.append("=" * 78)
        return "\n".join(lines)

    def predict(self, X: Float64Array | None = None) -> Float64Array:
        """Generate predictions.

        Parameters
        ----------
        X : Float64Array, optional
            Feature matrix. If None, returns fitted values.

        Returns
        -------
        Float64Array
            Predicted values.
        """
        if X is None:
            return self.fitted_values

        if self.network_ is None:
            raise ValueError("Network not stored. Cannot predict on new data.")

        import torch

        self.network_.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float()
            predictions = self.network_(X_tensor).squeeze().numpy()
        return predictions

    def __repr__(self) -> str:
        return (
            f"DeepResults(n_obs={self.n_obs}, "
            f"n_features={len(self.feature_names)}, "
            f"R2={self.r_squared:.4f}, "
            f"se_type='{self.se_type}')"
        )
