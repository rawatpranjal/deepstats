"""Results container for DeepPoisson estimation.

This module provides the PoissonResults class for inference on functionals
of the rate parameter lambda(X) in Poisson regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from .._typing import Float64Array

if TYPE_CHECKING:
    import torch.nn as nn


@dataclass
class PoissonResults:
    """Results from DeepPoisson estimation with inference on lambda functionals.

    This class holds inference results for:
    - E[lambda(X)]: Average rate parameter
    - Var[lambda(X)]: Heterogeneity in rate parameter
    - Quantiles of lambda(X): Distribution of rates

    Attributes
    ----------
    mean_lambda : float
        Average rate parameter: E[lambda(X)] = (1/n) * sum(lambda_hat(X_i))
    mean_lambda_se : float
        Standard error of mean_lambda: std(lambda_hat) / sqrt(n)
    var_lambda : float
        Variance of rate: Var[lambda(X)]
    var_lambda_se : float
        Standard error of var_lambda via influence function
    lambda_values : Float64Array
        Individual rate predictions: lambda(X_i) for each observation
    quantiles : dict[float, float]
        Quantiles of lambda(X) distribution
    quantile_se : dict[float, float]
        Bootstrap standard errors for each quantile
    influence_mean : Float64Array
        Influence function values for mean_lambda
    influence_var : Float64Array
        Influence function values for var_lambda
    n_obs : int
        Number of observations
    network_ : nn.Module
        Trained neural network
    loss_history_ : list[float]
        Training loss at each epoch
    deviance_ : float
        Model deviance
    cross_fit_folds : int
        Number of cross-fitting folds used

    Examples
    --------
    >>> model = DeepPoisson(epochs=100, cross_fit_folds=5)
    >>> result = model.fit(X, y)
    >>> print(result.summary())
    >>> print(f"E[lambda(X)] = {result.mean_lambda:.3f} (SE: {result.mean_lambda_se:.3f})")
    """

    # Point estimates
    mean_lambda: float
    mean_lambda_se: float
    var_lambda: float
    var_lambda_se: float

    # Individual predictions
    lambda_values: Float64Array

    # Quantiles
    quantiles: dict[float, float] = field(default_factory=dict)
    quantile_se: dict[float, float] = field(default_factory=dict)

    # Influence functions
    influence_mean: Float64Array = field(default_factory=lambda: np.array([]))
    influence_var: Float64Array = field(default_factory=lambda: np.array([]))

    # Model internals
    n_obs: int = 0
    network_: Any = None
    loss_history_: list[float] = field(default_factory=list)
    deviance_: float = 0.0
    cross_fit_folds: int = 5

    # Raw data for prediction (not serialized)
    _X: Float64Array = field(default_factory=lambda: np.array([]), repr=False)
    _device: Any = field(default=None, repr=False)

    @property
    def mean_lambda_tstat(self) -> float:
        """t-statistic for mean_lambda."""
        if self.mean_lambda_se < 1e-12:
            return np.inf
        return self.mean_lambda / self.mean_lambda_se

    @property
    def mean_lambda_pvalue(self) -> float:
        """Two-sided p-value for mean_lambda (H0: E[lambda] = 0)."""
        return float(2 * (1 - stats.norm.cdf(np.abs(self.mean_lambda_tstat))))

    @property
    def var_lambda_tstat(self) -> float:
        """t-statistic for var_lambda."""
        if self.var_lambda_se < 1e-12:
            return np.inf
        return self.var_lambda / self.var_lambda_se

    @property
    def std_lambda(self) -> float:
        """Standard deviation of lambda(X)."""
        return float(np.sqrt(self.var_lambda))

    @property
    def cv_lambda(self) -> float:
        """Coefficient of variation of lambda(X)."""
        if self.mean_lambda < 1e-12:
            return np.inf
        return self.std_lambda / self.mean_lambda

    def confint(self, alpha: float = 0.05) -> pd.DataFrame:
        """Confidence intervals for mean_lambda and var_lambda.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level. Default gives 95% CI.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['estimate', 'se', 'lower', 'upper'].
        """
        z = stats.norm.ppf(1 - alpha / 2)

        data = {
            "estimate": [self.mean_lambda, self.var_lambda],
            "se": [self.mean_lambda_se, self.var_lambda_se],
            "lower": [
                self.mean_lambda - z * self.mean_lambda_se,
                self.var_lambda - z * self.var_lambda_se,
            ],
            "upper": [
                self.mean_lambda + z * self.mean_lambda_se,
                self.var_lambda + z * self.var_lambda_se,
            ],
        }

        return pd.DataFrame(data, index=["E[lambda(X)]", "Var[lambda(X)]"])

    def quantile_confint(self, alpha: float = 0.05) -> pd.DataFrame:
        """Confidence intervals for quantiles.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level. Default gives 95% CI.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['estimate', 'se', 'lower', 'upper'].
        """
        z = stats.norm.ppf(1 - alpha / 2)

        rows = []
        for q in sorted(self.quantiles.keys()):
            est = self.quantiles[q]
            se = self.quantile_se.get(q, np.nan)
            rows.append(
                {
                    "quantile": q,
                    "estimate": est,
                    "se": se,
                    "lower": est - z * se if not np.isnan(se) else np.nan,
                    "upper": est + z * se if not np.isnan(se) else np.nan,
                }
            )

        return pd.DataFrame(rows).set_index("quantile")

    def summary(self, alpha: float = 0.05) -> str:
        """Generate a summary table for Poisson parameter inference.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for confidence intervals.

        Returns
        -------
        str
            Formatted summary table.
        """
        z = stats.norm.ppf(1 - alpha / 2)
        ci_level = int((1 - alpha) * 100)

        lines = []
        lines.append("=" * 70)
        lines.append("             Deep Poisson Regression - Parameter Inference")
        lines.append("=" * 70)
        lines.append(f"No. Observations:     {self.n_obs:,}")
        lines.append(f"Cross-fit Folds:      {self.cross_fit_folds}")
        lines.append(f"Model Deviance:       {self.deviance_:.4f}")
        lines.append("-" * 70)
        lines.append("")

        # Mean lambda
        lines.append("Rate Parameter Functionals")
        lines.append("-" * 70)
        lines.append(
            f"  E[lambda(X)]        {self.mean_lambda:>12.4f}  "
            f"(SE: {self.mean_lambda_se:.4f})"
        )
        mean_ci = (
            self.mean_lambda - z * self.mean_lambda_se,
            self.mean_lambda + z * self.mean_lambda_se,
        )
        lines.append(f"    {ci_level}% CI:           [{mean_ci[0]:.4f}, {mean_ci[1]:.4f}]")
        lines.append(
            f"    z-value:          {self.mean_lambda_tstat:.3f}  "
            f"P>|z|: {self.mean_lambda_pvalue:.4f}"
        )
        lines.append("")

        # Var lambda
        lines.append(
            f"  Var[lambda(X)]      {self.var_lambda:>12.4f}  "
            f"(SE: {self.var_lambda_se:.4f})"
        )
        var_ci = (
            self.var_lambda - z * self.var_lambda_se,
            self.var_lambda + z * self.var_lambda_se,
        )
        lines.append(f"    {ci_level}% CI:           [{var_ci[0]:.4f}, {var_ci[1]:.4f}]")
        lines.append(f"    Std[lambda(X)]:   {self.std_lambda:.4f}")
        lines.append(f"    CV[lambda(X)]:    {self.cv_lambda:.4f}")
        lines.append("")

        # Quantiles
        if self.quantiles:
            lines.append("-" * 70)
            lines.append("Quantiles of lambda(X)")
            lines.append("-" * 70)
            lines.append(
                f"  {'Quantile':>10}  {'Estimate':>12}  {'SE':>10}  {f'{ci_level}% CI':>20}"
            )
            for q in sorted(self.quantiles.keys()):
                est = self.quantiles[q]
                se = self.quantile_se.get(q, np.nan)
                if not np.isnan(se):
                    ci_str = f"[{est - z * se:.4f}, {est + z * se:.4f}]"
                else:
                    ci_str = "-"
                lines.append(f"  {q:>10.2f}  {est:>12.4f}  {se:>10.4f}  {ci_str:>20}")
            lines.append("")

        lines.append("-" * 70)
        # Lambda distribution summary
        lines.append("Lambda Distribution Summary")
        lines.append("-" * 70)
        lines.append(f"  Min:      {np.min(self.lambda_values):.4f}")
        lines.append(f"  Max:      {np.max(self.lambda_values):.4f}")
        lines.append(f"  Range:    {np.ptp(self.lambda_values):.4f}")
        lines.append("")

        if self.loss_history_:
            lines.append("-" * 70)
            lines.append(f"Epochs trained:       {len(self.loss_history_)}")
            lines.append(f"Final loss:           {self.loss_history_[-1]:.6f}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def predict(self, X: Float64Array | None = None) -> Float64Array:
        """Predict lambda(X) for new data.

        Parameters
        ----------
        X : Float64Array, optional
            Feature matrix. If None, returns stored lambda_values.

        Returns
        -------
        Float64Array
            Predicted rate parameters lambda(X).
        """
        if X is None:
            return self.lambda_values

        if self.network_ is None:
            raise ValueError("Network not stored. Cannot predict on new data.")

        import torch

        self.network_.eval()
        device = self._device if self._device is not None else torch.device("cpu")

        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(device)
            eta = self.network_(X_tensor).squeeze()
            # Clamp to prevent overflow in exp
            eta_clamped = torch.clamp(eta, max=20)
            lambda_pred = torch.exp(eta_clamped).cpu().numpy()

        return lambda_pred

    def __repr__(self) -> str:
        return (
            f"PoissonResults(n_obs={self.n_obs}, "
            f"mean_lambda={self.mean_lambda:.4f}, "
            f"var_lambda={self.var_lambda:.4f}, "
            f"cross_fit_folds={self.cross_fit_folds})"
        )
