"""GLM-specific results container.

Extends DeepResults with GLM-specific metrics and methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from .._typing import Float64Array
from .deep_results import DeepResults

if TYPE_CHECKING:
    from ..families.base import ExponentialFamily


@dataclass
class GLMResults(DeepResults):
    """Results container for Deep GLM estimation.

    Extends DeepResults with GLM-specific attributes and methods.

    Additional Attributes
    ---------------------
    dispersion_ : float
        Estimated dispersion parameter.
    deviance_ : float
        Model deviance (goodness of fit).
    family_obj_ : ExponentialFamily
        Distribution family object (for predictions).
    """

    dispersion_: float = 1.0
    deviance_: float = 0.0
    family_obj_: Any = None

    @property
    def null_deviance(self) -> float:
        """Null deviance (intercept-only model)."""
        if self.y_ is None:
            return np.nan

        import torch

        y_mean = np.mean(self.y_)
        mu_null = np.full_like(self.y_, y_mean)

        if self.family_obj_ is not None:
            return self.family_obj_.deviance(
                torch.from_numpy(self.y_).float(),
                torch.from_numpy(mu_null).float(),
                self.dispersion_,
            ).item()
        return np.nan

    @property
    def pseudo_r_squared(self) -> float:
        """McFadden's pseudo R-squared: 1 - deviance/null_deviance."""
        null_dev = self.null_deviance
        if np.isnan(null_dev) or null_dev < 1e-10:
            return np.nan
        return 1.0 - self.deviance_ / null_dev

    @property
    def aic(self) -> float:
        """Akaike Information Criterion.

        AIC = 2k + deviance/dispersion
        where k is the number of parameters.
        """
        k = len(self.feature_names)
        return 2 * k + self.deviance_ / self.dispersion_

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion.

        BIC = k * log(n) + deviance/dispersion
        """
        k = len(self.feature_names)
        return k * np.log(self.n_obs) + self.deviance_ / self.dispersion_

    def summary(self, alpha: float = 0.05) -> str:
        """Generate GLM summary table.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for confidence intervals.

        Returns
        -------
        str
            Formatted summary table.
        """
        try:
            from tabulate import tabulate
        except ImportError:
            tabulate = None

        lines = []
        lines.append("=" * 78)
        lines.append("              Deep Generalized Linear Model Results")
        lines.append("=" * 78)
        lines.append(f"Family:           {self.family.capitalize()}")
        lines.append(f"Link:             {self._get_link_name()}")
        lines.append(f"SE Type:          {self.se_type}")
        lines.append(f"No. Observations: {self.n_obs:,}")
        lines.append(f"Df Residuals:     {self.df_resid:,}")
        lines.append("-" * 78)

        if self.family != "normal":
            lines.append(f"Deviance:         {self.deviance_:.4f}")
            lines.append(f"Null Deviance:    {self.null_deviance:.4f}")
            lines.append(f"Pseudo R2:        {self.pseudo_r_squared:.6f}")
        else:
            lines.append(f"R-squared:        {self.r_squared:.6f}")
            lines.append(f"Adj. R-squared:   {self.adj_r_squared:.6f}")

        if self.dispersion_ != 1.0:
            lines.append(f"Dispersion:       {self.dispersion_:.6f}")

        lines.append(f"AIC:              {self.aic:.4f}")
        lines.append(f"BIC:              {self.bic:.4f}")
        lines.append("=" * 78)
        lines.append("")

        # Coefficient table
        ci = self.confint(alpha)
        ci_level = int((1 - alpha) * 100)

        rows = []
        for i, name in enumerate(self.feature_names):
            pval = self.pvalues[i]
            pval_str = f"{pval:.4f}" if not np.isnan(pval) else "-"
            rows.append([
                name,
                f"{self.params[i]:.6f}",
                f"{self.std_errors[i]:.6f}",
                f"{self.tvalues[i]:.3f}",
                pval_str,
                f"[{ci.iloc[i]['lower']:.4f}, {ci.iloc[i]['upper']:.4f}]",
            ])

        headers = ["", "AME", "std err", "z", "P>|z|", f"[{ci_level}% CI]"]

        if tabulate is not None:
            lines.append(tabulate(rows, headers=headers, tablefmt="simple"))
        else:
            # Fallback without tabulate
            header_line = f"{'':12} {'AME':>10} {'std err':>10} {'z':>8} {'P>|z|':>8} {f'[{ci_level}% CI]':>20}"
            lines.append(header_line)
            for row in rows:
                lines.append(f"{row[0]:12} {row[1]:>10} {row[2]:>10} {row[3]:>8} {row[4]:>8} {row[5]:>20}")

        lines.append("-" * 78)

        if self.loss_history_:
            lines.append(f"Epochs trained:   {len(self.loss_history_)}")
            lines.append(f"Final NLL:        {self.loss_history_[-1]:.6f}")

        lines.append("=" * 78)
        lines.append("Note: AME = Average Marginal Effects")

        return "\n".join(lines)

    def _get_link_name(self) -> str:
        """Get link function name from family."""
        if self.family_obj_ is not None:
            return self.family_obj_.canonical_link
        return "unknown"

    def predict(
        self,
        X: Float64Array | None = None,
        type: str = "response",
    ) -> Float64Array:
        """Generate predictions.

        Parameters
        ----------
        X : array-like, optional
            Feature matrix. If None, returns fitted values.
        type : str, default="response"
            - "response": predicted mean
            - "link": linear predictor

        Returns
        -------
        ndarray
            Predictions.
        """
        if X is None:
            return self.fitted_values

        if self.network_ is None:
            raise ValueError("Network not stored.")

        import torch

        self.network_.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(np.asarray(X, dtype=np.float64)).float()
            eta = self.network_(X_tensor).squeeze()

            if type == "link":
                return eta.numpy()

            if self.family_obj_ is not None:
                mu = self.family_obj_.inverse_link(eta)
                return mu.numpy()

            return eta.numpy()
