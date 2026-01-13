"""
Regime B: Analytic Lambda for linear models.

For linear models where Hessian = TT' (doesn't depend on theta or Y):

Λ(x) = E[TT' | X = x]

This can be estimated directly by regressing TT' on X,
independently of the theta estimation.
"""

from typing import Optional, TYPE_CHECKING, Literal
import torch
from torch import Tensor

from .base import BaseLambdaStrategy

if TYPE_CHECKING:
    from deep_inference.models import StructuralModel


class AnalyticLambda(BaseLambdaStrategy):
    """
    Regime B: Analytic Lambda estimation.

    For linear models: Λ(x) = E[TT' | X=x]

    Key insight: This doesn't depend on theta at all!
    We can estimate it independently, enabling 2-way cross-fitting.
    """

    requires_theta = False  # KEY: Doesn't depend on theta!
    requires_separate_fold = False  # 2-way split sufficient

    def __init__(
        self,
        method: Literal["mlp", "rf", "ridge", "aggregate"] = "aggregate",
        ridge_alpha: float = 1.0,
    ):
        """
        Initialize AnalyticLambda strategy.

        Args:
            method: Regression method for E[TT'|X]
            ridge_alpha: Regularization for ridge regression
        """
        self.method = method
        self.ridge_alpha = ridge_alpha
        self._model = None
        self._mean_outer = None
        self._d_theta = None

    def fit(
        self,
        X: Tensor,
        T: Tensor,
        Y: Tensor,
        theta_hat: Optional[Tensor],
        model: "StructuralModel",
    ) -> None:
        """
        Fit the Lambda estimator.

        Computes E[TT' | X] by regressing outer products on X.

        Args:
            X: (n, d_x) covariates
            T: (n,) treatments
            Y: (n,) outcomes (not used)
            theta_hat: Not used (Regime B doesn't need theta)
            model: Not used (we know Hessian = TT')
        """
        n = X.shape[0]
        device = X.device
        dtype = X.dtype

        # Augment T with intercept: t_aug = [1, T]
        ones = torch.ones(n, dtype=dtype, device=device)
        T_aug = torch.stack([ones, T], dim=1)  # (n, 2)
        self._d_theta = 2

        # Compute outer products: TT' for each observation
        # Shape: (n, 2, 2)
        outer_products = torch.einsum('bi,bj->bij', T_aug, T_aug)

        if self.method == "aggregate":
            # Simple: Λ = mean(TT')
            self._mean_outer = outer_products.mean(dim=0)

        elif self.method == "mlp":
            # Fit MLP to predict outer products from X
            self._fit_mlp(X, outer_products)

        elif self.method == "ridge":
            # Fit ridge regression
            self._fit_ridge(X, outer_products)

        elif self.method == "rf":
            # Fit random forest
            self._fit_rf(X, outer_products)

    def _fit_mlp(self, X: Tensor, outer_products: Tensor) -> None:
        """Fit MLP regression for E[TT'|X]."""
        from sklearn.neural_network import MLPRegressor
        import numpy as np

        # Flatten outer products to upper triangle
        n = X.shape[0]
        d_theta = outer_products.shape[1]

        # Get upper triangle indices
        triu_idx = torch.triu_indices(d_theta, d_theta)
        n_upper = len(triu_idx[0])

        # Flatten targets
        targets = outer_products[:, triu_idx[0], triu_idx[1]]  # (n, n_upper)

        # Fit sklearn MLP
        self._mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=200,
            early_stopping=True,
        )
        self._mlp.fit(X.numpy(), targets.numpy())
        self._triu_idx = triu_idx
        self._n_upper = n_upper

    def _fit_ridge(self, X: Tensor, outer_products: Tensor) -> None:
        """Fit ridge regression for E[TT'|X]."""
        from sklearn.linear_model import Ridge
        import numpy as np

        d_theta = outer_products.shape[1]
        triu_idx = torch.triu_indices(d_theta, d_theta)
        targets = outer_products[:, triu_idx[0], triu_idx[1]]

        self._ridge = Ridge(alpha=self.ridge_alpha)
        self._ridge.fit(X.numpy(), targets.numpy())
        self._triu_idx = triu_idx

    def _fit_rf(self, X: Tensor, outer_products: Tensor) -> None:
        """Fit random forest for E[TT'|X]."""
        from sklearn.ensemble import RandomForestRegressor
        import numpy as np

        d_theta = outer_products.shape[1]
        triu_idx = torch.triu_indices(d_theta, d_theta)
        targets = outer_products[:, triu_idx[0], triu_idx[1]]

        self._rf = RandomForestRegressor(n_estimators=100, max_depth=10)
        self._rf.fit(X.numpy(), targets.numpy())
        self._triu_idx = triu_idx

    def predict(self, X: Tensor, theta_hat: Optional[Tensor] = None) -> Tensor:
        """
        Predict Λ(x) = E[TT' | X=x].

        Args:
            X: (n, d_x) covariates
            theta_hat: Not used (Regime B doesn't need theta)

        Returns:
            (n, d_theta, d_theta) Lambda matrices
        """
        n = X.shape[0]
        d_theta = self._d_theta
        device = X.device
        dtype = X.dtype

        Lambda = torch.zeros(n, d_theta, d_theta, dtype=dtype, device=device)

        if self.method == "aggregate":
            # Constant Λ for all observations
            Lambda[:] = self._mean_outer

        elif self.method == "mlp":
            pred = self._mlp.predict(X.numpy())
            pred = torch.tensor(pred, dtype=dtype, device=device)
            # Reconstruct matrices from upper triangle
            Lambda[:, self._triu_idx[0], self._triu_idx[1]] = pred
            Lambda[:, self._triu_idx[1], self._triu_idx[0]] = pred

        elif self.method == "ridge":
            pred = self._ridge.predict(X.numpy())
            pred = torch.tensor(pred, dtype=dtype, device=device)
            Lambda[:, self._triu_idx[0], self._triu_idx[1]] = pred
            Lambda[:, self._triu_idx[1], self._triu_idx[0]] = pred

        elif self.method == "rf":
            pred = self._rf.predict(X.numpy())
            pred = torch.tensor(pred, dtype=dtype, device=device)
            Lambda[:, self._triu_idx[0], self._triu_idx[1]] = pred
            Lambda[:, self._triu_idx[1], self._triu_idx[0]] = pred

        return Lambda
