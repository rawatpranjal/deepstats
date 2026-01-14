"""Nonparametric estimation of Lambda(x) = E[l_theta_theta | X=x]."""

import torch
from torch import Tensor
import numpy as np
from typing import Optional, Literal
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


class LambdaEstimator:
    """
    Nonparametric estimator for Lambda(x) = E[l_theta_theta | X = x].

    Estimates the conditional Hessian as a function of covariates.
    For d_theta dimensional parameters, the Hessian is a d_theta x d_theta
    symmetric matrix, so we regress the d_theta*(d_theta+1)/2 unique elements.

    Methods:
        'mlp': Multi-layer perceptron (sklearn MLPRegressor)
        'rf': Random forest (sklearn RandomForestRegressor)
        'ridge': Ridge regression (sklearn Ridge)
        'lgbm': LightGBM (fast gradient boosting)
    """

    def __init__(
        self,
        method: Literal['mlp', 'rf', 'ridge', 'lgbm'] = 'mlp',
        theta_dim: int = 2,
        ridge_alpha: float = 1.0,
    ):
        """
        Initialize Lambda estimator.

        Args:
            method: Regression method ('mlp', 'rf', or 'ridge')
            theta_dim: Dimension of parameter vector
            ridge_alpha: Regularization strength for Ridge (default 1.0)
        """
        self.method = method
        self.theta_dim = theta_dim
        self.ridge_alpha = ridge_alpha
        self.n_outputs = theta_dim * (theta_dim + 1) // 2  # Upper triangle
        self.models = None

    def _create_model(self):
        """Create a fresh regression model."""
        if self.method == 'mlp':
            return MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=42,
            )
        elif self.method == 'rf':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
        elif self.method == 'ridge':
            return Ridge(alpha=self.ridge_alpha)
        elif self.method == 'lgbm':
            if not HAS_LGBM:
                raise ImportError("LightGBM not installed. Run: pip install lightgbm")
            # Wrap in MultiOutputRegressor for multi-output support
            # Heavy regularization to ensure stable (near-PSD) Lambda estimates
            return MultiOutputRegressor(
                LGBMRegressor(
                    n_estimators=50,        # Fewer trees
                    max_depth=3,            # Shallow trees
                    learning_rate=0.05,     # Slower learning
                    min_child_samples=50,   # More samples per leaf
                    reg_alpha=1.0,          # L1 regularization
                    reg_lambda=1.0,         # L2 regularization
                    random_state=42,
                    verbose=-1,
                )
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _hessian_to_flat(self, hessians: Tensor) -> np.ndarray:
        """
        Convert (n, d, d) Hessians to (n, d*(d+1)/2) flat upper triangle.

        Args:
            hessians: (n, theta_dim, theta_dim) Hessian matrices

        Returns:
            (n, n_outputs) flattened upper triangles
        """
        n = hessians.shape[0]
        d = self.theta_dim

        # Extract upper triangle indices
        flat = np.zeros((n, self.n_outputs))
        idx = 0
        for i in range(d):
            for j in range(i, d):
                flat[:, idx] = hessians[:, i, j].numpy()
                idx += 1

        return flat

    def _flat_to_hessian(self, flat: np.ndarray) -> Tensor:
        """
        Convert (n, d*(d+1)/2) flat to (n, d, d) Hessian matrices.

        Args:
            flat: (n, n_outputs) flattened upper triangles

        Returns:
            (n, theta_dim, theta_dim) Hessian matrices
        """
        n = flat.shape[0]
        d = self.theta_dim

        hessians = torch.zeros(n, d, d)
        idx = 0
        for i in range(d):
            for j in range(i, d):
                hessians[:, i, j] = torch.tensor(flat[:, idx])
                if i != j:
                    hessians[:, j, i] = torch.tensor(flat[:, idx])  # Symmetric
                idx += 1

        return hessians

    def fit(self, X: Tensor, hessians: Tensor) -> 'LambdaEstimator':
        """
        Fit the Lambda estimator.

        Args:
            X: (n, d_x) covariates
            hessians: (n, theta_dim, theta_dim) per-observation Hessians

        Returns:
            self (fitted estimator)
        """
        X_np = X.numpy() if isinstance(X, Tensor) else X
        flat_hessians = self._hessian_to_flat(hessians)

        # Fit one model for all outputs (multi-output regression)
        self.models = self._create_model()
        self.models.fit(X_np, flat_hessians)

        return self

    def predict(self, X: Tensor) -> Tensor:
        """
        Predict Lambda(x) at new points.

        Args:
            X: (n, d_x) covariates

        Returns:
            (n, theta_dim, theta_dim) predicted Hessian matrices
        """
        if self.models is None:
            raise RuntimeError("LambdaEstimator not fitted. Call fit() first.")

        X_np = X.numpy() if isinstance(X, Tensor) else X
        flat_pred = self.models.predict(X_np)

        # Handle single prediction
        if flat_pred.ndim == 1:
            flat_pred = flat_pred.reshape(1, -1)

        return self._flat_to_hessian(flat_pred)


class AggregateLambdaEstimator:
    """
    Simple aggregate Lambda estimator (no x-dependence).

    Computes Lambda = (1/n) * sum_i l_theta_theta_i

    Used when the Hessian doesn't depend on theta (e.g., linear models).
    """

    def __init__(self, theta_dim: int = 2):
        self.theta_dim = theta_dim
        self.lambda_matrix: Optional[Tensor] = None

    def fit(self, X: Tensor, hessians: Tensor) -> 'AggregateLambdaEstimator':
        """
        Fit by computing mean Hessian.

        Args:
            X: (n, d_x) covariates (unused)
            hessians: (n, theta_dim, theta_dim) per-observation Hessians

        Returns:
            self
        """
        self.lambda_matrix = hessians.mean(dim=0)
        return self

    def predict(self, X: Tensor) -> Tensor:
        """
        Return the same aggregate Lambda for all inputs.

        Args:
            X: (n, d_x) covariates

        Returns:
            (n, theta_dim, theta_dim) replicated Lambda matrix
        """
        if self.lambda_matrix is None:
            raise RuntimeError("AggregateLambdaEstimator not fitted.")

        n = X.shape[0]
        return self.lambda_matrix.unsqueeze(0).expand(n, -1, -1).clone()


class PropensityWeightedLambdaEstimator:
    """
    Propensity-weighted Lambda estimator for binary treatment.

    Computes Λ(x) = E[ℓ_θθ | X=x] by:
    1. Estimating propensity e(x) = P(T=1 | X=x)
    2. Computing arm-specific means: Λ₀ = E[ℓ_θθ | T=0], Λ₁ = E[ℓ_θθ | T=1]
    3. Combining: Λ(x) = (1 - e(x))·Λ₀ + e(x)·Λ₁

    This ensures full-rank Lambda even when individual Hessians are singular
    (as happens with binary T in Logit models).

    Note: For randomized treatment with e(x) ≈ 0.5 for all x, this is
    equivalent to AggregateLambdaEstimator.
    """

    def __init__(self, theta_dim: int = 2):
        self.theta_dim = theta_dim
        self.propensity_model = None
        self.Lambda0_mean: Optional[Tensor] = None
        self.Lambda1_mean: Optional[Tensor] = None

    def fit(
        self,
        X: Tensor,
        T: Tensor,
        hessians: Tensor
    ) -> 'PropensityWeightedLambdaEstimator':
        """
        Fit propensity model and compute arm-specific Lambda means.

        Args:
            X: (n, d_x) covariates
            T: (n,) binary treatment (0 or 1)
            hessians: (n, theta_dim, theta_dim) per-observation Hessians

        Returns:
            self (fitted estimator)
        """
        from sklearn.linear_model import LogisticRegression

        X_np = X.numpy() if isinstance(X, Tensor) else X
        T_np = T.numpy() if isinstance(T, Tensor) else T

        # Fit propensity model
        self.propensity_model = LogisticRegression(max_iter=1000)
        self.propensity_model.fit(X_np, T_np)

        # Separate by treatment arm
        T0_mask = T_np == 0
        T1_mask = T_np == 1

        # Compute arm-specific means
        if isinstance(hessians, Tensor):
            self.Lambda0_mean = hessians[T0_mask].mean(dim=0)
            self.Lambda1_mean = hessians[T1_mask].mean(dim=0)
        else:
            self.Lambda0_mean = torch.tensor(np.mean(hessians[T0_mask], axis=0))
            self.Lambda1_mean = torch.tensor(np.mean(hessians[T1_mask], axis=0))

        return self

    def predict(self, X: Tensor) -> Tensor:
        """
        Predict Λ(x) using propensity-weighted combination.

        Args:
            X: (n, d_x) covariates

        Returns:
            (n, theta_dim, theta_dim) predicted Hessian matrices
        """
        if self.propensity_model is None:
            raise RuntimeError("PropensityWeightedLambdaEstimator not fitted.")

        X_np = X.numpy() if isinstance(X, Tensor) else X

        # Get propensity scores
        e_x = self.propensity_model.predict_proba(X_np)[:, 1]

        # Weighted combination
        n = len(X_np)
        Lambda = torch.zeros(n, self.theta_dim, self.theta_dim)

        for i in range(n):
            Lambda[i] = (1 - e_x[i]) * self.Lambda0_mean + e_x[i] * self.Lambda1_mean

        return Lambda
