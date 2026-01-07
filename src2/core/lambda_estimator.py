"""Nonparametric estimation of Lambda(x) = E[l_theta_theta | X=x]."""

import torch
from torch import Tensor
import numpy as np
from typing import Optional, Literal
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


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
    """

    def __init__(
        self,
        method: Literal['mlp', 'rf', 'ridge'] = 'mlp',
        theta_dim: int = 2,
    ):
        """
        Initialize Lambda estimator.

        Args:
            method: Regression method ('mlp', 'rf', or 'ridge')
            theta_dim: Dimension of parameter vector
        """
        self.method = method
        self.theta_dim = theta_dim
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
            return Ridge(alpha=1.0)
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
