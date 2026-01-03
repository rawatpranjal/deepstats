"""Deep OLS estimator.

This module implements deep learning regression with robust standard errors,
following the sklearn estimator interface.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .._typing import Float64Array
from ..inference.standard_errors import compute_vcov
from ..networks.mlp import MLP
from ..results.deep_results import DeepResults
from .base import DeepEstimatorBase


class DeepOLS(DeepEstimatorBase):
    """Deep neural network regression with statistical inference.

    This estimator fits Y = g(X) + epsilon where g is a neural network,
    and provides heteroskedasticity-robust standard errors for
    average marginal effects.

    The estimator inherits from sklearn's BaseEstimator for pipeline
    compatibility (clone, GridSearchCV, etc.).

    Parameters
    ----------
    formula : str, optional
        R-style formula like "Y ~ X1 + X2". If provided, X should be
        a DataFrame containing all variables.
    hidden_dims : list[int], optional
        Hidden layer dimensions. Default is [64, 32].
    activation : str, default="relu"
        Activation function.
    dropout : float, default=0.0
        Dropout rate.
    epochs : int, default=100
        Number of training epochs.
    batch_size : int, default=128
        Mini-batch size.
    lr : float, default=1e-3
        Learning rate.
    weight_decay : float, default=0.0
        L2 regularization weight.
    robust_se : str, default="HC1"
        Standard error type: "iid", "HC0", "HC1", "HC2", "HC3".
    random_state : int, optional
        Random seed for reproducibility.
    verbose : int, default=1
        Verbosity level (0=silent, 1=progress bar, 2=detailed).
    device : str, optional
        Device ("cpu", "cuda", "mps"). Auto-detected if None.

    Attributes
    ----------
    results_ : DeepResults
        Estimation results after fitting.
    is_fitted_ : bool
        Whether the estimator has been fitted.

    Examples
    --------
    >>> from deepstats.estimators import DeepOLS
    >>> import numpy as np
    >>>
    >>> X = np.random.randn(1000, 3)
    >>> y = 2 + 0.5*X[:, 0] - 0.3*X[:, 1] + np.random.randn(1000) * 0.5
    >>>
    >>> model = DeepOLS(epochs=100, robust_se="HC1")
    >>> result = model.fit(X, y)
    >>> print(result.summary())
    >>>
    >>> # Using formula
    >>> model = DeepOLS(formula="wage ~ education + experience")
    >>> result = model.fit(df)
    """

    def __init__(
        self,
        formula: str | None = None,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        dropout: float = 0.0,
        epochs: int = 100,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        robust_se: Literal["iid", "HC0", "HC1", "HC2", "HC3"] = "HC1",
        random_state: int | None = None,
        verbose: int = 1,
        device: str | None = None,
    ) -> None:
        # Store all parameters (sklearn requirement: no computation here)
        self.formula = formula
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.robust_se = robust_se
        self.random_state = random_state
        self.verbose = verbose
        self.device = device

    def _fit_impl(
        self,
        X: Float64Array,
        y: Float64Array,
        feature_names: list[str],
    ) -> DeepResults:
        """Implement the fitting procedure."""
        # Set random seed
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Set device
        device = self._get_device()

        # Default hidden dims
        hidden_dims = self.hidden_dims or [64, 32]

        # Create network
        n_features = X.shape[1]
        network = MLP(
            input_dim=n_features,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=self.activation,
            dropout=self.dropout,
        ).to(device)

        # Create optimizer
        optimizer = optim.Adam(
            network.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Prepare data
        X_tensor = torch.from_numpy(X).float().to(device)
        y_tensor = torch.from_numpy(y).float().to(device).unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        # Training loop
        loss_history: list[float] = []
        loss_fn = nn.MSELoss()

        iterator = range(self.epochs)
        if self.verbose >= 1:
            iterator = tqdm(iterator, desc="Training", leave=False)

        for epoch in iterator:
            network.train()
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                y_pred = network(batch_X)
                loss = loss_fn(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)

            epoch_loss /= len(dataset)
            loss_history.append(epoch_loss)

            if self.verbose >= 2:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.6f}")

        # Compute fitted values and residuals
        network.eval()
        with torch.no_grad():
            fitted_tensor = network(X_tensor)
            fitted_values = fitted_tensor.squeeze().cpu().numpy()

        residuals = y - fitted_values

        # Estimate sigma
        n, p = X.shape
        sigma = np.sqrt(np.sum(residuals**2) / (n - p))

        # Compute average marginal effects via numerical differentiation
        params = self._compute_marginal_effects(network, X, device)

        # Compute robust standard errors
        # For neural nets, we use gradient-based SE approximation
        vcov_matrix, std_errors = self._compute_standard_errors(
            X, residuals, network, device
        )

        # Move network to CPU for storage
        network = network.cpu()

        return DeepResults(
            params=params,
            std_errors=std_errors,
            vcov_matrix=vcov_matrix,
            fitted_values=fitted_values,
            residuals=residuals,
            feature_names=feature_names,
            n_obs=n,
            df_resid=n - p,
            network_=network,
            loss_history_=loss_history,
            family="normal",
            se_type=self.robust_se,
            sigma_=sigma,
            y_=y,
            X_=X,
        )

    def _get_device(self) -> torch.device:
        """Determine computation device."""
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _compute_marginal_effects(
        self,
        network: nn.Module,
        X: Float64Array,
        device: torch.device,
    ) -> Float64Array:
        """Compute average marginal effects via numerical differentiation."""
        eps = 1e-4
        n_features = X.shape[1]
        effects = np.zeros(n_features)

        network.eval()
        with torch.no_grad():
            for j in range(n_features):
                X_plus = X.copy()
                X_minus = X.copy()
                X_plus[:, j] += eps
                X_minus[:, j] -= eps

                X_plus_t = torch.from_numpy(X_plus).float().to(device)
                X_minus_t = torch.from_numpy(X_minus).float().to(device)

                y_plus = network(X_plus_t).squeeze().cpu().numpy()
                y_minus = network(X_minus_t).squeeze().cpu().numpy()

                gradient = (y_plus - y_minus) / (2 * eps)
                effects[j] = np.mean(gradient)

        return effects

    def _compute_standard_errors(
        self,
        X: Float64Array,
        residuals: Float64Array,
        network: nn.Module,
        device: torch.device,
    ) -> tuple[Float64Array, Float64Array]:
        """Compute robust standard errors for marginal effects.

        Uses the delta method combined with robust variance estimation.
        """
        # Compute Jacobian of marginal effects w.r.t. predictions
        eps = 1e-4
        n, p = X.shape

        # For simplicity, use a linear approximation:
        # Treat the marginal effects as if they were linear coefficients
        # and use standard heteroskedasticity-robust SE formulas

        # Compute "design matrix" for marginal effects
        # This is an approximation using numerical gradients
        network.eval()
        gradients = np.zeros((n, p))

        with torch.no_grad():
            for j in range(p):
                X_plus = X.copy()
                X_minus = X.copy()
                X_plus[:, j] += eps
                X_minus[:, j] -= eps

                X_plus_t = torch.from_numpy(X_plus).float().to(device)
                X_minus_t = torch.from_numpy(X_minus).float().to(device)

                y_plus = network(X_plus_t).squeeze().cpu().numpy()
                y_minus = network(X_minus_t).squeeze().cpu().numpy()

                gradients[:, j] = (y_plus - y_minus) / (2 * eps)

        # Use gradient matrix as pseudo-design matrix for SE computation
        vcov_matrix = compute_vcov(gradients, residuals, se_type=self.robust_se)
        std_errors = np.sqrt(np.diag(vcov_matrix))

        return vcov_matrix, std_errors
