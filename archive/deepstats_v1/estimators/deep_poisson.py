"""Deep Poisson regression with inference on rate parameter functionals.

This module implements neural network Poisson regression with inference on:
- E[lambda(X)]: Average rate parameter
- Var[lambda(X)]: Heterogeneity in rate parameter
- Quantiles of lambda(X): Distribution of rates

References
----------
- Farrell, Liang, Misra (2021). Deep Neural Networks for Estimation and Inference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .._typing import Float64Array
from ..families.poisson import Poisson
from ..networks.mlp import MLP
from ..results.poisson_results import PoissonResults

if TYPE_CHECKING:
    pass


class DeepPoisson(BaseEstimator, RegressorMixin):
    """Deep Poisson regression with inference on rate parameter distribution.

    Model: Y ~ Poisson(lambda(X)) where lambda(X) = exp(NeuralNet(X))

    This estimator provides inference on functionals of lambda(X):
    - E[lambda(X)]: Average rate with SE via sample statistics
    - Var[lambda(X)]: Heterogeneity with SE via influence function
    - Quantiles of lambda(X): Via bootstrap

    Parameters
    ----------
    hidden_dims : list[int], optional
        Hidden layer dimensions. Default is [64, 32].
    activation : str, default="relu"
        Activation function: "relu", "leaky_relu", "tanh", "elu", "gelu".
    dropout : float, default=0.1
        Dropout rate.
    epochs : int, default=500
        Number of training epochs.
    batch_size : int, default=256
        Mini-batch size.
    lr : float, default=1e-3
        Learning rate.
    weight_decay : float, default=1e-4
        L2 regularization.
    cross_fit_folds : int, default=5
        Number of folds for cross-fitting. Set to 1 to disable cross-fitting.
        Cross-fitting refits the network on each fold for unbiased predictions.
    bootstrap_se : bool, default=True
        If True, compute SEs by bootstrapping the entire procedure (refit
        model on each bootstrap sample). This captures all sources of variation
        but is more expensive. If False, use sample statistics SEs which may
        underestimate the true SE.
    bootstrap_samples : int, default=50
        Number of bootstrap samples for SE estimation (if bootstrap_se=True)
        or for quantile SEs (if bootstrap_se=False).
    quantiles : tuple[float, ...], default=(0.1, 0.25, 0.5, 0.75, 0.9)
        Quantiles of lambda(X) to estimate.
    verbose : int, default=1
        Verbosity level (0=silent, 1=progress bar, 2=epoch details).
    random_state : int, optional
        Random seed for reproducibility.
    device : str, optional
        Computation device ("cpu" or "cuda"). Auto-detected if None.

    Examples
    --------
    >>> import numpy as np
    >>> from deepstats.estimators import DeepPoisson
    >>>
    >>> # Generate data
    >>> X = np.random.randn(1000, 5)
    >>> true_lambda = np.exp(0.5 + 0.3 * X[:, 0] - 0.2 * X[:, 1])
    >>> y = np.random.poisson(true_lambda)
    >>>
    >>> # Fit model with cross-fitting
    >>> model = DeepPoisson(epochs=300, cross_fit_folds=5)
    >>> result = model.fit(X, y)
    >>>
    >>> print(result.summary())
    >>> print(f"E[lambda(X)] = {result.mean_lambda:.3f} (SE: {result.mean_lambda_se:.3f})")
    >>> print(f"Var[lambda(X)] = {result.var_lambda:.3f} (SE: {result.var_lambda_se:.3f})")

    Notes
    -----
    The inference approach follows Farrell, Liang, Misra (2021):
    - If the neural net converges fast enough (faster than n^{-1/4}), estimation
      error is second-order compared to sampling variation.
    - We treat lambda_hat(X_i) as if they were the true values and compute
      standard sample statistics.
    - Cross-fitting (K-fold refit) removes overfitting bias by using
      out-of-sample predictions for inference.

    SE Formulas:
    - E[lambda]: SE = std(lambda_hat) / sqrt(n)
    - Var[lambda]: SE via influence function of the variance estimator
    - Quantiles: Bootstrap standard errors
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        dropout: float = 0.1,
        epochs: int = 500,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        cross_fit_folds: int = 5,
        bootstrap_se: bool = True,
        bootstrap_samples: int = 50,
        quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
        verbose: int = 1,
        random_state: int | None = None,
        device: str | None = None,
    ) -> None:
        # Store hyperparameters only - no computation in __init__
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.cross_fit_folds = cross_fit_folds
        self.bootstrap_se = bootstrap_se
        self.bootstrap_samples = bootstrap_samples
        self.quantiles = quantiles
        self.verbose = verbose
        self.random_state = random_state
        self.device = device

    def _get_device(self) -> torch.device:
        """Get computation device."""
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_network(self, n_features: int) -> MLP:
        """Create a fresh MLP network."""
        hidden_dims = self.hidden_dims or [64, 32]
        return MLP(
            input_dim=n_features,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=self.activation,
            dropout=self.dropout,
        )

    def _fit_network(
        self,
        X: Float64Array,
        y: Float64Array,
        device: torch.device,
    ) -> tuple[nn.Module, list[float]]:
        """Train neural network with Poisson NLL loss.

        Parameters
        ----------
        X : Float64Array
            Features (n, p).
        y : Float64Array
            Count outcomes (n,).
        device : torch.device
            Computation device.

        Returns
        -------
        tuple
            (trained_network, loss_history)
        """
        family = Poisson()
        family.validate_response(torch.from_numpy(y))

        n_features = X.shape[1]
        network = self._create_network(n_features).to(device)

        optimizer = optim.Adam(
            network.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        X_tensor = torch.from_numpy(X).float().to(device)
        y_tensor = torch.from_numpy(y).float().to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        loss_history: list[float] = []

        iterator = range(self.epochs)
        if self.verbose >= 1:
            iterator = tqdm(iterator, desc="Training Poisson", leave=False)

        for epoch in iterator:
            network.train()
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                # Forward: network predicts eta (log-lambda)
                eta = network(batch_X).squeeze()
                mu = family.inverse_link(eta)  # lambda = exp(eta)

                # Poisson NLL loss
                loss = family.nll_loss(batch_y, mu)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)

            epoch_loss /= len(dataset)
            loss_history.append(epoch_loss)

            if self.verbose >= 2:
                print(f"Epoch {epoch + 1}/{self.epochs}, NLL: {epoch_loss:.6f}")

        return network, loss_history

    def _cross_fit_predictions(
        self,
        X: Float64Array,
        y: Float64Array,
        device: torch.device,
    ) -> tuple[Float64Array, list[list[float]], nn.Module]:
        """Get unbiased lambda predictions via K-fold cross-fitting.

        Algorithm:
        1. Split data into K folds
        2. For each fold k:
           a. Refit network on data excluding fold k
           b. Predict lambda on fold k (out-of-sample)
        3. Return all out-of-sample predictions

        This breaks dependence between fitting and prediction,
        providing unbiased estimates for inference.

        Parameters
        ----------
        X : Float64Array
            Features (n, p).
        y : Float64Array
            Count outcomes (n,).
        device : torch.device
            Computation device.

        Returns
        -------
        tuple
            (lambda_oos, loss_histories, final_network)
            - lambda_oos: Out-of-sample predictions for each observation
            - loss_histories: Training loss curves for each fold
            - final_network: Network from the last fold (for prediction)
        """
        n = len(y)
        lambda_oos = np.zeros(n)
        loss_histories: list[list[float]] = []
        final_network = None

        kf = KFold(
            n_splits=self.cross_fit_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            if self.verbose >= 1:
                print(f"Cross-fitting fold {fold_idx + 1}/{self.cross_fit_folds}")

            # Get train/test data
            X_train, y_train = X[train_idx], y[train_idx]
            X_test = X[test_idx]

            # Create and train fresh network on training fold
            network, fold_loss = self._fit_network(X_train, y_train, device)
            loss_histories.append(fold_loss)

            # Predict on held-out fold (out-of-sample)
            network.eval()
            with torch.no_grad():
                X_tensor = torch.from_numpy(X_test).float().to(device)
                eta = network(X_tensor).squeeze()
                # Clamp to prevent overflow
                eta_clamped = torch.clamp(eta, max=20)
                lambda_test = torch.exp(eta_clamped).cpu().numpy()
                lambda_oos[test_idx] = lambda_test

            final_network = network

        return lambda_oos, loss_histories, final_network

    def _compute_mean_lambda_se(
        self,
        lambda_values: Float64Array,
    ) -> tuple[float, float, Float64Array]:
        """Compute E[lambda(X)] and its standard error.

        Estimator: mu_hat = mean(lambda_hat)
        SE: std(lambda_hat) / sqrt(n)

        Note: This SE only captures sampling variance in X, not model
        estimation variance. For more accurate SEs, use bootstrap_se=True
        in the fit() method.

        Parameters
        ----------
        lambda_values : Float64Array
            Predicted lambda values.

        Returns
        -------
        tuple
            (mean_lambda, se, influence_function)
        """
        n = len(lambda_values)
        mean_lambda = float(np.mean(lambda_values))

        # Influence function for mean: IF_i = lambda_i - mean_lambda
        influence = lambda_values - mean_lambda
        se = float(np.std(lambda_values, ddof=1) / np.sqrt(n))

        return mean_lambda, se, influence

    def _compute_var_lambda_se(
        self,
        lambda_values: Float64Array,
    ) -> tuple[float, float, Float64Array]:
        """Compute Var[lambda(X)] and its standard error via influence function.

        Estimator: var_hat = var(lambda_hat)
        Influence function: IF_i = (lambda_i - mean)^2 - var_hat
        SE: std(IF) / sqrt(n)

        Parameters
        ----------
        lambda_values : Float64Array
            Predicted lambda values.

        Returns
        -------
        tuple
            (var_lambda, se, influence_function)
        """
        n = len(lambda_values)
        mean_lambda = np.mean(lambda_values)
        var_lambda = float(np.var(lambda_values, ddof=0))

        # Influence function for variance
        centered = lambda_values - mean_lambda
        influence = centered**2 - var_lambda

        # SE via influence function
        se = float(np.std(influence, ddof=1) / np.sqrt(n))

        return var_lambda, se, influence

    def _compute_quantile_se_bootstrap(
        self,
        lambda_values: Float64Array,
        quantiles: tuple[float, ...],
        n_bootstrap: int,
    ) -> tuple[dict[float, float], dict[float, float]]:
        """Compute quantiles with bootstrap standard errors.

        Parameters
        ----------
        lambda_values : Float64Array
            Predicted lambda values.
        quantiles : tuple[float, ...]
            Quantiles to compute.
        n_bootstrap : int
            Number of bootstrap samples.

        Returns
        -------
        tuple
            (quantile_values, quantile_se)
        """
        n = len(lambda_values)
        rng = np.random.default_rng(self.random_state)

        q_values: dict[float, float] = {}
        q_se: dict[float, float] = {}

        for q in quantiles:
            q_values[q] = float(np.quantile(lambda_values, q))

            # Bootstrap SE
            boot_quantiles = []
            for _ in range(n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                boot_quantiles.append(np.quantile(lambda_values[idx], q))
            q_se[q] = float(np.std(boot_quantiles, ddof=1))

        return q_values, q_se

    def _compute_all_se_bootstrap(
        self,
        X: Float64Array,
        y: Float64Array,
        lambda_values: Float64Array,
        device: torch.device,
        n_bootstrap: int = 50,
    ) -> tuple[float, float, dict[float, float]]:
        """Compute SEs via bootstrap of entire procedure.

        This captures all sources of variation:
        - Sampling variance in X
        - Model estimation variance
        - Poisson noise

        Parameters
        ----------
        X : Float64Array
            Features.
        y : Float64Array
            Counts.
        lambda_values : Float64Array
            Original predictions (for point estimates).
        device : torch.device
            Computation device.
        n_bootstrap : int
            Number of bootstrap samples.

        Returns
        -------
        tuple
            (mean_se, var_se, quantile_se)
        """
        n = len(y)
        rng = np.random.default_rng(self.random_state)

        boot_means = []
        boot_vars = []
        boot_quantiles: dict[float, list[float]] = {q: [] for q in self.quantiles}

        if self.verbose >= 1:
            print(f"Bootstrap SE estimation ({n_bootstrap} samples)...")

        for b in range(n_bootstrap):
            # Resample with replacement
            idx = rng.choice(n, size=n, replace=True)
            X_boot = X[idx]
            y_boot = y[idx]

            # Fit model on bootstrap sample
            network, _ = self._fit_network(X_boot, y_boot, device)

            # Predict on bootstrap sample
            network.eval()
            with torch.no_grad():
                X_tensor = torch.from_numpy(X_boot).float().to(device)
                eta = network(X_tensor).squeeze()
                eta_clamped = torch.clamp(eta, max=20)
                lambda_boot = torch.exp(eta_clamped).cpu().numpy()

            # Compute statistics
            boot_means.append(np.mean(lambda_boot))
            boot_vars.append(np.var(lambda_boot, ddof=0))
            for q in self.quantiles:
                boot_quantiles[q].append(np.quantile(lambda_boot, q))

        # SEs are standard deviations of bootstrap estimates
        mean_se = float(np.std(boot_means, ddof=1))
        var_se = float(np.std(boot_vars, ddof=1))
        quantile_se = {q: float(np.std(boot_quantiles[q], ddof=1)) for q in self.quantiles}

        return mean_se, var_se, quantile_se

    def fit(
        self,
        X: Float64Array,
        y: Float64Array,
    ) -> PoissonResults:
        """Fit the Poisson regression model.

        Parameters
        ----------
        X : Float64Array or pd.DataFrame
            Feature matrix (n, p).
        y : Float64Array or pd.Series
            Count outcomes (n,).

        Returns
        -------
        PoissonResults
            Results with mean_lambda, var_lambda, quantiles, and SEs.
        """
        # Set random seeds
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Handle pandas inputs
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n, p = X.shape
        device = self._get_device()

        # Get lambda predictions
        if self.cross_fit_folds > 1:
            # Cross-fitting: refit on each fold
            lambda_values, loss_histories, network = self._cross_fit_predictions(
                X, y, device
            )
            # Flatten loss history (use last fold or average)
            loss_history = loss_histories[-1] if loss_histories else []
        else:
            # Single fit
            network, loss_history = self._fit_network(X, y, device)
            network.eval()
            with torch.no_grad():
                X_tensor = torch.from_numpy(X).float().to(device)
                eta = network(X_tensor).squeeze()
                eta_clamped = torch.clamp(eta, max=20)
                lambda_values = torch.exp(eta_clamped).cpu().numpy()

        # Compute point estimates
        mean_lambda, _, influence_mean = self._compute_mean_lambda_se(lambda_values)
        var_lambda, _, influence_var = self._compute_var_lambda_se(lambda_values)
        quantiles = {q: float(np.quantile(lambda_values, q)) for q in self.quantiles}

        # Compute SEs
        if self.bootstrap_se:
            # Bootstrap entire procedure for proper SEs
            mean_lambda_se, var_lambda_se, quantile_se = self._compute_all_se_bootstrap(
                X, y, lambda_values, device, self.bootstrap_samples
            )
        else:
            # Sample statistics SEs (may underestimate)
            _, mean_lambda_se, _ = self._compute_mean_lambda_se(lambda_values)
            _, var_lambda_se, _ = self._compute_var_lambda_se(lambda_values)
            _, quantile_se = self._compute_quantile_se_bootstrap(
                lambda_values, self.quantiles, self.bootstrap_samples
            )

        # Compute deviance
        family = Poisson()
        fitted_tensor = torch.from_numpy(lambda_values).float()
        y_tensor = torch.from_numpy(y).float()
        deviance = float(family.deviance(y_tensor, fitted_tensor, 1.0).item())

        # Move network to CPU for storage
        network = network.cpu()

        return PoissonResults(
            mean_lambda=mean_lambda,
            mean_lambda_se=mean_lambda_se,
            var_lambda=var_lambda,
            var_lambda_se=var_lambda_se,
            lambda_values=lambda_values,
            quantiles=quantiles,
            quantile_se=quantile_se,
            influence_mean=influence_mean,
            influence_var=influence_var,
            n_obs=n,
            network_=network,
            loss_history_=loss_history,
            deviance_=deviance,
            cross_fit_folds=self.cross_fit_folds,
            _X=X,
            _device=device,
        )

    def predict(self, X: Float64Array) -> Float64Array:
        """Predict lambda(X) for new data.

        Note: For new data prediction, use the results object's predict() method
        after fitting, as it stores the trained network.

        Parameters
        ----------
        X : Float64Array
            Feature matrix.

        Returns
        -------
        Float64Array
            Predicted rate parameters.
        """
        raise NotImplementedError(
            "Use result.predict(X) after fitting. "
            "The trained network is stored in the results object."
        )
