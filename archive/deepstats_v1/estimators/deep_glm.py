"""Deep Generalized Linear Model estimator.

This module implements deep neural network GLM with proper MLE training
and inference via Fisher information.

References
----------
- Farrell, Liang, Misra (2021). Deep Neural Networks for Estimation and Inference.
- McCullagh & Nelder (1989). Generalized Linear Models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .._typing import Float64Array
from ..families.base import ExponentialFamily
from ..families.bernoulli import Bernoulli
from ..families.normal import Normal
from ..families.poisson import Poisson
from ..inference.standard_errors import compute_vcov
from ..networks.mlp import MLP
from ..results.glm_results import GLMResults
from .base import DeepEstimatorBase

if TYPE_CHECKING:
    pass


# Family registry
FAMILIES: dict[str, type[ExponentialFamily]] = {
    "normal": Normal,
    "gaussian": Normal,
    "poisson": Poisson,
    "bernoulli": Bernoulli,
    "binomial": Bernoulli,
}


class DeepGLM(DeepEstimatorBase):
    """Deep Generalized Linear Model with statistical inference.

    This estimator fits:
        Y ~ F(g^{-1}(network(X)), dispersion)

    where F is the exponential family distribution, g^{-1} is the inverse
    link function, and network(X) is a neural network predicting the
    linear predictor eta.

    Training uses negative log-likelihood loss (proper MLE).
    Inference uses Fisher information and sandwich estimators.

    Parameters
    ----------
    family : str or ExponentialFamily, default="normal"
        Distribution family: "normal", "poisson", "bernoulli", or
        an ExponentialFamily instance.
    formula : str, optional
        R-style formula like "Y ~ X1 + X2".
    hidden_dims : list[int], optional
        Hidden layer dimensions. Default is [64, 32].
    activation : str, default="relu"
        Activation function.
    dropout : float, default=0.1
        Dropout rate.
    epochs : int, default=100
        Number of training epochs.
    batch_size : int, default=128
        Mini-batch size.
    lr : float, default=1e-3
        Learning rate.
    weight_decay : float, default=1e-4
        L2 regularization.
    robust_se : str, default="HC1"
        Standard error type for inference.
    estimate_dispersion : bool, default=True
        Whether to estimate dispersion (for families with dispersion).
    random_state : int, optional
        Random seed.
    verbose : int, default=1
        Verbosity level.
    device : str, optional
        Computation device.

    Examples
    --------
    >>> from deepstats.estimators import DeepGLM
    >>> import numpy as np
    >>>
    >>> # Poisson regression for count data
    >>> X = np.random.randn(1000, 3)
    >>> y = np.random.poisson(np.exp(0.5 * X[:, 0] + 0.3 * X[:, 1]))
    >>>
    >>> model = DeepGLM(family="poisson", epochs=100)
    >>> result = model.fit(X, y)
    >>> print(result.summary())
    >>>
    >>> # Binary classification with proper inference
    >>> y_binary = (np.random.rand(1000) > 0.5).astype(float)
    >>> model = DeepGLM(family="bernoulli", epochs=100)
    >>> result = model.fit(X, y_binary)
    """

    def __init__(
        self,
        family: str | ExponentialFamily = "normal",
        formula: str | None = None,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        dropout: float = 0.1,
        epochs: int = 100,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        robust_se: Literal["iid", "HC0", "HC1", "HC2", "HC3"] = "HC1",
        estimate_dispersion: bool = True,
        random_state: int | None = None,
        verbose: int = 1,
        device: str | None = None,
    ) -> None:
        self.family = family
        self.formula = formula
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.robust_se = robust_se
        self.estimate_dispersion = estimate_dispersion
        self.random_state = random_state
        self.verbose = verbose
        self.device = device

    def _get_family(self) -> ExponentialFamily:
        """Resolve family parameter to ExponentialFamily instance."""
        if isinstance(self.family, ExponentialFamily):
            return self.family

        family_name = self.family.lower() if isinstance(self.family, str) else str(self.family)
        if family_name not in FAMILIES:
            raise ValueError(
                f"Unknown family: {self.family}. "
                f"Available: {list(FAMILIES.keys())}"
            )
        return FAMILIES[family_name]()

    def _fit_impl(
        self,
        X: Float64Array,
        y: Float64Array,
        feature_names: list[str],
    ) -> GLMResults:
        """Implement the GLM fitting procedure."""
        # Set random seed
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        device = self._get_device()
        family = self._get_family()

        # Validate response
        family.validate_response(torch.from_numpy(y))

        # Default hidden dims
        hidden_dims = self.hidden_dims or [64, 32]

        # Create network (predicts linear predictor eta)
        n_features = X.shape[1]
        network = MLP(
            input_dim=n_features,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=self.activation,
            dropout=self.dropout,
        ).to(device)

        # Optimizer
        optimizer = optim.Adam(
            network.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Prepare data
        X_tensor = torch.from_numpy(X).float().to(device)
        y_tensor = torch.from_numpy(y).float().to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop with NLL loss
        loss_history: list[float] = []

        iterator = range(self.epochs)
        if self.verbose >= 1:
            iterator = tqdm(iterator, desc=f"Training ({family.name})", leave=False)

        for epoch in iterator:
            network.train()
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                # Forward: network predicts eta (linear predictor)
                eta = network(batch_X).squeeze()
                mu = family.inverse_link(eta)

                # NLL loss
                loss = family.nll_loss(batch_y, mu)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)

            epoch_loss /= len(dataset)
            loss_history.append(epoch_loss)

            if self.verbose >= 2:
                print(f"Epoch {epoch + 1}/{self.epochs}, NLL: {epoch_loss:.6f}")

        # Compute fitted values
        network.eval()
        with torch.no_grad():
            eta_fitted = network(X_tensor).squeeze()
            mu_fitted = family.inverse_link(eta_fitted)
            fitted_values = mu_fitted.cpu().numpy()

        residuals = y - fitted_values

        # Estimate dispersion if needed
        n, p = X.shape
        df_resid = n - p

        if family.has_dispersion and self.estimate_dispersion:
            dispersion = family.estimate_dispersion(
                torch.from_numpy(y).float(),
                torch.from_numpy(fitted_values).float(),
                df_resid,
            )
        else:
            dispersion = 1.0

        # Compute marginal effects and their gradients
        ame, X_grad = self._compute_marginal_effects_glm(network, X, family, device)

        # Compute robust standard errors using working residuals
        vcov_matrix = self._compute_glm_standard_errors(
            X_grad, y, fitted_values, family, dispersion
        )
        std_errors = np.sqrt(np.diag(vcov_matrix))

        # Compute deviance
        deviance = family.deviance(
            torch.from_numpy(y).float(),
            torch.from_numpy(fitted_values).float(),
            dispersion,
        ).item()

        # Move network to CPU
        network = network.cpu()

        return GLMResults(
            params=ame,
            std_errors=std_errors,
            vcov_matrix=vcov_matrix,
            fitted_values=fitted_values,
            residuals=residuals,
            feature_names=feature_names,
            n_obs=n,
            df_resid=df_resid,
            network_=network,
            loss_history_=loss_history,
            family=family.name,
            se_type=self.robust_se,
            sigma_=np.sqrt(dispersion) if family.name == "normal" else 1.0,
            y_=y,
            X_=X,
            dispersion_=dispersion,
            deviance_=deviance,
            family_obj_=family,
        )

    def _compute_marginal_effects_glm(
        self,
        network: nn.Module,
        X: Float64Array,
        family: ExponentialFamily,
        device: torch.device,
    ) -> tuple[Float64Array, Float64Array]:
        """Compute average marginal effects for GLM.

        For GLM, the marginal effect of x_j on E[Y] is:
        dE[Y]/dx_j = h'(eta) * deta/dx_j

        where eta = network(X) and h = inverse_link.
        """
        eps = 1e-4
        n, p = X.shape
        gradients = np.zeros((n, p))

        network.eval()
        with torch.no_grad():
            for j in range(p):
                X_plus = X.copy()
                X_minus = X.copy()
                X_plus[:, j] += eps
                X_minus[:, j] -= eps

                X_plus_t = torch.from_numpy(X_plus).float().to(device)
                X_minus_t = torch.from_numpy(X_minus).float().to(device)

                eta_plus = network(X_plus_t).squeeze()
                eta_minus = network(X_minus_t).squeeze()

                mu_plus = family.inverse_link(eta_plus)
                mu_minus = family.inverse_link(eta_minus)

                gradients[:, j] = (mu_plus.cpu().numpy() - mu_minus.cpu().numpy()) / (2 * eps)

        # Average marginal effects
        ame = np.mean(gradients, axis=0)

        return ame, gradients

    def _compute_glm_standard_errors(
        self,
        X_grad: Float64Array,
        y: Float64Array,
        fitted_values: Float64Array,
        family: ExponentialFamily,
        dispersion: float,
    ) -> Float64Array:
        """Compute robust standard errors for GLM marginal effects.

        Uses sandwich estimator with working residuals.
        """
        # Compute working residuals
        mu_t = torch.from_numpy(fitted_values).float()
        y_t = torch.from_numpy(y).float()

        V = family.variance(mu_t).numpy()
        g_prime = family.link_derivative(mu_t).numpy()

        # Working residuals: (y - mu) / (V * g' * phi)
        working_resid = (y - fitted_values) / (V * g_prime * dispersion)

        # Use existing robust SE infrastructure
        return compute_vcov(X_grad, working_resid, se_type=self.robust_se)

    def _get_device(self) -> torch.device:
        """Determine computation device."""
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def predict(
        self,
        X: Float64Array,
        type: Literal["response", "link"] = "response",
    ) -> Float64Array:
        """Generate predictions.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        type : str, default="response"
            Type of prediction:
            - "response": predicted mean mu
            - "link": linear predictor eta

        Returns
        -------
        ndarray
            Predictions.
        """
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "results_")

        family = self._get_family()
        X_array = np.asarray(X, dtype=np.float64)

        self.results_.network_.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_array).float()
            eta = self.results_.network_(X_tensor).squeeze()

            if type == "link":
                return eta.numpy()
            else:
                mu = family.inverse_link(eta)
                return mu.numpy()
