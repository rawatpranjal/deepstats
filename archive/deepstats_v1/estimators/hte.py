"""Heterogeneous Treatment Effects estimator via enriched structural models.

This module implements the DeepHTE estimator following Farrell, Liang, Misra
(2021, 2023) for estimating heterogeneous treatment effects with neural networks.

Model Specification
-------------------
Continuous: Y = a(X) + b(X) * T + ε
Binary:     P(Y=1) = sigmoid(a(X) + b(X) * T)

Where a(X) and b(X) are neural network outputs (parameter functions).

Inference Targets
-----------------
- E[b(X)]: Average Treatment Effect (ATE)
- Quantiles of b(X): Heterogeneity analysis
- b(X_i): Individual Treatment Effects (ITE)

References
----------
- Farrell, Liang, Misra (2021). "Deep Neural Networks for Estimation and Inference"
- Farrell, Liang, Misra (2023). "Deep Learning for Individual Heterogeneity"
- MisraLab course: github.com/MisraLab/cml.github.io
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .._typing import Float64Array
from ..families.base import ExponentialFamily
from ..families.normal import Normal
from ..families.bernoulli import Bernoulli
from ..families.poisson import Poisson
from ..families.gamma import Gamma
from ..families.exponential import Exponential
from ..formula.parser import FormulaParser, ParsedFormula
from ..networks.base import ParameterNetwork
from ..networks.registry import ArchitectureRegistry

if TYPE_CHECKING:
    pass

# Family registry mapping names to classes
FAMILIES: dict[str, type[ExponentialFamily]] = {
    "normal": Normal,
    "gaussian": Normal,
    "bernoulli": Bernoulli,
    "binary": Bernoulli,
    "poisson": Poisson,
    "gamma": Gamma,
    "exponential": Exponential,
}


@dataclass
class HTEResults:
    """Results from DeepHTE estimation.

    Attributes
    ----------
    ate : float
        Average Treatment Effect: E[b(X)]
    ate_se : float
        Standard error of ATE
    ite : Float64Array
        Individual Treatment Effects: b(X_i) for each observation
    ite_se : Float64Array
        Standard errors of ITEs (if computed)
    baseline : Float64Array
        Baseline predictions: a(X_i)
    quantiles : dict[float, float]
        Quantiles of b(X) distribution
    quantile_se : dict[float, float]
        Standard errors of quantiles
    influence_function : Float64Array
        Influence function values for each observation
    network_ : nn.Module
        Trained parameter network
    family : str
        Distribution family used
    n_obs : int
        Number of observations
    formula : str
        Formula used for estimation
    """

    # Point estimates
    ate: float
    ate_se: float

    # Individual effects
    ite: Float64Array
    ite_se: Float64Array | None = None

    # Baseline
    baseline: Float64Array = field(default_factory=lambda: np.array([]))

    # Heterogeneity
    quantiles: dict[float, float] = field(default_factory=dict)
    quantile_se: dict[float, float] = field(default_factory=dict)

    # Inference
    influence_function: Float64Array = field(default_factory=lambda: np.array([]))

    # Model components
    network_: nn.Module | None = None
    family: str = "normal"
    n_obs: int = 0
    formula: str = ""
    loss_history_: list[float] = field(default_factory=list)

    # Loss histories for diagnostics
    train_loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)

    # Private
    _y: Float64Array = field(default_factory=lambda: np.array([]), repr=False)
    _t: Float64Array = field(default_factory=lambda: np.array([]), repr=False)

    def summary(self, alpha: float = 0.05) -> str:
        """Generate summary table.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for confidence intervals.

        Returns
        -------
        str
            Formatted summary table.
        """
        from scipy import stats

        z = stats.norm.ppf(1 - alpha / 2)
        ci_lower = self.ate - z * self.ate_se
        ci_upper = self.ate + z * self.ate_se
        z_stat = self.ate / self.ate_se if self.ate_se > 0 else np.inf
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        lines = [
            "=" * 60,
            f"Heterogeneous Treatment Effects ({self.family.capitalize()} Family)",
            "=" * 60,
            "",
            "Average Treatment Effect (ATE)",
            "-" * 60,
            f"{'Estimate':<12} {'Std.Err':<10} {'z-value':<10} {'P>|z|':<8} [95% CI]",
            f"{self.ate:<12.4f} {self.ate_se:<10.4f} {z_stat:<10.2f} {p_value:<8.4f} [{ci_lower:.3f}, {ci_upper:.3f}]",
            "",
            "Treatment Effect Heterogeneity",
            "-" * 60,
            f"{'Quantile':<12} {'Estimate':<12} {'Std.Err':<10}",
        ]

        for q in sorted(self.quantiles.keys()):
            q_val = self.quantiles[q]
            q_se = self.quantile_se.get(q, np.nan)
            lines.append(f"{int(q*100):>3}%{'':<8} {q_val:<12.4f} {q_se:<10.4f}")

        lines.extend([
            "",
            f"Observations: {self.n_obs}",
            f"Family: {self.family}",
            f"ITE Range: [{self.ite.min():.3f}, {self.ite.max():.3f}]",
            "=" * 60,
        ])

        return "\n".join(lines)

    def ate_confint(self, alpha: float = 0.05) -> tuple[float, float]:
        """Compute confidence interval for ATE.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level (0.05 gives 95% CI).

        Returns
        -------
        tuple[float, float]
            (lower, upper) bounds of confidence interval.
        """
        from scipy import stats

        z = stats.norm.ppf(1 - alpha / 2)
        ci_lower = self.ate - z * self.ate_se
        ci_upper = self.ate + z * self.ate_se
        return ci_lower, ci_upper

    def predict(self, X: np.ndarray, T: np.ndarray | None = None) -> Float64Array:
        """Predict outcomes for new data.

        For families with non-identity link (e.g., bernoulli, poisson, gamma),
        returns predictions on the response scale (after applying inverse link).

        Parameters
        ----------
        X : np.ndarray
            Covariates of shape (n, p).
        T : np.ndarray, optional
            Treatment values. If None, uses T=1.

        Returns
        -------
        Float64Array
            Predicted outcomes on response scale.
        """
        if self.network_ is None:
            raise RuntimeError("Model not fitted yet")

        self.network_.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            params = self.network_(X_tensor)
            a = params[:, 0].numpy()
            b = params[:, 1].numpy()

        if T is None:
            T = np.ones(len(X))

        # Linear predictor
        eta = a + b * T

        # Apply inverse link based on family
        if self.family in ("normal", "gaussian"):
            return eta  # Identity link
        elif self.family in ("bernoulli", "binary"):
            return 1 / (1 + np.exp(-eta))  # Sigmoid
        elif self.family == "poisson":
            return np.exp(np.clip(eta, -20, 20))  # Exp link
        elif self.family in ("gamma", "exponential"):
            return np.exp(np.clip(eta, -20, 20))  # Log link (default)
        else:
            return eta  # Default to identity

    def cate(self, X: np.ndarray) -> Float64Array:
        """Get Conditional Average Treatment Effect for new X.

        Parameters
        ----------
        X : np.ndarray
            Covariates of shape (n, p).

        Returns
        -------
        Float64Array
            CATE predictions b(X).
        """
        if self.network_ is None:
            raise RuntimeError("Model not fitted yet")

        self.network_.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            params = self.network_(X_tensor)
            b = params[:, 1].numpy()

        return b


class DeepHTE(BaseEstimator, RegressorMixin):
    """Heterogeneous Treatment Effects via enriched structural models.

    Estimates heterogeneous treatment effects using neural networks
    following Farrell, Liang, Misra (2021, 2023).

    Model Specification:
        Continuous: Y = a(X) + b(X) * T + ε
        Binary:     P(Y=1) = sigmoid(a(X) + b(X) * T)

    The network outputs parameter functions [a(X), b(X)] which are then
    used in the structural loss (NLL for the specified family).

    Parameters
    ----------
    formula : str
        Formula specifying the model: "Y ~ a(X1 + X2) + b(X1 + X2) * T"
    family : str, default="normal"
        Distribution family: "normal" or "bernoulli"
    backbone : str, default="mlp"
        Network architecture: "mlp", "transformer", or "lstm"
    hidden_dims : list[int], optional
        Hidden layer dimensions for backbone. Default depends on architecture.
    epochs : int, default=1000
        Number of training epochs.
    lr : float, default=0.01
        Learning rate for Adam optimizer.
    batch_size : int, default=256
        Batch size for training.
    weight_decay : float, default=1e-4
        L2 regularization strength.
    dropout : float, default=0.1
        Dropout rate.
    cross_fit_folds : int, default=5
        Number of folds for cross-fitted standard error estimation.
        Set to 1 to disable cross-fitting (not recommended).
    verbose : int, default=1
        Verbosity level (0=silent, 1=progress, 2=detailed).
    random_state : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> import deepstats as ds
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Generate A/B test data
    >>> n = 1000
    >>> data = pd.DataFrame({
    ...     'Y': np.random.randn(n),
    ...     'X1': np.random.randn(n),
    ...     'X2': np.random.randn(n),
    ...     'T': np.random.binomial(1, 0.5, n)
    ... })
    >>>
    >>> model = ds.DeepHTE(
    ...     formula="Y ~ a(X1 + X2) + b(X1 + X2) * T",
    ...     family="normal",
    ...     backbone="mlp",
    ...     epochs=500
    ... )
    >>> result = model.fit(data)
    >>> print(result.summary())

    References
    ----------
    - Farrell, Liang, Misra (2021). "Deep Neural Networks for Estimation and Inference"
    - Farrell, Liang, Misra (2023). "Deep Learning for Individual Heterogeneity"
    """

    def __init__(
        self,
        formula: str | None = None,
        family: str | ExponentialFamily = "normal",
        backbone: Literal["mlp", "transformer", "lstm", "cnn", "resnet", "gnn", "text", "bow"] = "mlp",
        hidden_dims: list[int] | None = None,
        epochs: int = 1000,
        lr: float = 0.01,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        cross_fit_folds: int = 5,
        verbose: int = 1,
        random_state: int | None = None,
        device: str | None = None,
        # Scaling parameters
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        gradient_accumulation: int = 1,
    ) -> None:
        self.formula = formula
        self.family = family
        self.backbone = backbone
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.cross_fit_folds = cross_fit_folds
        self.verbose = verbose
        self.random_state = random_state
        self.device = device
        # Scaling parameters
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.gradient_accumulation = gradient_accumulation

    def fit(
        self,
        data: pd.DataFrame,
        validation_data: pd.DataFrame | None = None,
        validation_split: float = 0.0,
    ) -> HTEResults:
        """Fit the heterogeneous treatment effects model.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing outcome, treatment, and covariates.
        validation_data : pd.DataFrame, optional
            Separate validation data. If provided, used for validation loss.
        validation_split : float, default=0.0
            Fraction of data to use for validation (0-1). Only used if
            validation_data is not provided.

        Returns
        -------
        HTEResults
            Results object with ATE, ITEs, and inference.
        """
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        # Get device
        device = self._get_device()

        # Parse formula and extract data
        parser = FormulaParser()
        parsed = parser.parse(self.formula, data)

        # Get dimensions
        # Use all covariates (union of a and b covariates)
        all_covariates = list(set(parsed.a_covariates) | set(parsed.b_covariates))
        X = data[all_covariates].values.astype(np.float64)
        y = parsed.y
        t = parsed.t
        n, p = X.shape

        # Handle validation data
        X_val, y_val, t_val = None, None, None
        if validation_data is not None:
            parsed_val = parser.parse(self.formula, validation_data)
            X_val = validation_data[all_covariates].values.astype(np.float64)
            y_val = parsed_val.y
            t_val = parsed_val.t
        elif validation_split > 0:
            # Split data
            rng = np.random.default_rng(self.random_state)
            val_idx = rng.random(n) < validation_split
            X_val = X[val_idx]
            y_val = y[val_idx]
            t_val = t[val_idx]
            X = X[~val_idx]
            y = y[~val_idx]
            t = t[~val_idx]
            n = len(y)

        # Create network
        backbone_config = self._get_backbone_config(p)
        backbone = ArchitectureRegistry.create(
            self.backbone,
            input_dim=p,
            **backbone_config
        )

        # Parameter network: backbone + parameter layer [a(X), b(X)]
        network = ParameterNetwork(
            backbone=backbone,
            param_dim=2,
            param_names=["a", "b"],
        ).to(device)

        # Prepare training data - keep on CPU for scalability
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        t_tensor = torch.tensor(t, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor, t_tensor)

        # Configure DataLoader for scaling
        use_pin_memory = self.pin_memory and device.type == "cuda"
        dataloader_kwargs = {
            "batch_size": min(self.batch_size, n),
            "shuffle": True,
            "num_workers": self.num_workers,
            "pin_memory": use_pin_memory,
        }
        # Only add prefetch_factor if using workers
        if self.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = self.prefetch_factor
            dataloader_kwargs["persistent_workers"] = True

        dataloader = DataLoader(dataset, **dataloader_kwargs)

        # Prepare validation data - keep on CPU for batched evaluation
        X_val_tensor, y_val_tensor, t_val_tensor = None, None, None
        val_dataloader = None
        if X_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
            t_val_tensor = torch.tensor(t_val, dtype=torch.float32)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor, t_val_tensor)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=min(self.batch_size, len(X_val)),
                shuffle=False,
                num_workers=0,  # Validation doesn't need workers
                pin_memory=use_pin_memory,
            )

        # Optimizer
        optimizer = optim.Adam(
            network.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Training loop
        train_loss_history = []
        val_loss_history = []
        iterator = range(self.epochs)
        if self.verbose > 0:
            iterator = tqdm(iterator, desc="Training DeepHTE")

        network.train()
        for epoch in iterator:
            epoch_loss = 0.0
            n_batches = 0
            optimizer.zero_grad()

            for batch_idx, (batch_X, batch_y, batch_t) in enumerate(dataloader):
                # Async transfer to device
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                batch_t = batch_t.to(device, non_blocking=True)

                # Forward pass: get parameter functions
                params = network(batch_X)
                a = params[:, 0]
                b = params[:, 1]

                # Compute structural loss (scaled for gradient accumulation)
                loss = self._compute_loss(batch_y, batch_t, a, b)
                loss = loss / self.gradient_accumulation

                # Backward pass
                loss.backward()

                # Step optimizer after accumulating gradients
                if (batch_idx + 1) % self.gradient_accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * self.gradient_accumulation
                n_batches += 1

            # Handle remaining gradients if batches don't divide evenly
            if n_batches % self.gradient_accumulation != 0:
                optimizer.step()
                optimizer.zero_grad()

            avg_train_loss = epoch_loss / n_batches
            train_loss_history.append(avg_train_loss)

            # Compute validation loss using batched evaluation
            if val_dataloader is not None:
                network.eval()
                val_loss_sum = 0.0
                n_val = 0
                with torch.no_grad():
                    for vX, vy, vt in val_dataloader:
                        vX = vX.to(device, non_blocking=True)
                        vy = vy.to(device, non_blocking=True)
                        vt = vt.to(device, non_blocking=True)
                        params_val = network(vX)
                        a_val = params_val[:, 0]
                        b_val = params_val[:, 1]
                        batch_val_loss = self._compute_loss(vy, vt, a_val, b_val)
                        val_loss_sum += batch_val_loss.item() * len(vX)
                        n_val += len(vX)
                val_loss_history.append(val_loss_sum / n_val)
                network.train()

            if self.verbose > 0 and hasattr(iterator, "set_postfix"):
                postfix = {"train_loss": f"{avg_train_loss:.4f}"}
                if val_loss_history:
                    postfix["val_loss"] = f"{val_loss_history[-1]:.4f}"
                iterator.set_postfix(postfix)

        # Get final predictions (batched for large datasets)
        network.eval()
        a_list, b_list = [], []
        pred_dataloader = DataLoader(
            TensorDataset(X_tensor),
            batch_size=self.batch_size,
            shuffle=False,
        )
        with torch.no_grad():
            for (batch_X,) in pred_dataloader:
                batch_X = batch_X.to(device, non_blocking=True)
                params = network(batch_X)
                a_list.append(params[:, 0].cpu().numpy())
                b_list.append(params[:, 1].cpu().numpy())
        a = np.concatenate(a_list)
        b = np.concatenate(b_list)

        # Compute ATE and inference (use cross-fitting for valid SE)
        if self.cross_fit_folds > 1:
            ate, ate_se, influence_fn = self._compute_dr_ate_crossfit(
                X, y, t, network, device, self.cross_fit_folds
            )
        else:
            ate, ate_se, influence_fn = self._compute_dr_ate(y, t, a, b)

        # Compute quantiles with bootstrap SE
        quantiles, quantile_se = self._compute_quantiles(b)

        # Create results
        results = HTEResults(
            ate=ate,
            ate_se=ate_se,
            ite=b,
            baseline=a,
            quantiles=quantiles,
            quantile_se=quantile_se,
            influence_function=influence_fn,
            network_=network.cpu(),
            family=self._get_family_name(),
            n_obs=n,
            formula=self.formula,
            loss_history_=train_loss_history,
            train_loss_history=train_loss_history,
            val_loss_history=val_loss_history,
            _y=y,
            _t=t,
        )

        return results

    def fit_raw(
        self,
        X: torch.Tensor | np.ndarray | list,
        y: np.ndarray,
        t: np.ndarray,
        edge_index: torch.Tensor | np.ndarray | None = None,
        batch: torch.Tensor | np.ndarray | None = None,
        **kwargs,
    ) -> HTEResults:
        """Fit the model on raw (non-tabular) data.

        Use this method for image, text, or graph inputs where the backbone
        processes raw data directly instead of tabular features.

        Parameters
        ----------
        X : torch.Tensor | np.ndarray | list
            Raw input data:
            - Images: tensor of shape (n, C, H, W)
            - Text: tensor of shape (n, seq_len) with token indices
            - Graph: list of node feature tensors (one per graph)
        y : np.ndarray
            Outcome variable of shape (n,).
        t : np.ndarray
            Treatment indicator of shape (n,).
        edge_index : torch.Tensor | np.ndarray, optional
            Edge indices for graph data. Can be:
            - Single tensor of shape (2, num_edges) for one graph
            - List of tensors for multiple graphs
        batch : torch.Tensor | np.ndarray, optional
            Batch assignment for graph nodes.
        **kwargs
            Additional backbone config (in_channels, image_size, vocab_size, etc.)

        Returns
        -------
        HTEResults
            Results object with ATE, ITEs, and inference.

        Examples
        --------
        >>> # Image data
        >>> model = DeepHTE(backbone="cnn", epochs=500)
        >>> result = model.fit_raw(images, y, t, in_channels=3, image_size=32)

        >>> # Graph data
        >>> model = DeepHTE(backbone="gnn", epochs=500)
        >>> result = model.fit_raw(node_features, y, t, edge_index=edges)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        device = self._get_device()
        n = len(y)
        y = np.asarray(y, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)

        # Convert inputs to tensors
        if isinstance(X, np.ndarray):
            if self.backbone in ("text", "bow"):
                # Text tokens should be long integers
                X = torch.tensor(X, dtype=torch.long)
            else:
                X = torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, list):
            # For graphs: list of node features
            X = [torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x for x in X]

        # Handle edge_index for graphs
        if edge_index is not None:
            if isinstance(edge_index, np.ndarray):
                edge_index = torch.tensor(edge_index, dtype=torch.long)
            elif isinstance(edge_index, list):
                edge_index = [torch.tensor(e, dtype=torch.long) if isinstance(e, np.ndarray) else e for e in edge_index]

        # Determine input dimension and create backbone
        if self.backbone in ("cnn", "resnet"):
            # Image backbone - input_dim from image shape
            if isinstance(X, torch.Tensor) and X.dim() == 4:
                in_channels = X.shape[1]
                image_size = X.shape[2]
                kwargs.setdefault("in_channels", in_channels)
                kwargs.setdefault("image_size", image_size)
            input_dim = kwargs.get("in_channels", 3) * kwargs.get("image_size", 32) ** 2

        elif self.backbone == "gnn":
            # Graph backbone - input_dim from node features
            if isinstance(X, list) and len(X) > 0:
                input_dim = X[0].shape[-1]
            elif isinstance(X, torch.Tensor):
                input_dim = X.shape[-1]
            else:
                input_dim = kwargs.get("input_dim", 16)

        elif self.backbone in ("text", "bow"):
            # Text backbone - input_dim is vocab_size
            input_dim = kwargs.get("vocab_size", 10000)

        else:
            # Tabular backbones
            if isinstance(X, torch.Tensor):
                input_dim = X.shape[-1]
            else:
                raise ValueError(f"Unknown input format for backbone {self.backbone}")

        # Create network
        backbone_config = self._get_backbone_config(input_dim, **kwargs)
        backbone = ArchitectureRegistry.create(
            self.backbone,
            input_dim=input_dim,
            **backbone_config
        )

        network = ParameterNetwork(
            backbone=backbone,
            param_dim=2,
            param_names=["a", "b"],
        ).to(device)

        # Prepare outcome and treatment
        y_tensor = torch.tensor(y, dtype=torch.float32)
        t_tensor = torch.tensor(t, dtype=torch.float32)

        # Optimizer
        optimizer = optim.Adam(
            network.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Training loop
        train_loss_history = []
        iterator = range(self.epochs)
        if self.verbose > 0:
            iterator = tqdm(iterator, desc=f"Training DeepHTE ({self.backbone})")

        network.train()

        # Handle different data types
        if self.backbone in ("cnn", "resnet"):
            # Image training loop
            dataset = TensorDataset(X, y_tensor, t_tensor)
            dataloader = DataLoader(
                dataset,
                batch_size=min(self.batch_size, n),
                shuffle=True,
            )

            for epoch in iterator:
                epoch_loss = 0.0
                n_batches = 0
                for batch_X, batch_y, batch_t in dataloader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    batch_t = batch_t.to(device)

                    optimizer.zero_grad()
                    params = network(batch_X)
                    a, b = params[:, 0], params[:, 1]
                    loss = self._compute_loss(batch_y, batch_t, a, b)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                train_loss_history.append(epoch_loss / n_batches)
                if self.verbose > 0 and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix({"loss": f"{train_loss_history[-1]:.4f}"})

        elif self.backbone == "gnn":
            # Graph training loop (mini-batch over graphs)
            # Combine all graphs into batched format
            if isinstance(X, list):
                all_node_features, all_edge_index, batch_assign = self._batch_graphs(X, edge_index)
            else:
                all_node_features = X
                all_edge_index = edge_index
                batch_assign = batch if batch is not None else torch.zeros(X.shape[0], dtype=torch.long)

            all_node_features = all_node_features.to(device)
            all_edge_index = all_edge_index.to(device)
            batch_assign = batch_assign.to(device)
            y_tensor = y_tensor.to(device)
            t_tensor = t_tensor.to(device)

            for epoch in iterator:
                optimizer.zero_grad()

                # Forward through GNN backbone
                graph_features = backbone(all_node_features, all_edge_index, batch_assign)
                params = network.param_layer(graph_features)
                a, b = params[:, 0], params[:, 1]

                loss = self._compute_loss(y_tensor, t_tensor, a, b)
                loss.backward()
                optimizer.step()

                train_loss_history.append(loss.item())
                if self.verbose > 0 and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix({"loss": f"{loss.item():.4f}"})

        elif self.backbone in ("text", "bow"):
            # Text training loop
            dataset = TensorDataset(X, y_tensor, t_tensor)
            dataloader = DataLoader(
                dataset,
                batch_size=min(self.batch_size, n),
                shuffle=True,
            )

            for epoch in iterator:
                epoch_loss = 0.0
                n_batches = 0
                for batch_X, batch_y, batch_t in dataloader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    batch_t = batch_t.to(device)

                    optimizer.zero_grad()
                    params = network(batch_X)
                    a, b = params[:, 0], params[:, 1]
                    loss = self._compute_loss(batch_y, batch_t, a, b)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                train_loss_history.append(epoch_loss / n_batches)
                if self.verbose > 0 and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix({"loss": f"{train_loss_history[-1]:.4f}"})

        else:
            # Tabular training loop (same as fit())
            dataset = TensorDataset(X, y_tensor, t_tensor)
            dataloader = DataLoader(
                dataset,
                batch_size=min(self.batch_size, n),
                shuffle=True,
            )

            for epoch in iterator:
                epoch_loss = 0.0
                n_batches = 0
                for batch_X, batch_y, batch_t in dataloader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    batch_t = batch_t.to(device)

                    optimizer.zero_grad()
                    params = network(batch_X)
                    a, b = params[:, 0], params[:, 1]
                    loss = self._compute_loss(batch_y, batch_t, a, b)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                train_loss_history.append(epoch_loss / n_batches)
                if self.verbose > 0 and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix({"loss": f"{train_loss_history[-1]:.4f}"})

        # Get final predictions
        network.eval()
        with torch.no_grad():
            if self.backbone in ("cnn", "resnet", "text", "bow"):
                # Batched prediction for image/text
                a_list, b_list = [], []
                pred_dataloader = DataLoader(
                    TensorDataset(X),
                    batch_size=self.batch_size,
                    shuffle=False,
                )
                for (batch_X,) in pred_dataloader:
                    batch_X = batch_X.to(device)
                    params = network(batch_X)
                    a_list.append(params[:, 0].cpu().numpy())
                    b_list.append(params[:, 1].cpu().numpy())
                a = np.concatenate(a_list)
                b = np.concatenate(b_list)

            elif self.backbone == "gnn":
                # Graph prediction
                graph_features = backbone(all_node_features, all_edge_index, batch_assign)
                params = network.param_layer(graph_features)
                a = params[:, 0].cpu().numpy()
                b = params[:, 1].cpu().numpy()

            else:
                # Tabular prediction
                X_device = X.to(device)
                params = network(X_device)
                a = params[:, 0].cpu().numpy()
                b = params[:, 1].cpu().numpy()

        # Compute ATE using doubly robust estimator
        ate, ate_se, influence_fn = self._compute_dr_ate(y, t, a, b)

        # Compute quantiles
        quantiles, quantile_se = self._compute_quantiles(b)

        # Create results
        results = HTEResults(
            ate=ate,
            ate_se=ate_se,
            ite=b,
            baseline=a,
            quantiles=quantiles,
            quantile_se=quantile_se,
            influence_function=influence_fn,
            network_=network.cpu(),
            family=self._get_family_name(),
            n_obs=n,
            formula=self.formula or f"raw_{self.backbone}",
            loss_history_=train_loss_history,
            train_loss_history=train_loss_history,
            val_loss_history=[],
            _y=y,
            _t=t,
        )

        return results

    def _batch_graphs(
        self,
        node_features_list: list[torch.Tensor],
        edge_index_list: list[torch.Tensor] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch multiple graphs into a single graph.

        Parameters
        ----------
        node_features_list : list[torch.Tensor]
            List of node features, one tensor per graph.
        edge_index_list : list[torch.Tensor] | torch.Tensor
            List of edge indices, one per graph.

        Returns
        -------
        tuple
            (batched_node_features, batched_edge_index, batch_assignment)
        """
        all_nodes = []
        all_edges = []
        batch_assign = []
        node_offset = 0

        if isinstance(edge_index_list, torch.Tensor):
            # Single edge_index for all - assume single graph per sample
            edge_index_list = [edge_index_list] * len(node_features_list)

        for i, (nodes, edges) in enumerate(zip(node_features_list, edge_index_list)):
            all_nodes.append(nodes)
            # Offset edge indices
            all_edges.append(edges + node_offset)
            batch_assign.extend([i] * nodes.shape[0])
            node_offset += nodes.shape[0]

        batched_nodes = torch.cat(all_nodes, dim=0)
        batched_edges = torch.cat(all_edges, dim=1)
        batch_tensor = torch.tensor(batch_assign, dtype=torch.long)

        return batched_nodes, batched_edges, batch_tensor

    def _get_device(self) -> torch.device:
        """Get compute device."""
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_backbone_config(self, input_dim: int, **kwargs) -> dict:
        """Get configuration for backbone network.

        Parameters
        ----------
        input_dim : int
            Input dimension (for tabular) or ignored for non-tabular.
        **kwargs
            Additional backbone-specific config (in_channels, image_size, etc.)
        """
        config = {"dropout": self.dropout}

        if self.hidden_dims is not None:
            config["hidden_dims"] = self.hidden_dims

        # Architecture-specific defaults
        if self.backbone == "transformer":
            config.setdefault("d_model", 64)
            config.setdefault("num_heads", 4)
            config.setdefault("num_layers", 2)
        elif self.backbone == "lstm":
            config.setdefault("hidden_size", 64)
            config.setdefault("num_layers", 2)
        elif self.backbone in ("cnn", "resnet"):
            # Image backbone config
            config["in_channels"] = kwargs.get("in_channels", 3)
            config["image_size"] = kwargs.get("image_size", 32)
            config["output_dim"] = kwargs.get("output_dim", 64)
            config.setdefault("hidden_channels", [32, 64, 128])
        elif self.backbone == "gnn":
            # Graph backbone config
            config["hidden_dim"] = kwargs.get("hidden_dim", 64)
            config["output_dim"] = kwargs.get("output_dim", 64)
            config["num_layers"] = kwargs.get("num_layers", 3)
        elif self.backbone in ("text", "bow"):
            # Text backbone config
            config["vocab_size"] = kwargs.get("vocab_size", 10000)
            config["embed_dim"] = kwargs.get("embed_dim", 128)
            config["output_dim"] = kwargs.get("output_dim", 64)
        else:  # mlp
            config.setdefault("hidden_dims", [64, 32])

        return config

    def _get_family(self) -> ExponentialFamily:
        """Get family instance.

        Returns
        -------
        ExponentialFamily
            Family instance for computing loss and predictions.
        """
        if isinstance(self.family, ExponentialFamily):
            return self.family
        if self.family not in FAMILIES:
            raise ValueError(
                f"Unknown family: {self.family}. "
                f"Available: {list(FAMILIES.keys())}"
            )
        return FAMILIES[self.family]()

    def _get_family_name(self) -> str:
        """Get family name for results."""
        if isinstance(self.family, ExponentialFamily):
            return self.family.name
        return self.family

    def _compute_loss(
        self,
        y: torch.Tensor,
        t: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute structural loss using proper NLL for the family.

        The enriched structural model is:
            Y ~ Family(eta), where eta = a(X) + b(X) * T

        Parameters
        ----------
        y : torch.Tensor
            Outcomes.
        t : torch.Tensor
            Treatment indicators.
        a : torch.Tensor
            Baseline predictions a(X).
        b : torch.Tensor
            Treatment effect predictions b(X).

        Returns
        -------
        torch.Tensor
            Negative log-likelihood loss.
        """
        # Linear predictor: eta = a(X) + b(X) * T
        eta = a + b * t

        # Get family and compute proper NLL
        family = self._get_family()
        mu = family.inverse_link(eta)
        return family.nll_loss(y, mu)

    def _compute_dr_ate(
        self,
        y: np.ndarray,
        t: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
    ) -> tuple[float, float, np.ndarray]:
        """Compute doubly robust ATE with estimated dispersion.

        Following the MisraLab implementation:
        IF = (mu1 + T*(Y-mu1)/e) - (mu0 + (1-T)*(Y-mu0)/(1-e))

        Parameters
        ----------
        y : np.ndarray
            Outcomes.
        t : np.ndarray
            Treatment indicators.
        a : np.ndarray
            Baseline predictions a(X).
        b : np.ndarray
            Treatment effect predictions b(X).

        Returns
        -------
        tuple
            (ate, ate_se, influence_function)
        """
        n = len(y)

        # Potential outcomes
        mu0 = a  # E[Y|T=0, X]
        mu1 = a + b  # E[Y|T=1, X]

        # Propensity (clip to avoid division by near-zero)
        e = np.clip(t.mean(), 0.01, 0.99)

        # Influence function (doubly robust)
        IF = (mu1 + t * (y - mu1) / e) - (mu0 + (1 - t) * (y - mu0) / (1 - e))

        # ATE and SE
        ate = IF.mean()
        ate_se = IF.std() / np.sqrt(n)

        return ate, ate_se, IF

    def _compute_dr_ate_crossfit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        network: nn.Module,
        device: torch.device,
        n_folds: int = 5,
    ) -> tuple[float, float, np.ndarray]:
        """Compute ATE with cross-fitted standard errors via K-fold refitting.

        Proper cross-fitting for valid inference with neural networks:
        1. Split data into K folds
        2. For each fold, REFIT model on K-1 folds
        3. Predict on held-out fold using refitted model
        4. Compute influence function on out-of-sample predictions

        This breaks the dependence between model fitting and inference,
        providing valid standard errors even with flexible neural networks.

        Parameters
        ----------
        X : np.ndarray
            Covariates.
        y : np.ndarray
            Outcomes.
        t : np.ndarray
            Treatment indicators.
        network : nn.Module
            Trained parameter network (used as template for architecture).
        device : torch.device
            Computation device.
        n_folds : int, default=5
            Number of cross-fitting folds.

        Returns
        -------
        tuple
            (ate, ate_se, influence_function)
        """
        from sklearn.model_selection import KFold

        n = len(y)
        IF_full = np.zeros(n)
        a_full = np.zeros(n)
        b_full = np.zeros(n)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            # Get train/test splits
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            t_train, t_test = t[train_idx], t[test_idx]

            # Create a new network with same architecture
            fold_network = self._create_fresh_network(X.shape[1], device)

            # Train on this fold's training data
            self._fit_network_on_data(
                fold_network, X_train, y_train, t_train, device
            )

            # Predict on held-out test fold
            fold_network.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                params = fold_network(X_tensor)
                a_test = params[:, 0].cpu().numpy()
                b_test = params[:, 1].cpu().numpy()

            # Store predictions
            a_full[test_idx] = a_test
            b_full[test_idx] = b_test

            # Compute influence function on held-out fold
            mu0 = a_test  # E[Y|T=0, X]
            mu1 = a_test + b_test  # E[Y|T=1, X]

            # Propensity (clip to avoid division by zero)
            e = np.clip(t_test.mean(), 0.01, 0.99)

            # Doubly robust influence function
            IF_test = (mu1 + t_test * (y_test - mu1) / e) - \
                      (mu0 + (1 - t_test) * (y_test - mu0) / (1 - e))
            IF_full[test_idx] = IF_test

        ate = IF_full.mean()
        ate_se = IF_full.std() / np.sqrt(n)

        return ate, ate_se, IF_full

    def _create_fresh_network(
        self,
        input_dim: int,
        device: torch.device,
    ) -> nn.Module:
        """Create a fresh network with same architecture.

        Parameters
        ----------
        input_dim : int
            Input dimension.
        device : torch.device
            Computation device.

        Returns
        -------
        nn.Module
            Fresh network instance.
        """
        backbone_config = self._get_backbone_config(input_dim)
        backbone = ArchitectureRegistry.create(
            self.backbone,
            input_dim=input_dim,
            **backbone_config
        )
        network = ParameterNetwork(
            backbone=backbone,
            param_dim=2,
            param_names=["a", "b"],
        ).to(device)
        return network

    def _fit_network_on_data(
        self,
        network: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        device: torch.device,
    ) -> None:
        """Fit network on provided data (for cross-fitting).

        Parameters
        ----------
        network : nn.Module
            Network to train.
        X : np.ndarray
            Covariates.
        y : np.ndarray
            Outcomes.
        t : np.ndarray
            Treatment indicators.
        device : torch.device
            Computation device.
        """
        n = len(y)

        # Prepare data
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        t_tensor = torch.tensor(t, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor, t_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, n),
            shuffle=True,
        )

        # Optimizer
        optimizer = optim.Adam(
            network.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Training loop (reduced epochs for cross-fitting efficiency)
        network.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y, batch_t in dataloader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                batch_t = batch_t.to(device, non_blocking=True)

                optimizer.zero_grad()
                params = network(batch_X)
                a = params[:, 0]
                b = params[:, 1]
                loss = self._compute_loss(batch_y, batch_t, a, b)
                loss.backward()
                optimizer.step()

    def _compute_quantiles(
        self,
        b: np.ndarray,
        quantiles: list[float] | None = None,
        n_bootstrap: int = 200,
    ) -> tuple[dict[float, float], dict[float, float]]:
        """Compute quantiles of treatment effects with bootstrap SE.

        Parameters
        ----------
        b : np.ndarray
            Treatment effects b(X).
        quantiles : list[float], optional
            Quantiles to compute. Default is [0.1, 0.25, 0.5, 0.75, 0.9].
        n_bootstrap : int, default=200
            Number of bootstrap samples for SE.

        Returns
        -------
        tuple
            (quantile_values, quantile_se)
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        n = len(b)
        q_values = {}
        q_se = {}

        for q in quantiles:
            q_values[q] = np.quantile(b, q)

            # Bootstrap SE
            boot_quantiles = []
            for _ in range(n_bootstrap):
                idx = np.random.choice(n, n, replace=True)
                boot_quantiles.append(np.quantile(b[idx], q))
            q_se[q] = np.std(boot_quantiles)

        return q_values, q_se

    def predict(self, data: pd.DataFrame) -> Float64Array:
        """Predict outcomes for new data.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with covariates and treatment.

        Returns
        -------
        Float64Array
            Predicted outcomes.
        """
        # This is a convenience method; actual prediction uses HTEResults
        raise NotImplementedError("Use result.predict() after fitting")
