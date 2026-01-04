"""Multi-layer perceptron implementation.

This module provides neural network architectures for use as
flexible function approximators in econometric models.

References
----------
- Farrell, Liang, Misra (2021). "Deep Neural Networks for Estimation and Inference"
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .._typing import Float64Array
from .base import BackboneNetwork
from .registry import ArchitectureRegistry


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture.

    A flexible feedforward neural network that can be customized
    with different hidden layer sizes, activations, and regularization.

    This class is designed to be sklearn-compatible when wrapped
    in an estimator class.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list[int], optional
        Hidden layer dimensions. Default is [64, 32].
    output_dim : int, default=1
        Number of outputs.
    activation : str, default="relu"
        Activation function: "relu", "leaky_relu", "tanh", "elu", "gelu".
    dropout : float, default=0.0
        Dropout rate (0 to 1).
    batch_norm : bool, default=False
        Whether to use batch normalization.

    Examples
    --------
    >>> net = MLP(input_dim=10, hidden_dims=[64, 32])
    >>> x = torch.randn(100, 10)
    >>> y = net(x)  # Shape: (100, 1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        output_dim: int = 1,
        activation: Literal["relu", "leaky_relu", "tanh", "elu", "gelu"] = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout

        # Build activation
        activation_fn = self._get_activation(activation)

        # Build layers
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(activation_fn)

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations: dict[str, nn.Module] = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
        }
        if name.lower() not in activations:
            raise ValueError(
                f"Unknown activation: {name}. Available: {list(activations.keys())}"
            )
        return activations[name.lower()]

    def _init_weights(self) -> None:
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor (batch_size, output_dim).
        """
        return self.network(x)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        arch = f"{self.input_dim} -> " + " -> ".join(
            str(d) for d in self.hidden_dims
        ) + f" -> {self.output_dim}"
        return f"MLP({arch}, params={self.count_parameters():,})"


class MLPClassifier(MLP):
    """MLP for binary classification with sigmoid output.

    This is a convenience wrapper around MLP that applies sigmoid
    to the output for use in propensity score estimation.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list[int], optional
        Hidden layer dimensions. Default is [32, 16].
    **kwargs
        Additional arguments passed to MLP.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        **kwargs,
    ) -> None:
        if hidden_dims is None:
            hidden_dims = [32, 16]
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            **kwargs,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sigmoid activation."""
        logits = self.network(x)
        return self.sigmoid(logits)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability predictions (for sklearn compatibility)."""
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
        # Return [P(y=0), P(y=1)]
        return torch.cat([1 - probs, probs], dim=1)


class MLPBackbone(BackboneNetwork):
    """MLP backbone for structural models.

    This class provides an MLP that outputs hidden representations
    instead of predictions, for use as a backbone in parameter networks.

    Architecture:
        X (batch, d_in) → Linear → ReLU → ... → Hidden (batch, d_out)

    The output is the last hidden layer before the output projection,
    which can be used by a parameter layer to produce θ(X).

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list[int], optional
        Hidden layer dimensions. Default is [64, 32].
    activation : str, default="relu"
        Activation function.
    dropout : float, default=0.0
        Dropout rate.
    batch_norm : bool, default=False
        Whether to use batch normalization.

    Examples
    --------
    >>> backbone = MLPBackbone(input_dim=10, hidden_dims=[64, 32])
    >>> x = torch.randn(100, 10)
    >>> h = backbone(x)  # Shape: (100, 32)
    """

    architecture_name = "mlp"

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        activation: Literal["relu", "leaky_relu", "tanh", "elu", "gelu"] = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        if hidden_dims is None:
            hidden_dims = [64, 32]

        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        # Build layers
        layers: list[nn.Module] = []
        prev_dim = input_dim
        activation_fn = self._get_activation(activation)

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(activation_fn)

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get hidden representation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Hidden representation (batch_size, output_dim).
        """
        return self.network(x)

    def __repr__(self) -> str:
        arch = f"{self.input_dim} -> " + " -> ".join(
            str(d) for d in self.hidden_dims
        )
        return f"MLPBackbone({arch}, params={self.count_parameters():,})"


# Register MLP backbone with the registry
@ArchitectureRegistry.register(
    "mlp",
    default_config={"hidden_dims": [64, 32], "activation": "relu"},
)
def create_mlp_backbone(input_dim: int, **kwargs) -> MLPBackbone:
    """Create an MLPBackbone."""
    return MLPBackbone(input_dim=input_dim, **kwargs)


def create_network(
    input_dim: int,
    network_type: Literal["mlp", "mlp_classifier"] = "mlp",
    **kwargs,
) -> nn.Module:
    """Factory function to create networks.

    This is a legacy function. For new code, use ArchitectureRegistry.create().

    Parameters
    ----------
    input_dim : int
        Number of input features.
    network_type : str, default="mlp"
        Type: "mlp" or "mlp_classifier".
    **kwargs
        Arguments passed to network constructor.

    Returns
    -------
    nn.Module
        Constructed network.
    """
    if network_type == "mlp":
        return MLP(input_dim=input_dim, **kwargs)
    elif network_type == "mlp_classifier":
        return MLPClassifier(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown network type: {network_type}")
