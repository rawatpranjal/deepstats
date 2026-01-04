"""Base classes for neural network architectures.

This module provides abstract base classes for network architectures
used in enriched structural models following Farrell, Liang, Misra (2021, 2023).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class NetworkArchitecture(nn.Module, ABC):
    """Abstract base class for all network architectures.

    All network architectures in deepstats inherit from this class,
    ensuring a consistent interface for use with structural models.

    The key insight from Farrell et al. is that networks should output
    parameter functions θ(X), not predictions directly. The architecture
    provides the backbone that learns representations from covariates.

    Attributes
    ----------
    input_dim : int
        Number of input features.
    output_dim : int
        Number of output dimensions (typically hidden_dim for backbones,
        or param_dim for complete parameter networks).
    architecture_name : str
        Human-readable name of the architecture.

    References
    ----------
    - Farrell, Liang, Misra (2021). "Deep Neural Networks for Estimation and Inference"
    - Farrell, Liang, Misra (2023). "Deep Learning for Individual Heterogeneity"
    """

    architecture_name: str = "base"

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Number of input features."""
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Number of output dimensions."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        pass

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_last_hidden_dim(self) -> int:
        """Get the dimension of the last hidden layer.

        This is used when attaching a parameter layer on top of
        a backbone network.

        Returns
        -------
        int
            Dimension of the last hidden layer output.
        """
        return self.output_dim

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Get default configuration for this architecture.

        Returns
        -------
        dict
            Default hyperparameters for this architecture.
        """
        return {}


class BackboneNetwork(NetworkArchitecture):
    """Base class for backbone networks that learn representations.

    Backbone networks take covariates X and output hidden representations
    that can be used by a parameter layer to produce θ(X).

    The architecture is:
        X → Backbone → Hidden → Parameter Layer → [a(X), b(X), ...]

    Subclasses should implement the forward pass to transform inputs
    to a fixed-size hidden representation.
    """

    architecture_name: str = "backbone"

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        """Initialize backbone network.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        hidden_dims : list[int], optional
            Hidden layer dimensions. Default architecture-specific.
        activation : str, default="relu"
            Activation function.
        dropout : float, default=0.0
            Dropout rate.
        batch_norm : bool, default=False
            Whether to use batch normalization.
        """
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims or [64, 32]
        self.activation = activation
        self.dropout_rate = dropout
        self.batch_norm = batch_norm

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._hidden_dims[-1] if self._hidden_dims else self._input_dim

    @property
    def hidden_dims(self) -> list[int]:
        return self._hidden_dims

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations: dict[str, nn.Module] = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "selu": nn.SELU(),
            "swish": nn.SiLU(),
        }
        if name.lower() not in activations:
            raise ValueError(
                f"Unknown activation: {name}. Available: {list(activations.keys())}"
            )
        return activations[name.lower()]


class ParameterNetwork(NetworkArchitecture):
    """Network that outputs parameter functions θ(X).

    This is a complete network that combines a backbone with a
    parameter layer to output the parameter functions.

    Architecture:
        X → Backbone → Hidden → Linear(hidden, param_dim) → θ(X)

    The output θ(X) = [θ₁(X), θ₂(X), ...] are the parameter functions
    that enter the structural loss.
    """

    architecture_name: str = "parameter_network"

    def __init__(
        self,
        backbone: BackboneNetwork,
        param_dim: int,
        param_names: list[str] | None = None,
    ) -> None:
        """Initialize parameter network.

        Parameters
        ----------
        backbone : BackboneNetwork
            Backbone network for learning representations.
        param_dim : int
            Number of parameter functions to output.
        param_names : list[str], optional
            Names for parameter functions (e.g., ["a", "b"]).
        """
        super().__init__()
        self.backbone = backbone
        self.param_dim = param_dim
        self.param_names = param_names or [f"theta_{i}" for i in range(param_dim)]

        # Parameter layer: maps hidden representation to parameters
        self.param_layer = nn.Linear(backbone.output_dim, param_dim)
        self._init_param_layer()

    def _init_param_layer(self) -> None:
        """Initialize parameter layer weights."""
        nn.init.xavier_uniform_(self.param_layer.weight)
        if self.param_layer.bias is not None:
            nn.init.zeros_(self.param_layer.bias)

    @property
    def input_dim(self) -> int:
        return self.backbone.input_dim

    @property
    def output_dim(self) -> int:
        return self.param_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get parameter functions.

        Parameters
        ----------
        x : torch.Tensor
            Covariates of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Parameter functions of shape (batch_size, param_dim).
            Column i is θᵢ(X).
        """
        hidden = self.backbone(x)
        params = self.param_layer(hidden)
        return params

    def get_params_dict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get parameter functions as a dictionary.

        Parameters
        ----------
        x : torch.Tensor
            Covariates of shape (batch_size, input_dim).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping parameter names to their values.
        """
        params = self.forward(x)
        return {
            name: params[:, i] for i, name in enumerate(self.param_names)
        }
