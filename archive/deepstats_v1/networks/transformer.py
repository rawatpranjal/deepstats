"""Transformer architecture for tabular data.

This module provides a Transformer-based backbone network for
learning representations from tabular/high-dimensional covariates.

The architecture treats features as a sequence of tokens, enabling
attention-based learning of feature interactions.

References
----------
- Vaswani et al. (2017). "Attention Is All You Need"
- Gorishniy et al. (2021). "Revisiting Deep Learning Models for Tabular Data"
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BackboneNetwork
from .registry import ArchitectureRegistry


class TabularTransformer(BackboneNetwork):
    """Transformer backbone for tabular data.

    This architecture treats each feature as a token and applies
    self-attention to learn feature interactions. This is particularly
    useful for high-dimensional covariates where features may have
    complex dependencies.

    Architecture:
        X (batch, d_in) → Embedding → Positional Encoding →
        TransformerEncoder → Pooling → Hidden (batch, d_model)

    Parameters
    ----------
    input_dim : int
        Number of input features.
    d_model : int, default=64
        Dimension of the transformer embedding.
    num_heads : int, default=4
        Number of attention heads.
    num_layers : int, default=2
        Number of transformer encoder layers.
    d_ff : int, default=128
        Dimension of feedforward network.
    dropout : float, default=0.1
        Dropout rate.
    pooling : str, default="mean"
        Pooling strategy: "mean", "cls", or "last".

    Examples
    --------
    >>> backbone = TabularTransformer(input_dim=50, d_model=64, num_heads=4)
    >>> x = torch.randn(100, 50)
    >>> h = backbone(x)  # Shape: (100, 64)
    """

    architecture_name = "transformer"

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        pooling: Literal["mean", "cls", "last"] = "mean",
        hidden_dims: list[int] | None = None,  # For compatibility
        activation: str = "gelu",
        batch_norm: bool = False,
    ) -> None:
        # Set hidden_dims based on d_model for compatibility
        if hidden_dims is None:
            hidden_dims = [d_model]

        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.pooling = pooling

        # Feature embedding: project each feature to d_model
        # Treats each feature as a separate token
        self.feature_embedding = nn.Linear(1, d_model)

        # CLS token for classification pooling
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=input_dim + 1,  # +1 for potential CLS token
            dropout=dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Layer norm for final output
        self.output_norm = nn.LayerNorm(d_model)

        self._init_weights()

    @property
    def output_dim(self) -> int:
        return self.d_model

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Hidden representation of shape (batch_size, d_model).
        """
        batch_size, n_features = x.shape

        # Reshape to (batch, seq_len, 1) and embed each feature
        x = x.unsqueeze(-1)  # (batch, n_features, 1)
        x = self.feature_embedding(x)  # (batch, n_features, d_model)

        # Add CLS token if using cls pooling
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer
        x = self.transformer(x)  # (batch, seq_len, d_model)

        # Pool to get fixed-size representation
        if self.pooling == "mean":
            x = x.mean(dim=1)
        elif self.pooling == "cls":
            x = x[:, 0, :]  # CLS token
        elif self.pooling == "last":
            x = x[:, -1, :]

        # Apply layer norm
        x = self.output_norm(x)

        return x

    def __repr__(self) -> str:
        return (
            f"TabularTransformer(d_in={self.input_dim}, d_model={self.d_model}, "
            f"heads={self.num_heads}, layers={self.num_layers}, "
            f"params={self.count_parameters():,})"
        )


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer.

    Uses sinusoidal positional encoding as in the original
    Transformer paper.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 1000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Input with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Register the transformer architecture
@ArchitectureRegistry.register(
    "transformer",
    default_config={"d_model": 64, "num_heads": 4, "num_layers": 2, "d_ff": 128},
)
def create_transformer(input_dim: int, **kwargs) -> TabularTransformer:
    """Create a TabularTransformer backbone."""
    return TabularTransformer(input_dim=input_dim, **kwargs)
