"""LSTM architecture for sequential/tabular data.

This module provides an LSTM-based backbone network for
learning representations from covariates.

The architecture can treat features as a sequence, enabling
recurrent learning of feature dependencies.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .base import BackboneNetwork
from .registry import ArchitectureRegistry


class LSTMBackbone(BackboneNetwork):
    """LSTM backbone for tabular/sequential data.

    This architecture treats features as a sequence and applies
    LSTM to learn sequential dependencies. Can be bidirectional
    for capturing dependencies in both directions.

    Architecture:
        X (batch, d_in) → Embedding → LSTM → Pooling → Hidden

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_size : int, default=64
        Size of LSTM hidden state.
    num_layers : int, default=2
        Number of LSTM layers.
    bidirectional : bool, default=True
        Whether to use bidirectional LSTM.
    dropout : float, default=0.1
        Dropout rate between LSTM layers.
    pooling : str, default="last"
        Pooling strategy: "last", "mean", or "max".

    Examples
    --------
    >>> backbone = LSTMBackbone(input_dim=50, hidden_size=64)
    >>> x = torch.randn(100, 50)
    >>> h = backbone(x)  # Shape: (100, 128) for bidirectional
    """

    architecture_name = "lstm"

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
        pooling: Literal["last", "mean", "max"] = "last",
        hidden_dims: list[int] | None = None,  # For compatibility
        activation: str = "tanh",
        batch_norm: bool = False,
    ) -> None:
        # Compute output dim based on bidirectionality
        effective_hidden = hidden_size * 2 if bidirectional else hidden_size

        if hidden_dims is None:
            hidden_dims = [effective_hidden]

        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling = pooling
        self._effective_hidden = effective_hidden

        # Feature embedding: project each feature to embedding dim
        self.embedding_dim = min(hidden_size, 32)
        self.feature_embedding = nn.Linear(1, self.embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection to fixed size
        self.output_projection = nn.Linear(effective_hidden, effective_hidden)
        self.output_norm = nn.LayerNorm(effective_hidden)

        self._init_weights()

    @property
    def output_dim(self) -> int:
        return self._effective_hidden

    def _init_weights(self) -> None:
        """Initialize weights using orthogonal initialization for LSTM."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        # Initialize other layers
        for module in [self.feature_embedding, self.output_projection]:
            if hasattr(module, "weight"):
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Hidden representation of shape (batch_size, output_dim).
        """
        batch_size, n_features = x.shape

        # Reshape to (batch, seq_len, 1) and embed each feature
        x = x.unsqueeze(-1)  # (batch, n_features, 1)
        x = self.feature_embedding(x)  # (batch, n_features, embedding_dim)

        # Apply LSTM
        # output: (batch, seq_len, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        output, (h_n, c_n) = self.lstm(x)

        # Pool to get fixed-size representation
        if self.pooling == "last":
            if self.bidirectional:
                # Concatenate last hidden states from both directions
                h_forward = h_n[-2, :, :]  # Last layer, forward
                h_backward = h_n[-1, :, :]  # Last layer, backward
                pooled = torch.cat([h_forward, h_backward], dim=-1)
            else:
                pooled = h_n[-1, :, :]
        elif self.pooling == "mean":
            pooled = output.mean(dim=1)
        elif self.pooling == "max":
            pooled, _ = output.max(dim=1)

        # Apply output projection and normalization
        out = self.output_projection(pooled)
        out = self.output_norm(out)

        return out

    def __repr__(self) -> str:
        bidir = "bi" if self.bidirectional else "uni"
        return (
            f"LSTMBackbone(d_in={self.input_dim}, hidden={self.hidden_size}, "
            f"layers={self.num_layers}, {bidir}, params={self.count_parameters():,})"
        )


# Register the LSTM architecture
@ArchitectureRegistry.register(
    "lstm",
    default_config={"hidden_size": 64, "num_layers": 2, "bidirectional": True},
)
def create_lstm(input_dim: int, **kwargs) -> LSTMBackbone:
    """Create an LSTMBackbone."""
    return LSTMBackbone(input_dim=input_dim, **kwargs)
