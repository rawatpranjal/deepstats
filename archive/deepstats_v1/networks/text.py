"""Text backbone for sequence inputs.

This module provides text/sequence backbones for processing
text or token sequences in heterogeneous treatment effects estimation.

Examples
--------
>>> from deepstats.networks.text import TextBackbone
>>> backbone = TextBackbone(vocab_size=10000, embed_dim=128)
>>> tokens = torch.randint(0, 10000, (16, 50))  # batch of 16, seq_len 50
>>> features = backbone(tokens)  # (16, output_dim)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .base import BackboneNetwork
from .registry import ArchitectureRegistry


@ArchitectureRegistry.register("text")
class TextBackbone(BackboneNetwork):
    """Text backbone using embeddings + pooling.

    Processes token sequences and outputs a fixed-size feature vector.
    Supports different pooling strategies and optional transformer encoding.

    Parameters
    ----------
    vocab_size : int, default=10000
        Size of vocabulary (number of unique tokens).
    embed_dim : int, default=128
        Dimension of token embeddings.
    output_dim : int, default=64
        Dimension of output feature vector.
    max_seq_len : int, default=512
        Maximum sequence length for positional embeddings.
    pooling : str, default="mean"
        Pooling strategy: "mean", "max", "cls", or "attention".
    use_transformer : bool, default=True
        Whether to use transformer encoder layers.
    num_layers : int, default=2
        Number of transformer layers (if use_transformer=True).
    num_heads : int, default=4
        Number of attention heads (if use_transformer=True).
    dropout : float, default=0.1
        Dropout rate.
    input_dim : int, optional
        For registry compatibility (ignored, uses vocab_size).

    Attributes
    ----------
    input_dim : int
        Vocabulary size.
    output_dim : int
        Dimension of output features.

    Examples
    --------
    >>> backbone = TextBackbone(vocab_size=5000, embed_dim=64)
    >>> tokens = torch.randint(0, 5000, (8, 100))
    >>> out = backbone(tokens)
    >>> print(out.shape)  # (8, 64)
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 128,
        output_dim: int = 64,
        max_seq_len: int = 512,
        pooling: Literal["mean", "max", "cls", "attention"] = "mean",
        use_transformer: bool = True,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        input_dim: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(input_dim=vocab_size, hidden_dims=[output_dim], dropout=dropout)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self._output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.pooling = pooling
        self.use_transformer = use_transformer

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Positional embedding
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Optional CLS token for "cls" pooling
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        else:
            self.cls_token = None

        # Transformer encoder (optional)
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            # Simple LSTM fallback
            self.encoder = nn.LSTM(
                embed_dim,
                embed_dim // 2,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0,
            )

        # Attention pooling (optional)
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.Tanh(),
                nn.Linear(embed_dim // 2, 1),
            )
        else:
            self.attention = None

        # Output projection
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, output_dim)

    @property
    def input_dim(self) -> int:
        """Vocabulary size."""
        return self.vocab_size

    @property
    def output_dim(self) -> int:
        """Output feature dimension."""
        return self._output_dim

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Token indices of shape (batch, seq_len) or
            pre-embedded vectors of shape (batch, seq_len, embed_dim).
        attention_mask : torch.Tensor, optional
            Mask for padding tokens (1 = valid, 0 = padding).

        Returns
        -------
        torch.Tensor
            Feature tensor of shape (batch, output_dim).
        """
        batch_size = x.shape[0]

        # Handle both token indices and pre-embedded inputs
        if x.dim() == 2 and x.dtype in (torch.long, torch.int32, torch.int64):
            # Token indices
            seq_len = x.shape[1]
            x = self.embedding(x)  # (batch, seq_len, embed_dim)
        elif x.dim() == 3:
            # Pre-embedded
            seq_len = x.shape[1]
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # Add positional embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        x = x + self.pos_embedding(positions)

        # Add CLS token if using cls pooling
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            seq_len += 1

        # Encode
        if self.use_transformer:
            # Create attention mask for transformer if provided
            if attention_mask is not None:
                if self.cls_token is not None:
                    # Add mask for CLS token
                    cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
                    attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
                # Convert to transformer format (True = masked)
                src_key_padding_mask = attention_mask == 0
            else:
                src_key_padding_mask = None

            x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        else:
            # LSTM encoder
            x, _ = self.encoder(x)

        # Pooling
        if self.pooling == "cls":
            # Use CLS token representation
            pooled = x[:, 0]
        elif self.pooling == "mean":
            # Mean pooling (masked if mask provided)
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1)
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)
        elif self.pooling == "max":
            # Max pooling
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1)
                x = x.masked_fill(mask == 0, float("-inf"))
            pooled = x.max(dim=1)[0]
        elif self.pooling == "attention":
            # Attention pooling
            attn_weights = self.attention(x).squeeze(-1)  # (batch, seq_len)
            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))
            attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(-1)
            pooled = (x * attn_weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Output projection
        pooled = self.dropout(pooled)
        output = self.fc(pooled)

        return output


@ArchitectureRegistry.register("bow")
class BagOfWordsBackbone(BackboneNetwork):
    """Simple bag-of-words backbone.

    Fast baseline that ignores word order. Uses embedding averaging.

    Parameters
    ----------
    vocab_size : int, default=10000
        Size of vocabulary.
    embed_dim : int, default=128
        Embedding dimension.
    output_dim : int, default=64
        Output feature dimension.
    dropout : float, default=0.0
        Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.0,
        input_dim: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(input_dim=vocab_size, hidden_dims=[output_dim], dropout=dropout)
        self.vocab_size = vocab_size
        self._output_dim = output_dim

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean", padding_idx=0)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(embed_dim, output_dim)

    @property
    def input_dim(self) -> int:
        return self.vocab_size

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Token indices of shape (batch, seq_len).

        Returns
        -------
        torch.Tensor
            Feature tensor of shape (batch, output_dim).
        """
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
