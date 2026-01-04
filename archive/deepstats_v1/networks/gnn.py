"""Graph Neural Network backbone for graph inputs.

This module provides GNN backbones for processing graph-structured data
in heterogeneous treatment effects estimation.

The implementation uses a simple message passing framework that doesn't
require PyTorch Geometric, but is compatible with it.

Examples
--------
>>> from deepstats.networks.gnn import GNNBackbone
>>> backbone = GNNBackbone(input_dim=16, hidden_dim=32)
>>> # node_features: (num_nodes, input_dim)
>>> # edge_index: (2, num_edges) - source and target node indices
>>> node_features = torch.randn(100, 16)
>>> edge_index = torch.randint(0, 100, (2, 300))
>>> batch = torch.zeros(100, dtype=torch.long)  # all nodes in one graph
>>> features = backbone(node_features, edge_index, batch)  # (1, output_dim)
"""

from __future__ import annotations

from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BackboneNetwork
from .registry import ArchitectureRegistry


class MessagePassingLayer(nn.Module):
    """Basic message passing layer.

    Aggregates neighbor features and updates node representations.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        aggregation: Literal["mean", "sum", "max"] = "mean",
    ) -> None:
        super().__init__()
        self.aggregation = aggregation

        # Message and update functions
        self.message_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (num_nodes, in_dim).
        edge_index : torch.Tensor
            Edge indices of shape (2, num_edges).

        Returns
        -------
        torch.Tensor
            Updated node features of shape (num_nodes, out_dim).
        """
        num_nodes = x.size(0)
        source, target = edge_index[0], edge_index[1]

        # Create messages
        source_features = x[source]
        target_features = x[target]
        messages = self.message_mlp(torch.cat([source_features, target_features], dim=-1))

        # Aggregate messages per target node
        aggregated = torch.zeros(num_nodes, messages.size(-1), device=x.device)

        if self.aggregation == "sum":
            aggregated.scatter_add_(0, target.unsqueeze(-1).expand_as(messages), messages)
        elif self.aggregation == "mean":
            aggregated.scatter_add_(0, target.unsqueeze(-1).expand_as(messages), messages)
            counts = torch.zeros(num_nodes, device=x.device)
            counts.scatter_add_(0, target, torch.ones_like(target, dtype=torch.float))
            counts = counts.clamp(min=1).unsqueeze(-1)
            aggregated = aggregated / counts
        elif self.aggregation == "max":
            # For max, we need scatter_reduce (PyTorch 1.11+)
            aggregated = torch.zeros(num_nodes, messages.size(-1), device=x.device)
            for i in range(num_nodes):
                mask = target == i
                if mask.any():
                    aggregated[i] = messages[mask].max(dim=0)[0]

        # Update node features
        updated = self.update_mlp(torch.cat([x, aggregated], dim=-1))

        return updated


class GCNLayer(nn.Module):
    """Graph Convolutional Network layer (simplified Kipf & Welling 2017)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with symmetric normalization."""
        num_nodes = x.size(0)
        source, target = edge_index[0], edge_index[1]

        # Compute degree for normalization
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, target, torch.ones_like(target, dtype=torch.float))
        deg = deg.clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)

        # Normalize adjacency
        norm = deg_inv_sqrt[source] * deg_inv_sqrt[target]

        # Message passing with normalization
        messages = x[source] * norm.unsqueeze(-1)
        aggregated = torch.zeros(num_nodes, x.size(-1), device=x.device)
        aggregated.scatter_add_(0, target.unsqueeze(-1).expand_as(messages), messages)

        # Add self-loop
        aggregated = aggregated + x

        # Linear transformation
        out = self.linear(aggregated)

        return out


@ArchitectureRegistry.register("gnn")
class GNNBackbone(BackboneNetwork):
    """Graph Neural Network backbone.

    Processes graph-structured data and outputs a graph-level feature vector.
    Uses message passing with various aggregation strategies.

    Parameters
    ----------
    input_dim : int
        Dimension of input node features.
    hidden_dim : int, default=64
        Dimension of hidden layers.
    output_dim : int, default=64
        Dimension of output feature vector.
    num_layers : int, default=3
        Number of message passing layers.
    gnn_type : str, default="mpnn"
        Type of GNN: "mpnn" (message passing), "gcn" (graph conv).
    aggregation : str, default="mean"
        Aggregation for message passing: "mean", "sum", "max".
    pooling : str, default="mean"
        Graph-level pooling: "mean", "sum", "max".
    dropout : float, default=0.0
        Dropout rate.

    Attributes
    ----------
    input_dim : int
        Input node feature dimension.
    output_dim : int
        Output feature dimension.

    Examples
    --------
    >>> backbone = GNNBackbone(input_dim=16, hidden_dim=32)
    >>> x = torch.randn(50, 16)  # 50 nodes
    >>> edge_index = torch.randint(0, 50, (2, 100))
    >>> batch = torch.zeros(50, dtype=torch.long)
    >>> out = backbone(x, edge_index, batch)
    >>> print(out.shape)  # (1, 64)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        num_layers: int = 3,
        gnn_type: Literal["mpnn", "gcn"] = "mpnn",
        aggregation: Literal["mean", "sum", "max"] = "mean",
        pooling: Literal["mean", "sum", "max"] = "mean",
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(input_dim=input_dim, hidden_dims=[output_dim], dropout=dropout)
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.pooling = pooling

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GNN layers
        layers = []
        for _ in range(num_layers):
            if gnn_type == "gcn":
                layers.append(GCNLayer(hidden_dim, hidden_dim))
            else:  # mpnn
                layers.append(MessagePassingLayer(hidden_dim, hidden_dim, aggregation))
        self.gnn_layers = nn.ModuleList(layers)

        # Layer norm for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (num_nodes, input_dim).
        edge_index : torch.Tensor
            Edge indices of shape (2, num_edges).
        batch : torch.Tensor, optional
            Batch assignment for each node. Shape (num_nodes,).
            If None, assumes all nodes belong to a single graph.

        Returns
        -------
        torch.Tensor
            Graph-level features of shape (num_graphs, output_dim).
        """
        num_nodes = x.size(0)

        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # Message passing layers
        for gnn_layer, layer_norm in zip(self.gnn_layers, self.layer_norms):
            residual = x
            x = gnn_layer(x, edge_index)
            x = layer_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + residual  # Residual connection

        # Graph-level pooling
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)

        num_graphs = batch.max().item() + 1

        if self.pooling == "mean":
            pooled = torch.zeros(num_graphs, x.size(-1), device=x.device)
            pooled.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
            counts = torch.zeros(num_graphs, device=x.device)
            counts.scatter_add_(0, batch, torch.ones(num_nodes, device=x.device))
            pooled = pooled / counts.clamp(min=1).unsqueeze(-1)
        elif self.pooling == "sum":
            pooled = torch.zeros(num_graphs, x.size(-1), device=x.device)
            pooled.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
        elif self.pooling == "max":
            pooled = torch.zeros(num_graphs, x.size(-1), device=x.device)
            for i in range(num_graphs):
                mask = batch == i
                if mask.any():
                    pooled[i] = x[mask].max(dim=0)[0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Output projection
        output = self.output_proj(pooled)

        return output


class GraphData:
    """Simple container for graph data.

    Compatible with PyTorch Geometric Data objects.

    Attributes
    ----------
    x : torch.Tensor
        Node features of shape (num_nodes, num_features).
    edge_index : torch.Tensor
        Edge indices of shape (2, num_edges).
    batch : torch.Tensor, optional
        Batch assignment for nodes.
    y : torch.Tensor, optional
        Graph-level labels.
    """

    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        self.x = x
        self.edge_index = edge_index
        self.batch = batch
        self.y = y

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def num_nodes(self) -> int:
        return self.x.size(0)

    @property
    def num_edges(self) -> int:
        return self.edge_index.size(1)

    @property
    def num_graphs(self) -> int:
        if self.batch is None:
            return 1
        return self.batch.max().item() + 1

    def to(self, device: torch.device) -> "GraphData":
        """Move data to device."""
        return GraphData(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            batch=self.batch.to(device) if self.batch is not None else None,
            y=self.y.to(device) if self.y is not None else None,
        )


def create_random_graph(
    num_nodes: int,
    num_features: int,
    edge_prob: float = 0.1,
    seed: int | None = None,
) -> GraphData:
    """Create a random graph for testing.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    num_features : int
        Dimension of node features.
    edge_prob : float, default=0.1
        Probability of edge between any two nodes.
    seed : int, optional
        Random seed.

    Returns
    -------
    GraphData
        Random graph data.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Random node features
    x = torch.randn(num_nodes, num_features)

    # Random edges (Erdos-Renyi)
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if torch.rand(1).item() < edge_prob:
                edges.append([i, j])
                edges.append([j, i])  # Undirected

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return GraphData(x=x, edge_index=edge_index)
