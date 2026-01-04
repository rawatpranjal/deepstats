"""Graph DGP wrappers for simulation studies.

Wraps the graph HTE generator from deepstats.datasets.multimodal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from deepstats.datasets.multimodal import make_graph_hte, GraphHTEData


@dataclass
class GraphDGP:
    """Wrapper for graph DGP functions.

    Attributes
    ----------
    name : str
        Name of the DGP scenario.
    heterogeneity : str
        Type of heterogeneity (density, size, centrality).
    min_nodes : int
        Minimum nodes per graph.
    max_nodes : int
        Maximum nodes per graph.
    node_feature_dim : int
        Dimension of node features.
    description : str
        Description of the scenario.
    """

    name: str
    heterogeneity: Literal["density", "size", "centrality"]
    min_nodes: int = 10
    max_nodes: int = 50
    node_feature_dim: int = 8
    description: str = ""

    def generate(
        self, seed: int, n: int = 2000, noise_scale: float = 1.0
    ) -> dict:
        """Generate graph HTE data.

        Parameters
        ----------
        seed : int
            Random seed.
        n : int, default=2000
            Number of observations.
        noise_scale : float, default=1.0
            Noise scale.

        Returns
        -------
        dict
            Dictionary with graph data and ground truth.
        """
        data = make_graph_hte(
            n=n,
            num_nodes_range=(self.min_nodes, self.max_nodes),
            num_features=self.node_feature_dim,
            heterogeneity=self.heterogeneity,
            noise_scale=noise_scale,
            seed=seed,
        )

        return {
            "node_features": data.node_features,
            "edge_indices": data.edge_indices,
            "treatment": data.treatment,
            "outcome": data.outcome,
            "true_ate": data.true_ate,
            "true_ite": data.true_ite,
            "graph_features": data.graph_features,
        }


# Registry of graph DGPs
GRAPH_DGPS: dict[str, GraphDGP] = {
    "density": GraphDGP(
        name="density",
        heterogeneity="density",
        description="Treatment effect varies with graph density",
    ),
    "size": GraphDGP(
        name="size",
        heterogeneity="size",
        description="Treatment effect varies with graph size",
    ),
    "centrality": GraphDGP(
        name="centrality",
        heterogeneity="centrality",
        description="Treatment effect varies with centrality measures",
    ),
}


def get_graph_dgp(name: str) -> GraphDGP:
    """Get a graph DGP by name."""
    if name not in GRAPH_DGPS:
        raise ValueError(
            f"Unknown graph DGP: {name}. Available: {list(GRAPH_DGPS.keys())}"
        )
    return GRAPH_DGPS[name]


def list_graph_dgps() -> list[str]:
    """List available graph DGP names."""
    return list(GRAPH_DGPS.keys())
