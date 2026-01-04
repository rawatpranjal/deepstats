"""Tests for multimodal architectures and data generators."""

import numpy as np
import pytest
import torch

from deepstats.networks import (
    ArchitectureRegistry,
    CNNBackbone,
    ResNetBackbone,
    TextBackbone,
    BagOfWordsBackbone,
    GNNBackbone,
    GraphData,
    create_random_graph,
)
from deepstats.datasets.multimodal import (
    make_image_hte,
    make_text_hte,
    make_graph_hte,
    multimodal_to_tensors,
)


class TestCNNBackbone:
    """Test CNN backbone for images."""

    def test_cnn_backbone_forward(self):
        """Test CNN forward pass."""
        backbone = CNNBackbone(
            in_channels=3,
            image_size=32,
            hidden_channels=[16, 32],
            output_dim=64,
        )

        x = torch.randn(8, 3, 32, 32)
        out = backbone(x)

        assert out.shape == (8, 64)

    def test_cnn_backbone_flattened_input(self):
        """Test CNN with flattened input."""
        backbone = CNNBackbone(
            in_channels=3,
            image_size=16,
            hidden_channels=[16, 32],
            output_dim=32,
        )

        # Flattened input
        x = torch.randn(4, 3 * 16 * 16)
        out = backbone(x)

        assert out.shape == (4, 32)

    def test_cnn_registry(self):
        """Test CNN via registry."""
        backbone = ArchitectureRegistry.create(
            "cnn",
            input_dim=1 * 28 * 28,  # Required by registry
            in_channels=1,
            image_size=28,
            output_dim=32,
        )

        x = torch.randn(4, 1, 28, 28)
        out = backbone(x)

        assert out.shape == (4, 32)


class TestResNetBackbone:
    """Test ResNet backbone."""

    def test_resnet_backbone_forward(self):
        """Test ResNet forward pass."""
        backbone = ResNetBackbone(
            in_channels=3,
            image_size=32,
            hidden_channels=[16, 32],
            blocks_per_stage=1,
            output_dim=64,
        )

        x = torch.randn(4, 3, 32, 32)
        out = backbone(x)

        assert out.shape == (4, 64)


class TestTextBackbone:
    """Test text backbone."""

    def test_text_backbone_forward(self):
        """Test text backbone forward pass."""
        backbone = TextBackbone(
            vocab_size=1000,
            embed_dim=64,
            output_dim=32,
            max_seq_len=100,
            num_layers=1,
            num_heads=2,
        )

        tokens = torch.randint(0, 1000, (8, 50))
        out = backbone(tokens)

        assert out.shape == (8, 32)

    def test_text_backbone_pooling_modes(self):
        """Test different pooling strategies."""
        for pooling in ["mean", "max", "cls", "attention"]:
            backbone = TextBackbone(
                vocab_size=500,
                embed_dim=32,
                output_dim=16,
                pooling=pooling,
                num_layers=1,
                num_heads=2,
            )

            tokens = torch.randint(0, 500, (4, 30))
            out = backbone(tokens)

            assert out.shape == (4, 16)

    def test_text_backbone_with_mask(self):
        """Test text backbone with attention mask."""
        backbone = TextBackbone(
            vocab_size=500,
            embed_dim=32,
            output_dim=16,
            pooling="mean",
            num_layers=1,
        )

        tokens = torch.randint(0, 500, (4, 30))
        # Mask: first 20 tokens are valid
        mask = torch.zeros(4, 30)
        mask[:, :20] = 1

        out = backbone(tokens, attention_mask=mask)

        assert out.shape == (4, 16)

    def test_text_registry(self):
        """Test text backbone via registry."""
        backbone = ArchitectureRegistry.create(
            "text",
            input_dim=1000,  # Required by registry (vocab_size)
            vocab_size=1000,
            embed_dim=32,
            output_dim=16,
        )

        tokens = torch.randint(0, 1000, (4, 20))
        out = backbone(tokens)

        assert out.shape == (4, 16)


class TestBagOfWordsBackbone:
    """Test bag-of-words backbone."""

    def test_bow_forward(self):
        """Test BOW forward pass."""
        backbone = BagOfWordsBackbone(
            vocab_size=500,
            embed_dim=32,
            output_dim=16,
        )

        tokens = torch.randint(0, 500, (8, 50))
        out = backbone(tokens)

        assert out.shape == (8, 16)


class TestGNNBackbone:
    """Test GNN backbone."""

    def test_gnn_backbone_forward(self):
        """Test GNN forward pass."""
        backbone = GNNBackbone(
            input_dim=16,
            hidden_dim=32,
            output_dim=24,
            num_layers=2,
        )

        # Single graph
        x = torch.randn(50, 16)
        edge_index = torch.randint(0, 50, (2, 100))
        batch = torch.zeros(50, dtype=torch.long)

        out = backbone(x, edge_index, batch)

        assert out.shape == (1, 24)

    def test_gnn_backbone_batched(self):
        """Test GNN with batched graphs."""
        backbone = GNNBackbone(
            input_dim=8,
            hidden_dim=16,
            output_dim=12,
            num_layers=2,
        )

        # Two graphs batched together
        x = torch.randn(80, 8)  # 40 + 40 nodes
        edge_index = torch.cat([
            torch.randint(0, 40, (2, 50)),
            torch.randint(40, 80, (2, 50)),
        ], dim=1)
        batch = torch.cat([
            torch.zeros(40, dtype=torch.long),
            torch.ones(40, dtype=torch.long),
        ])

        out = backbone(x, edge_index, batch)

        assert out.shape == (2, 12)

    def test_gnn_pooling_modes(self):
        """Test different pooling strategies."""
        for pooling in ["mean", "sum", "max"]:
            backbone = GNNBackbone(
                input_dim=8,
                hidden_dim=16,
                output_dim=12,
                pooling=pooling,
            )

            x = torch.randn(30, 8)
            edge_index = torch.randint(0, 30, (2, 50))

            out = backbone(x, edge_index)

            assert out.shape == (1, 12)

    def test_gnn_registry(self):
        """Test GNN via registry."""
        backbone = ArchitectureRegistry.create(
            "gnn",
            input_dim=16,
            hidden_dim=32,
            output_dim=24,
        )

        x = torch.randn(40, 16)
        edge_index = torch.randint(0, 40, (2, 80))

        out = backbone(x, edge_index)

        assert out.shape == (1, 24)


class TestGraphData:
    """Test GraphData container."""

    def test_graph_data_creation(self):
        """Test GraphData creation."""
        x = torch.randn(20, 8)
        edge_index = torch.randint(0, 20, (2, 40))

        data = GraphData(x=x, edge_index=edge_index)

        assert data.num_nodes == 20
        assert data.num_edges == 40
        assert data.num_graphs == 1

    def test_create_random_graph(self):
        """Test random graph creation."""
        graph = create_random_graph(
            num_nodes=30,
            num_features=8,
            edge_prob=0.1,
            seed=42,
        )

        assert graph.num_nodes == 30
        assert graph.x.shape == (30, 8)
        assert graph.edge_index.shape[0] == 2


class TestMultimodalDataGenerators:
    """Test multimodal data generators."""

    def test_make_image_hte(self):
        """Test image HTE data generation."""
        data = make_image_hte(
            n=100,
            image_size=16,
            channels=3,
            seed=42,
        )

        assert data.images.shape == (100, 3, 16, 16)
        assert len(data.treatment) == 100
        assert len(data.outcome) == 100
        assert isinstance(data.true_ate, float)
        assert len(data.true_ite) == 100

    def test_make_image_hte_heterogeneity(self):
        """Test different image heterogeneity types."""
        for het in ["brightness", "texture", "color", "complex"]:
            data = make_image_hte(
                n=50,
                image_size=16,
                heterogeneity=het,
                seed=42,
            )

            assert len(data.images) == 50
            assert isinstance(data.true_ate, float)

    def test_make_text_hte(self):
        """Test text HTE data generation."""
        data = make_text_hte(
            n=100,
            seq_len=30,
            vocab_size=500,
            seed=42,
        )

        assert data.tokens.shape == (100, 30)
        assert len(data.treatment) == 100
        assert isinstance(data.true_ate, float)
        assert data.vocab_size == 500

    def test_make_text_hte_heterogeneity(self):
        """Test different text heterogeneity types."""
        for het in ["length", "frequency", "pattern", "complex"]:
            data = make_text_hte(
                n=50,
                seq_len=20,
                heterogeneity=het,
                seed=42,
            )

            assert len(data.tokens) == 50

    def test_make_graph_hte(self):
        """Test graph HTE data generation."""
        data = make_graph_hte(
            n=50,
            num_nodes_range=(5, 15),
            num_features=8,
            seed=42,
        )

        assert len(data.node_features) == 50
        assert len(data.edge_indices) == 50
        assert len(data.treatment) == 50
        assert isinstance(data.true_ate, float)

    def test_make_graph_hte_heterogeneity(self):
        """Test different graph heterogeneity types."""
        for het in ["density", "size", "centrality", "complex"]:
            data = make_graph_hte(
                n=30,
                num_nodes_range=(5, 10),
                heterogeneity=het,
                seed=42,
            )

            assert len(data.node_features) == 30

    def test_multimodal_to_tensors(self):
        """Test conversion to PyTorch tensors."""
        image_data = make_image_hte(n=10, image_size=8, seed=42)
        tensors = multimodal_to_tensors(image_data)

        assert isinstance(tensors["images"], torch.Tensor)
        assert isinstance(tensors["treatment"], torch.Tensor)
        assert tensors["images"].shape == (10, 3, 8, 8)


class TestArchitectureRegistryComplete:
    """Test that all new architectures are registered."""

    def test_all_architectures_available(self):
        """Test all architectures are in registry."""
        available = ArchitectureRegistry.available()

        assert "cnn" in available
        assert "resnet" in available
        assert "text" in available
        assert "bow" in available
        assert "gnn" in available
        assert "mlp" in available
        assert "transformer" in available
        assert "lstm" in available


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
