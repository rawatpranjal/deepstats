"""Neural network architectures for deepstats.

This module provides network architectures for use in enriched
structural models following Farrell, Liang, Misra (2021, 2023).

Available Architectures
-----------------------
Tabular:
- MLP: Multi-layer perceptron (feedforward)
- TabularTransformer: Transformer for tabular data
- LSTMBackbone: LSTM for sequential/tabular data

Image:
- CNNBackbone: Convolutional neural network
- ResNetBackbone: ResNet-style with residual connections

Text:
- TextBackbone: Embeddings + Transformer/LSTM + pooling
- BagOfWordsBackbone: Simple embedding averaging

Graph:
- GNNBackbone: Graph neural network with message passing

Architecture Registry
---------------------
Use ArchitectureRegistry to dynamically create networks:

>>> from deepstats.networks import ArchitectureRegistry
>>> backbone = ArchitectureRegistry.create("transformer", input_dim=50)
>>> backbone = ArchitectureRegistry.create("cnn", in_channels=3, image_size=32)
>>> backbone = ArchitectureRegistry.create("text", vocab_size=10000)
>>> backbone = ArchitectureRegistry.create("gnn", input_dim=16)

Examples
--------
>>> from deepstats.networks import MLPBackbone, ArchitectureRegistry
>>>
>>> # Direct instantiation
>>> backbone = MLPBackbone(input_dim=10, hidden_dims=[64, 32])
>>>
>>> # Via registry
>>> backbone = ArchitectureRegistry.create("mlp", input_dim=10)
>>>
>>> # List available architectures
>>> print(ArchitectureRegistry.available())
['bow', 'cnn', 'gcn', 'gnn', 'lstm', 'mlp', 'resnet', 'text', 'transformer']
"""

# Base classes
from .base import BackboneNetwork, NetworkArchitecture, ParameterNetwork

# Registry
from .registry import ArchitectureRegistry, create_backbone

# MLP architectures
from .mlp import MLP, MLPBackbone, MLPClassifier, create_network

# Transformer architecture
from .transformer import TabularTransformer

# LSTM architecture
from .lstm import LSTMBackbone

# CNN architectures
from .cnn import CNNBackbone, ResNetBackbone

# Text architectures
from .text import TextBackbone, BagOfWordsBackbone

# Graph architectures
from .gnn import GNNBackbone, GraphData, create_random_graph

__all__ = [
    # Base classes
    "NetworkArchitecture",
    "BackboneNetwork",
    "ParameterNetwork",
    # Registry
    "ArchitectureRegistry",
    "create_backbone",
    # MLP
    "MLP",
    "MLPBackbone",
    "MLPClassifier",
    "create_network",
    # Transformer
    "TabularTransformer",
    # LSTM
    "LSTMBackbone",
    # CNN
    "CNNBackbone",
    "ResNetBackbone",
    # Text
    "TextBackbone",
    "BagOfWordsBackbone",
    # Graph
    "GNNBackbone",
    "GraphData",
    "create_random_graph",
]
