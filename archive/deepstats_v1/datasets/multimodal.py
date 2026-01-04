"""Multimodal data generators for HTE with images, text, and graphs.

This module provides synthetic data generators for testing HTE estimation
with non-tabular inputs (images, text sequences, and graphs).

Examples
--------
>>> from deepstats.datasets.multimodal import make_image_hte
>>> data = make_image_hte(n=1000, image_size=32)
>>> print(data.images.shape)  # (1000, 3, 32, 32)
>>> print(data.true_ate)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


@dataclass
class ImageHTEData:
    """Container for image HTE data.

    Attributes
    ----------
    images : np.ndarray
        Image array of shape (n, channels, height, width).
    treatment : np.ndarray
        Treatment indicators (0 or 1).
    outcome : np.ndarray
        Outcome values.
    true_ate : float
        True average treatment effect.
    true_ite : np.ndarray
        True individual treatment effects.
    image_features : np.ndarray
        Underlying image features used to generate effects.
    """

    images: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    true_ate: float
    true_ite: np.ndarray
    image_features: np.ndarray


@dataclass
class TextHTEData:
    """Container for text HTE data.

    Attributes
    ----------
    tokens : np.ndarray
        Token indices of shape (n, seq_len).
    treatment : np.ndarray
        Treatment indicators.
    outcome : np.ndarray
        Outcome values.
    true_ate : float
        True average treatment effect.
    true_ite : np.ndarray
        True individual treatment effects.
    text_features : np.ndarray
        Underlying text features used to generate effects.
    vocab_size : int
        Size of vocabulary.
    """

    tokens: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    true_ate: float
    true_ite: np.ndarray
    text_features: np.ndarray
    vocab_size: int


@dataclass
class GraphHTEData:
    """Container for graph HTE data.

    Attributes
    ----------
    node_features : list[np.ndarray]
        List of node feature arrays, one per graph.
    edge_indices : list[np.ndarray]
        List of edge index arrays of shape (2, num_edges).
    treatment : np.ndarray
        Treatment indicators.
    outcome : np.ndarray
        Outcome values.
    true_ate : float
        True average treatment effect.
    true_ite : np.ndarray
        True individual treatment effects.
    graph_features : np.ndarray
        Underlying graph features used to generate effects.
    """

    node_features: list[np.ndarray]
    edge_indices: list[np.ndarray]
    treatment: np.ndarray
    outcome: np.ndarray
    true_ate: float
    true_ite: np.ndarray
    graph_features: np.ndarray


def make_image_hte(
    n: int = 1000,
    image_size: int = 32,
    channels: int = 3,
    treatment_prob: float = 0.5,
    heterogeneity: Literal["brightness", "texture", "color", "complex"] = "brightness",
    noise_scale: float = 1.0,
    seed: int | None = None,
) -> ImageHTEData:
    """Generate synthetic image data with heterogeneous treatment effects.

    Creates simple synthetic images where treatment effects depend on
    image features like brightness, texture, or color.

    Parameters
    ----------
    n : int, default=1000
        Number of observations.
    image_size : int, default=32
        Height and width of images.
    channels : int, default=3
        Number of color channels.
    treatment_prob : float, default=0.5
        Probability of treatment.
    heterogeneity : str, default="brightness"
        What image feature drives heterogeneity:
        - "brightness": Treatment effect varies with average brightness
        - "texture": Effect varies with edge density
        - "color": Effect varies with color distribution
        - "complex": Combination of features
    noise_scale : float, default=1.0
        Scale of outcome noise.
    seed : int, optional
        Random seed.

    Returns
    -------
    ImageHTEData
        Container with images, treatment, outcomes, and true effects.

    Examples
    --------
    >>> data = make_image_hte(n=500, image_size=28, heterogeneity="brightness")
    >>> print(data.images.shape)  # (500, 3, 28, 28)
    >>> print(f"True ATE: {data.true_ate:.3f}")
    """
    rng = np.random.default_rng(seed)

    # Generate underlying image features
    # These are the "covariates" that drive heterogeneity
    brightness = rng.uniform(0, 1, n)  # Average brightness
    texture = rng.uniform(0, 1, n)  # Edge density proxy
    color_balance = rng.uniform(-1, 1, n)  # Red-blue balance

    image_features = np.column_stack([brightness, texture, color_balance])

    # Generate synthetic images based on features
    images = np.zeros((n, channels, image_size, image_size))

    for i in range(n):
        # Base image with controlled brightness
        base = rng.uniform(0, brightness[i], (channels, image_size, image_size))

        # Add texture (edges/patterns)
        if texture[i] > 0.5:
            # Add stripe pattern
            pattern = np.sin(np.linspace(0, texture[i] * 10 * np.pi, image_size))
            base += 0.2 * pattern.reshape(1, 1, -1)

        # Adjust color balance
        if channels == 3:
            base[0] += 0.1 * color_balance[i]  # Red
            base[2] -= 0.1 * color_balance[i]  # Blue

        # Add random noise for realism
        base += rng.standard_normal(base.shape) * 0.05

        # Clip to valid range
        images[i] = np.clip(base, 0, 1)

    # Treatment assignment
    treatment = rng.binomial(1, treatment_prob, n)

    # Define treatment effect function based on heterogeneity type
    if heterogeneity == "brightness":
        # Brighter images have larger treatment effects
        true_b = 2.0 + 2.0 * brightness
    elif heterogeneity == "texture":
        # More textured images have larger effects
        true_b = 1.5 + 1.5 * texture
    elif heterogeneity == "color":
        # Color balance affects treatment effect
        true_b = 2.0 + 1.0 * color_balance
    else:  # complex
        # Combination of all features
        true_b = (
            1.5
            + 1.0 * brightness
            + 0.5 * texture
            + 0.3 * color_balance
            + 0.5 * brightness * texture
        )

    # Baseline function
    true_a = 0.5 * brightness - 0.3 * texture

    # Generate outcome
    epsilon = rng.standard_normal(n) * noise_scale
    outcome = true_a + true_b * treatment + epsilon

    return ImageHTEData(
        images=images.astype(np.float32),
        treatment=treatment,
        outcome=outcome.astype(np.float32),
        true_ate=float(np.mean(true_b)),
        true_ite=true_b.astype(np.float32),
        image_features=image_features.astype(np.float32),
    )


def make_text_hte(
    n: int = 1000,
    seq_len: int = 50,
    vocab_size: int = 1000,
    treatment_prob: float = 0.5,
    heterogeneity: Literal["length", "frequency", "pattern", "complex"] = "frequency",
    noise_scale: float = 1.0,
    seed: int | None = None,
) -> TextHTEData:
    """Generate synthetic text data with heterogeneous treatment effects.

    Creates synthetic token sequences where treatment effects depend on
    sequence properties like average token frequency or patterns.

    Parameters
    ----------
    n : int, default=1000
        Number of observations.
    seq_len : int, default=50
        Length of each sequence.
    vocab_size : int, default=1000
        Size of vocabulary.
    treatment_prob : float, default=0.5
        Probability of treatment.
    heterogeneity : str, default="frequency"
        What text feature drives heterogeneity:
        - "length": Effect varies with effective sequence length
        - "frequency": Effect varies with average token frequency
        - "pattern": Effect varies with repetition patterns
        - "complex": Combination of features
    noise_scale : float, default=1.0
        Scale of outcome noise.
    seed : int, optional
        Random seed.

    Returns
    -------
    TextHTEData
        Container with tokens, treatment, outcomes, and true effects.

    Examples
    --------
    >>> data = make_text_hte(n=500, seq_len=100)
    >>> print(data.tokens.shape)  # (500, 100)
    >>> print(f"True ATE: {data.true_ate:.3f}")
    """
    rng = np.random.default_rng(seed)

    # Define token frequency distribution (Zipf-like)
    token_freqs = 1.0 / np.arange(1, vocab_size + 1)
    token_freqs = token_freqs / token_freqs.sum()

    # Generate text features that will drive heterogeneity
    avg_frequency = rng.uniform(0, 1, n)  # Average token frequency
    effective_length = rng.uniform(0.5, 1.0, n)  # Fraction of non-padding
    repetition = rng.uniform(0, 1, n)  # Repetition score

    text_features = np.column_stack([avg_frequency, effective_length, repetition])

    # Generate token sequences
    tokens = np.zeros((n, seq_len), dtype=np.int64)

    for i in range(n):
        # Adjust token distribution based on features
        # Higher avg_frequency -> more common tokens
        adjusted_probs = token_freqs ** (1 - 0.5 * avg_frequency[i])
        adjusted_probs = adjusted_probs / adjusted_probs.sum()

        # Generate tokens
        actual_len = int(effective_length[i] * seq_len)
        tokens[i, :actual_len] = rng.choice(vocab_size, size=actual_len, p=adjusted_probs)

        # Add repetition if high repetition score
        if repetition[i] > 0.7:
            repeat_len = int(actual_len * 0.3)
            repeat_start = rng.integers(0, max(1, actual_len - repeat_len))
            repeat_source = tokens[i, repeat_start : repeat_start + repeat_len]
            insert_pos = rng.integers(0, actual_len)
            tokens[i, insert_pos : insert_pos + min(repeat_len, seq_len - insert_pos)] = (
                repeat_source[: min(repeat_len, seq_len - insert_pos)]
            )

    # Treatment assignment
    treatment = rng.binomial(1, treatment_prob, n)

    # Define treatment effect function
    if heterogeneity == "length":
        true_b = 1.5 + 1.5 * effective_length
    elif heterogeneity == "frequency":
        true_b = 2.0 - 1.0 * avg_frequency  # Rare tokens -> larger effect
    elif heterogeneity == "pattern":
        true_b = 1.5 + 1.0 * repetition
    else:  # complex
        true_b = (
            1.5
            + 0.5 * effective_length
            - 0.5 * avg_frequency
            + 0.3 * repetition
            + 0.2 * effective_length * (1 - avg_frequency)
        )

    # Baseline function
    true_a = 0.5 * avg_frequency - 0.3 * effective_length

    # Generate outcome
    epsilon = rng.standard_normal(n) * noise_scale
    outcome = true_a + true_b * treatment + epsilon

    return TextHTEData(
        tokens=tokens,
        treatment=treatment,
        outcome=outcome.astype(np.float32),
        true_ate=float(np.mean(true_b)),
        true_ite=true_b.astype(np.float32),
        text_features=text_features.astype(np.float32),
        vocab_size=vocab_size,
    )


def make_graph_hte(
    n: int = 1000,
    num_nodes_range: tuple[int, int] = (10, 50),
    num_features: int = 16,
    edge_prob: float = 0.1,
    treatment_prob: float = 0.5,
    heterogeneity: Literal["density", "size", "centrality", "complex"] = "density",
    noise_scale: float = 1.0,
    seed: int | None = None,
) -> GraphHTEData:
    """Generate synthetic graph data with heterogeneous treatment effects.

    Creates random graphs where treatment effects depend on graph
    structural properties like density or size.

    Parameters
    ----------
    n : int, default=1000
        Number of graphs.
    num_nodes_range : tuple, default=(10, 50)
        Range of number of nodes per graph.
    num_features : int, default=16
        Dimension of node features.
    edge_prob : float, default=0.1
        Base probability of edge between any two nodes.
    treatment_prob : float, default=0.5
        Probability of treatment.
    heterogeneity : str, default="density"
        What graph property drives heterogeneity:
        - "density": Effect varies with edge density
        - "size": Effect varies with number of nodes
        - "centrality": Effect varies with centrality measures
        - "complex": Combination of features
    noise_scale : float, default=1.0
        Scale of outcome noise.
    seed : int, optional
        Random seed.

    Returns
    -------
    GraphHTEData
        Container with graphs, treatment, outcomes, and true effects.

    Examples
    --------
    >>> data = make_graph_hte(n=500, num_nodes_range=(10, 30))
    >>> print(len(data.node_features))  # 500
    >>> print(f"True ATE: {data.true_ate:.3f}")
    """
    rng = np.random.default_rng(seed)

    node_features_list = []
    edge_indices_list = []

    # Graph-level features for heterogeneity
    sizes = []
    densities = []
    centralities = []

    for _ in range(n):
        # Random graph size
        num_nodes = rng.integers(num_nodes_range[0], num_nodes_range[1] + 1)
        sizes.append(num_nodes / num_nodes_range[1])  # Normalized

        # Random node features
        node_feats = rng.standard_normal((num_nodes, num_features)).astype(np.float32)
        node_features_list.append(node_feats)

        # Generate edges (Erdos-Renyi)
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if rng.random() < edge_prob:
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected

        if edges:
            edge_index = np.array(edges, dtype=np.int64).T
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)

        edge_indices_list.append(edge_index)

        # Compute graph properties
        num_edges = len(edges) // 2
        max_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_edges if max_edges > 0 else 0
        densities.append(density)

        # Simple centrality: average degree
        if edges:
            degrees = np.bincount(edge_index[0], minlength=num_nodes)
            avg_degree = degrees.mean() / num_nodes
        else:
            avg_degree = 0
        centralities.append(avg_degree)

    sizes = np.array(sizes)
    densities = np.array(densities)
    centralities = np.array(centralities)

    graph_features = np.column_stack([sizes, densities, centralities]).astype(np.float32)

    # Treatment assignment
    treatment = rng.binomial(1, treatment_prob, n)

    # Define treatment effect function
    if heterogeneity == "density":
        true_b = 2.0 + 2.0 * densities
    elif heterogeneity == "size":
        true_b = 1.5 + 1.5 * sizes
    elif heterogeneity == "centrality":
        true_b = 2.0 + 3.0 * centralities
    else:  # complex
        true_b = (
            1.5
            + 1.0 * densities
            + 0.5 * sizes
            + 0.5 * centralities
            + 0.3 * densities * sizes
        )

    # Baseline function
    true_a = 0.5 * sizes - 0.3 * densities

    # Generate outcome
    epsilon = rng.standard_normal(n) * noise_scale
    outcome = true_a + true_b * treatment + epsilon

    return GraphHTEData(
        node_features=node_features_list,
        edge_indices=edge_indices_list,
        treatment=treatment,
        outcome=outcome.astype(np.float32),
        true_ate=float(np.mean(true_b)),
        true_ite=true_b.astype(np.float32),
        graph_features=graph_features,
    )


@dataclass
class TimeSeriesHTEData:
    """Container for time series HTE data.

    Attributes
    ----------
    sequences : np.ndarray
        Time series array of shape (n, seq_len, n_features).
    treatment : np.ndarray
        Treatment indicators (0 or 1).
    outcome : np.ndarray
        Outcome values.
    true_ate : float
        True average treatment effect.
    true_ite : np.ndarray
        True individual treatment effects.
    ts_features : np.ndarray
        Underlying time series features (trend, volatility, seasonality).
    """

    sequences: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    true_ate: float
    true_ite: np.ndarray
    ts_features: np.ndarray


@dataclass
class ImageTextHTEData:
    """Container for multimodal image+text HTE data.

    Attributes
    ----------
    images : np.ndarray
        Image array of shape (n, channels, height, width).
    tokens : np.ndarray
        Token indices of shape (n, seq_len).
    treatment : np.ndarray
        Treatment indicators (0 or 1).
    outcome : np.ndarray
        Outcome values.
    true_ate : float
        True average treatment effect.
    true_ite : np.ndarray
        True individual treatment effects.
    image_features : np.ndarray
        Underlying image features.
    text_features : np.ndarray
        Underlying text features.
    vocab_size : int
        Size of vocabulary.
    """

    images: np.ndarray
    tokens: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    true_ate: float
    true_ite: np.ndarray
    image_features: np.ndarray
    text_features: np.ndarray
    vocab_size: int


def make_timeseries_hte(
    n: int = 1000,
    seq_len: int = 50,
    n_features: int = 3,
    treatment_prob: float = 0.5,
    heterogeneity: Literal["trend", "volatility", "seasonality", "complex"] = "trend",
    noise_scale: float = 1.0,
    seed: int | None = None,
) -> TimeSeriesHTEData:
    """Generate synthetic time series data with heterogeneous treatment effects.

    Creates synthetic multivariate time series where treatment effects depend on
    time series properties like trend strength, volatility, or seasonality.

    Parameters
    ----------
    n : int, default=1000
        Number of observations.
    seq_len : int, default=50
        Length of each time series.
    n_features : int, default=3
        Number of features/channels in time series.
    treatment_prob : float, default=0.5
        Probability of treatment.
    heterogeneity : str, default="trend"
        What time series feature drives heterogeneity:
        - "trend": Effect varies with trend strength
        - "volatility": Effect varies with volatility
        - "seasonality": Effect varies with seasonal pattern strength
        - "complex": Combination of features
    noise_scale : float, default=1.0
        Scale of outcome noise.
    seed : int, optional
        Random seed.

    Returns
    -------
    TimeSeriesHTEData
        Container with time series, treatment, outcomes, and true effects.
    """
    rng = np.random.default_rng(seed)

    # Generate underlying time series features
    trend_strength = rng.uniform(0, 1, n)
    volatility = rng.uniform(0.1, 2.0, n)
    seasonality = rng.uniform(0, 1, n)

    ts_features = np.column_stack([trend_strength, volatility, seasonality])

    # Generate time series
    sequences = np.zeros((n, seq_len, n_features))
    t = np.linspace(0, 1, seq_len)

    for i in range(n):
        for f in range(n_features):
            # Base trend
            trend = trend_strength[i] * t * 2

            # Seasonal component
            period = rng.integers(5, 15)
            seasonal = seasonality[i] * np.sin(2 * np.pi * t * period)

            # Random walk with volatility
            noise = rng.standard_normal(seq_len) * volatility[i] * 0.1
            random_walk = np.cumsum(noise)

            # Combine components
            sequences[i, :, f] = trend + seasonal + random_walk

    # Treatment assignment
    treatment = rng.binomial(1, treatment_prob, n)

    # Define treatment effect function
    if heterogeneity == "trend":
        true_b = 2.0 + 2.0 * trend_strength
    elif heterogeneity == "volatility":
        true_b = 1.5 + 1.0 * volatility
    elif heterogeneity == "seasonality":
        true_b = 2.0 + 1.5 * seasonality
    else:  # complex
        true_b = (
            1.5
            + 1.0 * trend_strength
            + 0.5 * volatility
            + 0.5 * seasonality
            + 0.3 * trend_strength * seasonality
        )

    # Baseline function
    true_a = 0.5 * trend_strength - 0.2 * volatility

    # Generate outcome
    epsilon = rng.standard_normal(n) * noise_scale
    outcome = true_a + true_b * treatment + epsilon

    return TimeSeriesHTEData(
        sequences=sequences.astype(np.float32),
        treatment=treatment,
        outcome=outcome.astype(np.float32),
        true_ate=float(np.mean(true_b)),
        true_ite=true_b.astype(np.float32),
        ts_features=ts_features.astype(np.float32),
    )


def make_image_text_hte(
    n: int = 1000,
    image_size: int = 32,
    channels: int = 3,
    seq_len: int = 50,
    vocab_size: int = 1000,
    treatment_prob: float = 0.5,
    heterogeneity: Literal["image_dominant", "text_dominant", "interaction", "complex"] = "interaction",
    noise_scale: float = 1.0,
    seed: int | None = None,
) -> ImageTextHTEData:
    """Generate synthetic multimodal image+text data with heterogeneous treatment effects.

    Creates paired image and text data where treatment effects depend on
    features from both modalities.

    Parameters
    ----------
    n : int, default=1000
        Number of observations.
    image_size : int, default=32
        Height and width of images.
    channels : int, default=3
        Number of color channels.
    seq_len : int, default=50
        Length of text sequences.
    vocab_size : int, default=1000
        Size of vocabulary.
    treatment_prob : float, default=0.5
        Probability of treatment.
    heterogeneity : str, default="interaction"
        What features drive heterogeneity:
        - "image_dominant": Effect mainly from image features
        - "text_dominant": Effect mainly from text features
        - "interaction": Effect from interaction of both modalities
        - "complex": Complex combination
    noise_scale : float, default=1.0
        Scale of outcome noise.
    seed : int, optional
        Random seed.

    Returns
    -------
    ImageTextHTEData
        Container with images, text, treatment, outcomes, and true effects.
    """
    rng = np.random.default_rng(seed)

    # Generate image features
    brightness = rng.uniform(0, 1, n)
    texture = rng.uniform(0, 1, n)
    color_balance = rng.uniform(-1, 1, n)
    image_features = np.column_stack([brightness, texture, color_balance])

    # Generate text features
    avg_frequency = rng.uniform(0, 1, n)
    effective_length = rng.uniform(0.5, 1.0, n)
    repetition = rng.uniform(0, 1, n)
    text_features = np.column_stack([avg_frequency, effective_length, repetition])

    # Generate images
    images = np.zeros((n, channels, image_size, image_size))
    for i in range(n):
        base = rng.uniform(0, brightness[i], (channels, image_size, image_size))
        if texture[i] > 0.5:
            pattern = np.sin(np.linspace(0, texture[i] * 10 * np.pi, image_size))
            base += 0.2 * pattern.reshape(1, 1, -1)
        if channels == 3:
            base[0] += 0.1 * color_balance[i]
            base[2] -= 0.1 * color_balance[i]
        base += rng.standard_normal(base.shape) * 0.05
        images[i] = np.clip(base, 0, 1)

    # Generate tokens
    token_freqs = 1.0 / np.arange(1, vocab_size + 1)
    token_freqs = token_freqs / token_freqs.sum()
    tokens = np.zeros((n, seq_len), dtype=np.int64)
    for i in range(n):
        adjusted_probs = token_freqs ** (1 - 0.5 * avg_frequency[i])
        adjusted_probs = adjusted_probs / adjusted_probs.sum()
        actual_len = int(effective_length[i] * seq_len)
        tokens[i, :actual_len] = rng.choice(vocab_size, size=actual_len, p=adjusted_probs)

    # Treatment assignment
    treatment = rng.binomial(1, treatment_prob, n)

    # Define treatment effect function based on heterogeneity
    if heterogeneity == "image_dominant":
        true_b = 2.0 + 1.5 * brightness + 0.5 * texture + 0.2 * avg_frequency
    elif heterogeneity == "text_dominant":
        true_b = 2.0 + 0.2 * brightness + 1.5 * effective_length - 0.5 * avg_frequency
    elif heterogeneity == "interaction":
        # Treatment effect depends on both modalities interacting
        true_b = (
            1.5
            + 0.5 * brightness
            + 0.5 * effective_length
            + 1.0 * brightness * effective_length  # Key interaction
            + 0.3 * texture * repetition
        )
    else:  # complex
        true_b = (
            1.5
            + 0.5 * brightness
            + 0.3 * texture
            + 0.4 * effective_length
            - 0.3 * avg_frequency
            + 0.5 * brightness * effective_length
            + 0.2 * color_balance * repetition
        )

    # Baseline function
    true_a = 0.3 * brightness - 0.2 * avg_frequency + 0.1 * texture

    # Generate outcome
    epsilon = rng.standard_normal(n) * noise_scale
    outcome = true_a + true_b * treatment + epsilon

    return ImageTextHTEData(
        images=images.astype(np.float32),
        tokens=tokens,
        treatment=treatment,
        outcome=outcome.astype(np.float32),
        true_ate=float(np.mean(true_b)),
        true_ite=true_b.astype(np.float32),
        image_features=image_features.astype(np.float32),
        text_features=text_features.astype(np.float32),
        vocab_size=vocab_size,
    )


def multimodal_to_tensors(
    data: ImageHTEData | TextHTEData | GraphHTEData | TimeSeriesHTEData | ImageTextHTEData,
    device: str = "cpu",
) -> dict:
    """Convert multimodal data to PyTorch tensors.

    Parameters
    ----------
    data : ImageHTEData, TextHTEData, or GraphHTEData
        Multimodal data container.
    device : str, default="cpu"
        Device to place tensors on.

    Returns
    -------
    dict
        Dictionary with tensor versions of the data.
    """
    result = {
        "treatment": torch.tensor(data.treatment, dtype=torch.float32, device=device),
        "outcome": torch.tensor(data.outcome, dtype=torch.float32, device=device),
        "true_ite": torch.tensor(data.true_ite, dtype=torch.float32, device=device),
    }

    if isinstance(data, ImageHTEData):
        result["images"] = torch.tensor(data.images, dtype=torch.float32, device=device)
        result["features"] = torch.tensor(
            data.image_features, dtype=torch.float32, device=device
        )
    elif isinstance(data, TextHTEData):
        result["tokens"] = torch.tensor(data.tokens, dtype=torch.long, device=device)
        result["features"] = torch.tensor(
            data.text_features, dtype=torch.float32, device=device
        )
    elif isinstance(data, GraphHTEData):
        result["node_features"] = [
            torch.tensor(nf, dtype=torch.float32, device=device)
            for nf in data.node_features
        ]
        result["edge_indices"] = [
            torch.tensor(ei, dtype=torch.long, device=device)
            for ei in data.edge_indices
        ]
        result["features"] = torch.tensor(
            data.graph_features, dtype=torch.float32, device=device
        )

    return result
