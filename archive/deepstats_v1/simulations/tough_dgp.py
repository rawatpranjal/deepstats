"""Tough high-dimensional DGP patterns.

This module provides challenging data generating processes with complex
nonlinear patterns that require neural networks with sufficient capacity
to recover. These are designed to test the limits of HTE estimation.

Pattern Types
-------------
- deep_interactions: 3-way and 4-way covariate interactions
- threshold: Step functions and threshold effects
- multifreq: Multi-frequency periodic patterns
- sparse_nonlinear: Complex effects from few variables among many
- mixed: Combination of all pattern types (most challenging)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .dgp import SimulationData


def make_tough_highdim_scenario(
    seed: int,
    n: int = 5000,
    p: int = 100,
    sparsity: int = 10,
    pattern: Literal[
        "deep_interactions", "threshold", "multifreq",
        "sparse_nonlinear", "mixed"
    ] = "mixed",
    noise_scale: float = 1.0,
    treatment_prob: float = 0.5,
) -> SimulationData:
    """Generate tough high-dimensional patterns for HTE recovery.

    These patterns are designed to challenge neural network estimators
    with complex nonlinearities, interactions, and threshold effects.

    Parameters
    ----------
    seed : int
        Random seed.
    n : int, default=5000
        Sample size (larger samples help recover complex patterns).
    p : int, default=100
        Total number of covariates.
    sparsity : int, default=10
        Number of relevant variables (rest are noise).
    pattern : str, default="mixed"
        Type of pattern:
        - "deep_interactions": 3-way and 4-way interactions
        - "threshold": Step functions and thresholds
        - "multifreq": Multi-frequency periodic patterns
        - "sparse_nonlinear": Complex effects from few variables
        - "mixed": Combination of all (most challenging)
    noise_scale : float, default=1.0
        Scale of outcome noise.
    treatment_prob : float, default=0.5
        Probability of treatment.

    Returns
    -------
    SimulationData
        Container with data and true values.

    Examples
    --------
    >>> data = make_tough_highdim_scenario(
    ...     seed=42, n=5000, p=100, pattern="mixed"
    ... )
    >>> print(f"True ATE: {data.true_ate:.4f}")
    >>> print(f"ITE std: {data.true_ite.std():.4f}")
    """
    rng = np.random.default_rng(seed)

    # Generate covariates
    X = rng.standard_normal((n, p))

    # Select relevant variable indices
    idx = np.arange(min(sparsity, p))

    # Treatment
    T = rng.binomial(1, treatment_prob, n)

    # Get pattern functions
    if pattern == "deep_interactions":
        a_func, b_func = _deep_interactions_pattern(idx)
    elif pattern == "threshold":
        a_func, b_func = _threshold_pattern(idx)
    elif pattern == "multifreq":
        a_func, b_func = _multifreq_pattern(idx)
    elif pattern == "sparse_nonlinear":
        a_func, b_func = _sparse_nonlinear_pattern(idx)
    elif pattern == "mixed":
        a_func, b_func = _mixed_pattern(idx)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    true_a = a_func(X)
    true_b = b_func(X)

    # Generate outcome
    epsilon = rng.standard_normal(n) * noise_scale
    Y = true_a + true_b * T + epsilon

    # Create DataFrame
    columns = {f"X{i+1}": X[:, i] for i in range(p)}
    columns["T"] = T
    columns["Y"] = Y
    data = pd.DataFrame(columns)

    return SimulationData(
        data=data,
        true_ate=float(np.mean(true_b)),
        true_ite=true_b,
        true_baseline=true_a,
        scenario=f"tough_{pattern}_p{p}_s{sparsity}",
    )


def _deep_interactions_pattern(idx):
    """Deep 3-way and 4-way interactions."""

    def a_func(X):
        x1, x2, x3, x4 = X[:, idx[0]], X[:, idx[1]], X[:, idx[2]], X[:, idx[3]]
        return (
            0.5 * x1 * x2 * x3 +                    # 3-way interaction
            0.3 * x2 * x3 * x4 -                    # 3-way interaction
            0.2 * x1 * x2 * x3 * x4 +              # 4-way interaction
            0.4 * x1 * x2 +                        # 2-way for easier learning
            0.25 * x3 - 0.15 * x4                   # Linear terms
        )

    def b_func(X):
        x1, x2, x3, x4 = X[:, idx[0]], X[:, idx[1]], X[:, idx[2]], X[:, idx[3]]
        x5 = X[:, idx[4]] if len(idx) > 4 else X[:, idx[0]]
        return (
            2.0 +
            0.5 * x1 * x2 -                        # 2-way interaction
            0.3 * x2 * x3 +                        # 2-way interaction
            0.2 * x1 * x3 * x4 +                   # 3-way interaction
            0.15 * x1 * x2 * x5                    # 3-way interaction
        )

    return a_func, b_func


def _threshold_pattern(idx):
    """Threshold and step function effects."""

    def a_func(X):
        x1, x2, x3 = X[:, idx[0]], X[:, idx[1]], X[:, idx[2]]
        return (
            np.where(x1 > 0, 1.0, -1.0) +              # Step function
            np.where(x2 > 0.5, 0.5, 0) +              # Threshold
            np.where(x3 < -0.5, -0.3, 0.3) +          # Threshold
            0.2 * x1 * (x1 > 0)                        # Hinge
        )

    def b_func(X):
        x1, x2, x3 = X[:, idx[0]], X[:, idx[1]], X[:, idx[2]]
        x4 = X[:, idx[3]] if len(idx) > 3 else X[:, idx[0]]
        return (
            2.0 * (x1 > 0).astype(float) +            # Treatment effect only if x1 > 0
            1.5 * (x2 > 0).astype(float) * (x3 < 0).astype(float) +  # Interaction threshold
            0.5 * np.clip(x4, -1, 1) +                # Clipped linear
            0.3 * np.where(np.abs(x1) < 0.5, 1, 0)    # Band indicator
        )

    return a_func, b_func


def _multifreq_pattern(idx):
    """Multi-frequency periodic patterns."""

    def a_func(X):
        x1, x2, x3 = X[:, idx[0]], X[:, idx[1]], X[:, idx[2]]
        return (
            np.sin(x1) +                              # Low frequency
            0.5 * np.sin(3 * x2) +                    # Medium frequency
            0.3 * np.sin(7 * x3) +                    # High frequency
            0.2 * np.cos(2 * x1) * np.sin(x2)         # Interaction periodic
        )

    def b_func(X):
        x1, x2, x3 = X[:, idx[0]], X[:, idx[1]], X[:, idx[2]]
        x4 = X[:, idx[3]] if len(idx) > 3 else X[:, idx[0]]
        return (
            1.5 +
            np.cos(2 * x1) -                          # Periodic
            0.5 * np.cos(5 * x2) +                    # Higher frequency
            0.2 * np.sin(x1 * x2) +                   # Product frequency
            0.3 * np.sin(3 * x3) * np.cos(x4)         # Interaction periodic
        )

    return a_func, b_func


def _sparse_nonlinear_pattern(idx):
    """Complex nonlinear effects from few variables."""

    def a_func(X):
        x1, x2, x3 = X[:, idx[0]], X[:, idx[1]], X[:, idx[2]]
        x4, x5 = X[:, idx[3]], X[:, idx[4]] if len(idx) > 4 else (X[:, idx[0]], X[:, idx[1]])
        return (
            np.exp(-x1**2) * np.sin(2 * x2) +         # Gaussian-modulated sinusoid
            0.5 * np.tanh(x3) * (x4 > 0).astype(float) +  # Tanh with threshold
            0.3 * x5**3 / (1 + x5**2) -               # Rational function
            0.2 * np.log(1 + np.exp(x1))              # Softplus
        )

    def b_func(X):
        x1, x2, x3 = X[:, idx[0]], X[:, idx[1]], X[:, idx[2]]
        x4, x5 = X[:, idx[3]], X[:, idx[4]] if len(idx) > 4 else (X[:, idx[0]], X[:, idx[1]])
        return (
            2.0 +
            np.sin(x1) * np.cos(x2) -                 # Periodic interaction
            0.5 * np.exp(-x3**2) * x4 +               # Gaussian interaction
            0.3 * np.tanh(x1 + x2) +                  # Nonlinear combination
            0.2 * np.sign(x5) * np.abs(x5)**0.5       # Signed square root
        )

    return a_func, b_func


def _mixed_pattern(idx):
    """Most challenging: mix of all pattern types."""

    def a_func(X):
        x = [X[:, idx[i]] for i in range(min(8, len(idx)))]
        # Pad with first variable if not enough
        while len(x) < 8:
            x.append(X[:, idx[0]])
        x1, x2, x3, x4, x5, x6, x7, x8 = x

        return (
            # 3-way interaction
            0.5 * x1 * x2 * x3 +
            # Periodic
            np.sin(2 * x4) +
            # Threshold + polynomial
            np.where(x5 > 0, 1, -1).astype(float) * x6**2 +
            # Gaussian
            0.3 * np.exp(-x7**2) +
            # High frequency
            0.2 * np.sin(5 * x8)
        )

    def b_func(X):
        x = [X[:, idx[i]] for i in range(min(8, len(idx)))]
        while len(x) < 8:
            x.append(X[:, idx[0]])
        x1, x2, x3, x4, x5, x6, x7, x8 = x

        return (
            2.0 +
            # Periodic + threshold
            np.cos(x1) * (x2 > 0).astype(float) +
            # Interaction + cubic
            0.5 * x3 * x4 - 0.3 * x5**3 +
            # Gaussian envelope
            0.2 * np.exp(-x6**2) +
            # Deep interaction
            0.15 * x1 * x2 * x7 +
            # Multi-frequency
            0.1 * np.sin(3 * x8)
        )

    return a_func, b_func


def make_deep_interaction_scenario(
    seed: int,
    n: int = 5000,
    p: int = 50,
) -> SimulationData:
    """Convenience function for deep interaction pattern."""
    return make_tough_highdim_scenario(
        seed=seed, n=n, p=p, pattern="deep_interactions"
    )


def make_threshold_scenario(
    seed: int,
    n: int = 5000,
    p: int = 50,
) -> SimulationData:
    """Convenience function for threshold pattern."""
    return make_tough_highdim_scenario(
        seed=seed, n=n, p=p, pattern="threshold"
    )


def make_multifreq_scenario(
    seed: int,
    n: int = 5000,
    p: int = 50,
) -> SimulationData:
    """Convenience function for multi-frequency pattern."""
    return make_tough_highdim_scenario(
        seed=seed, n=n, p=p, pattern="multifreq"
    )


def make_sparse_nonlinear_scenario(
    seed: int,
    n: int = 5000,
    p: int = 100,
    sparsity: int = 5,
) -> SimulationData:
    """Convenience function for sparse nonlinear pattern."""
    return make_tough_highdim_scenario(
        seed=seed, n=n, p=p, sparsity=sparsity, pattern="sparse_nonlinear"
    )


def make_mixed_tough_scenario(
    seed: int,
    n: int = 5000,
    p: int = 100,
) -> SimulationData:
    """Convenience function for mixed (hardest) pattern."""
    return make_tough_highdim_scenario(
        seed=seed, n=n, p=p, pattern="mixed"
    )
