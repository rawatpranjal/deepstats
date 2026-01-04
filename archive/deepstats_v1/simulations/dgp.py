"""Extended data generating processes for simulation studies.

This module provides DGPs designed to test specific scenarios:
- Overfitting: small n, high p, simple true function
- Underfitting: large n, complex true function
- Balanced: appropriate complexity for sample size
- High noise: low signal-to-noise ratio
- Sparse: many irrelevant covariates
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class SimulationData:
    """Container for simulation data.

    Attributes
    ----------
    data : pd.DataFrame
        DataFrame with outcome Y, treatment T, and covariates.
    true_ate : float
        True average treatment effect.
    true_ite : np.ndarray
        True individual treatment effects.
    true_baseline : np.ndarray
        True baseline predictions a(X).
    scenario : str
        Name of the scenario.
    """

    data: pd.DataFrame
    true_ate: float
    true_ite: np.ndarray
    true_baseline: np.ndarray
    scenario: str


def make_overfit_scenario(
    seed: int,
    n: int = 100,
    p: int = 50,
    treatment_prob: float = 0.5,
    noise_scale: float = 0.5,
) -> SimulationData:
    """Generate data prone to overfitting.

    Small sample size with high dimensionality and simple true functions.
    Models should struggle to generalize.

    Parameters
    ----------
    seed : int
        Random seed.
    n : int, default=100
        Sample size (deliberately small).
    p : int, default=50
        Number of covariates (high relative to n).
    treatment_prob : float, default=0.5
        Probability of treatment.
    noise_scale : float, default=0.5
        Scale of noise (low to make overfitting evident).

    Returns
    -------
    SimulationData
        Container with data and true values.
    """
    rng = np.random.default_rng(seed)

    # Generate covariates
    X = rng.standard_normal((n, p))

    # Treatment
    T = rng.binomial(1, treatment_prob, n)

    # Simple true functions (only depend on first 2 covariates)
    def a_func(X):
        return 0.5 * X[:, 0] - 0.3 * X[:, 1]

    def b_func(X):
        return 1.5 + 0.5 * X[:, 0]  # Simple linear

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
        scenario="overfit",
    )


def make_underfit_scenario(
    seed: int,
    n: int = 5000,
    p: int = 10,
    treatment_prob: float = 0.5,
    noise_scale: float = 1.0,
) -> SimulationData:
    """Generate data prone to underfitting.

    Large sample size with complex true functions that simple models
    cannot capture.

    Parameters
    ----------
    seed : int
        Random seed.
    n : int, default=5000
        Sample size (large).
    p : int, default=10
        Number of covariates.
    treatment_prob : float, default=0.5
        Probability of treatment.
    noise_scale : float, default=1.0
        Scale of noise.

    Returns
    -------
    SimulationData
        Container with data and true values.
    """
    rng = np.random.default_rng(seed)

    # Generate covariates
    X = rng.standard_normal((n, p))

    # Treatment
    T = rng.binomial(1, treatment_prob, n)

    # Complex true functions with interactions and nonlinearities
    def a_func(X):
        return (
            0.5 * np.sin(2 * X[:, 0])
            + 0.3 * X[:, 1] ** 2
            - 0.4 * X[:, 2]
            + 0.2 * X[:, 0] * X[:, 1]
            + 0.15 * np.exp(-X[:, 3] ** 2)
            - 0.25 * X[:, 4] ** 3
            + 0.1 * np.cos(X[:, 5])
        )

    def b_func(X):
        return (
            2.0
            + np.sin(X[:, 0])
            - 0.5 * X[:, 1] ** 2
            + 0.3 * np.exp(-X[:, 2] ** 2)
            + 0.25 * X[:, 0] ** 3
            + 0.2 * X[:, 0] * X[:, 1]
            - 0.15 * X[:, 3] * X[:, 4]
        )

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
        scenario="underfit",
    )


def make_balanced_scenario(
    seed: int,
    n: int = 2000,
    p: int = 5,
    treatment_prob: float = 0.5,
    noise_scale: float = 1.0,
    complexity: Literal["low", "medium", "high"] = "medium",
) -> SimulationData:
    """Generate balanced data for good fit.

    Appropriate sample size and complexity for successful estimation.

    Parameters
    ----------
    seed : int
        Random seed.
    n : int, default=2000
        Sample size.
    p : int, default=5
        Number of covariates.
    treatment_prob : float, default=0.5
        Probability of treatment.
    noise_scale : float, default=1.0
        Scale of noise.
    complexity : str, default="medium"
        Complexity of true functions.

    Returns
    -------
    SimulationData
        Container with data and true values.
    """
    rng = np.random.default_rng(seed)

    # Generate covariates
    X = rng.standard_normal((n, p))

    # Treatment
    T = rng.binomial(1, treatment_prob, n)

    # True functions based on complexity
    if complexity == "low":

        def a_func(X):
            return 0.5 * X[:, 0] - 0.3 * X[:, 1]

        def b_func(X):
            return np.ones(len(X)) * 2.0

    elif complexity == "medium":

        def a_func(X):
            return 0.2 * X[:, 0] - 1.3 * X[:, 1] - 0.5 * X[:, 2]

        def b_func(X):
            return 2.0 - X[:, 1] + 0.25 * X[:, 0] ** 2

    else:  # high

        def a_func(X):
            return (
                0.3 * np.sin(X[:, 0])
                + 0.2 * X[:, 1] ** 2
                - 0.3 * X[:, 2]
                + 0.1 * X[:, 0] * X[:, 1]
            )

        def b_func(X):
            return (
                1.5
                + 0.5 * np.sin(X[:, 0])
                - 0.3 * X[:, 1] ** 2
                + 0.2 * X[:, 0] ** 2
            )

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
        scenario=f"balanced_{complexity}",
    )


def make_high_noise_scenario(
    seed: int,
    n: int = 2000,
    p: int = 5,
    treatment_prob: float = 0.5,
    signal_to_noise: float = 0.5,
) -> SimulationData:
    """Generate high-noise data.

    Low signal-to-noise ratio makes estimation challenging.

    Parameters
    ----------
    seed : int
        Random seed.
    n : int, default=2000
        Sample size.
    p : int, default=5
        Number of covariates.
    treatment_prob : float, default=0.5
        Probability of treatment.
    signal_to_noise : float, default=0.5
        Ratio of signal variance to noise variance.

    Returns
    -------
    SimulationData
        Container with data and true values.
    """
    rng = np.random.default_rng(seed)

    # Generate covariates
    X = rng.standard_normal((n, p))

    # Treatment
    T = rng.binomial(1, treatment_prob, n)

    # True functions
    def a_func(X):
        return 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2]

    def b_func(X):
        return 2.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1]

    true_a = a_func(X)
    true_b = b_func(X)

    # Calculate signal variance
    signal = true_a + true_b * T
    signal_var = np.var(signal)

    # Set noise to achieve desired SNR
    noise_var = signal_var / signal_to_noise
    noise_scale = np.sqrt(noise_var)

    # Generate outcome
    epsilon = rng.standard_normal(n) * noise_scale
    Y = signal + epsilon

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
        scenario=f"high_noise_snr{signal_to_noise}",
    )


def make_sparse_scenario(
    seed: int,
    n: int = 2000,
    p: int = 100,
    p_relevant: int = 5,
    treatment_prob: float = 0.5,
    noise_scale: float = 1.0,
) -> SimulationData:
    """Generate sparse high-dimensional data.

    Many covariates but only a few are relevant.

    Parameters
    ----------
    seed : int
        Random seed.
    n : int, default=2000
        Sample size.
    p : int, default=100
        Total number of covariates.
    p_relevant : int, default=5
        Number of relevant covariates.
    treatment_prob : float, default=0.5
        Probability of treatment.
    noise_scale : float, default=1.0
        Scale of noise.

    Returns
    -------
    SimulationData
        Container with data and true values.
    """
    rng = np.random.default_rng(seed)

    # Generate covariates
    X = rng.standard_normal((n, p))

    # Treatment
    T = rng.binomial(1, treatment_prob, n)

    # Only first p_relevant covariates matter
    def a_func(X):
        result = np.zeros(len(X))
        for i in range(min(p_relevant, X.shape[1])):
            result += 0.3 * ((-1) ** i) * X[:, i]
        return result

    def b_func(X):
        result = np.ones(len(X)) * 2.0
        for i in range(min(p_relevant, X.shape[1])):
            result += 0.2 * ((-1) ** i) * X[:, i]
            if i < 2:
                result += 0.1 * X[:, i] ** 2
        return result

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
        scenario=f"sparse_p{p}_relevant{p_relevant}",
    )


def make_confounded_scenario(
    seed: int,
    n: int = 2000,
    p: int = 5,
    confounding_strength: float = 0.5,
    noise_scale: float = 1.0,
) -> SimulationData:
    """Generate data with confounding (non-random treatment).

    Treatment assignment depends on covariates, violating randomization.
    This tests robustness to propensity score misspecification.

    Parameters
    ----------
    seed : int
        Random seed.
    n : int, default=2000
        Sample size.
    p : int, default=5
        Number of covariates.
    confounding_strength : float, default=0.5
        Strength of confounding (0=random, 1=strong).
    noise_scale : float, default=1.0
        Scale of noise.

    Returns
    -------
    SimulationData
        Container with data and true values.
    """
    rng = np.random.default_rng(seed)

    # Generate covariates
    X = rng.standard_normal((n, p))

    # Treatment depends on X (confounding)
    propensity_logit = confounding_strength * (0.5 * X[:, 0] - 0.3 * X[:, 1])
    propensity = 1 / (1 + np.exp(-propensity_logit))
    T = rng.binomial(1, propensity)

    # True functions
    def a_func(X):
        return 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2]

    def b_func(X):
        return 2.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1]

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
        scenario=f"confounded_{confounding_strength}",
    )
