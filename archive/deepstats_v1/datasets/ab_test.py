"""A/B test data generators with heterogeneous treatment effects.

This module generates synthetic data for A/B testing following the
enriched structural model: Y = a(X) + b(X) * T + epsilon.

Examples
--------
>>> from deepstats.datasets.ab_test import make_ab_test
>>> data = make_ab_test(n=1000, seed=42)
>>> print(f"True ATE: {data['true_ate']:.3f}")
>>> print(f"Columns: {list(data['data'].columns)}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class ABTestData:
    """Container for A/B test data.

    Attributes
    ----------
    data : pd.DataFrame
        DataFrame with outcome Y, treatment T, and covariates.
    true_ate : float
        True average treatment effect E[b(X)].
    true_ite : np.ndarray
        True individual treatment effects b(X_i).
    true_baseline : np.ndarray
        True baseline predictions a(X_i).
    a_function : callable
        The true baseline function a(X).
    b_function : callable
        The true treatment effect function b(X).
    """

    data: pd.DataFrame
    true_ate: float
    true_ite: np.ndarray
    true_baseline: np.ndarray
    a_function: callable
    b_function: callable


def make_ab_test(
    n: int = 5000,
    p: int = 10,
    treatment_prob: float = 0.5,
    heterogeneity: Literal["none", "linear", "nonlinear", "complex"] = "nonlinear",
    noise_scale: float = 1.0,
    seed: int | None = None,
) -> ABTestData:
    """Generate A/B test data with heterogeneous treatment effects.

    Model: Y = a(X) + b(X) * T + epsilon

    Parameters
    ----------
    n : int, default=5000
        Number of observations.
    p : int, default=10
        Number of covariates.
    treatment_prob : float, default=0.5
        Probability of treatment (for randomized experiment).
    heterogeneity : str, default="nonlinear"
        Type of heterogeneity in treatment effect:
        - "none": Constant treatment effect
        - "linear": Linear in covariates
        - "nonlinear": Polynomial/interaction terms
        - "complex": Highly nonlinear (sine, exp, etc.)
    noise_scale : float, default=1.0
        Scale of the noise term.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ABTestData
        Container with data, true ATE, and true ITEs.

    Examples
    --------
    >>> result = make_ab_test(n=1000, heterogeneity="nonlinear", seed=42)
    >>> print(result.data.head())
    >>> print(f"True ATE: {result.true_ate:.3f}")
    """
    rng = np.random.default_rng(seed)

    # Generate covariates
    X = rng.standard_normal((n, p))

    # Treatment assignment (randomized)
    T = rng.binomial(1, treatment_prob, n)

    # Define parameter functions based on heterogeneity type
    if heterogeneity == "none":
        # Constant treatment effect
        def a_func(X):
            return 0.2 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2]

        def b_func(X):
            return np.ones(len(X)) * 2.0  # Constant ATE = 2

    elif heterogeneity == "linear":
        # Linear heterogeneity
        def a_func(X):
            return 0.2 * X[:, 0] - 1.3 * X[:, 1] - 0.5 * X[:, 2]

        def b_func(X):
            return 2.0 - X[:, 1] + 0.5 * X[:, 0]

    elif heterogeneity == "nonlinear":
        # Polynomial heterogeneity (from MisraLab)
        def a_func(X):
            return 0.2 * X[:, 0] - 1.3 * X[:, 1] - 0.5 * X[:, 2]

        def b_func(X):
            return 2.0 - X[:, 1] + 0.25 * X[:, 0] ** 3

    else:  # complex
        # Highly nonlinear
        def a_func(X):
            return (
                0.5 * np.sin(2 * X[:, 0])
                + 0.3 * X[:, 1] ** 2
                - 0.4 * X[:, 2]
                + 0.2 * X[:, 0] * X[:, 1]
            )

        def b_func(X):
            return (
                1.5
                + np.sin(X[:, 0])
                - 0.5 * X[:, 1] ** 2
                + 0.3 * np.exp(-X[:, 2] ** 2)
                + 0.25 * X[:, 0] ** 3
            )

    # Compute true values
    true_a = a_func(X)
    true_b = b_func(X)

    # Generate outcome
    epsilon = rng.standard_normal(n) * noise_scale
    Y = true_a + true_b * T + epsilon

    # True ATE
    true_ate = true_b.mean()

    # Create DataFrame
    columns = {f"X{i+1}": X[:, i] for i in range(p)}
    columns["T"] = T
    columns["Y"] = Y
    data = pd.DataFrame(columns)

    return ABTestData(
        data=data,
        true_ate=true_ate,
        true_ite=true_b,
        true_baseline=true_a,
        a_function=a_func,
        b_function=b_func,
    )


def make_ab_test_binary(
    n: int = 5000,
    p: int = 10,
    treatment_prob: float = 0.5,
    heterogeneity: Literal["none", "linear", "nonlinear", "complex"] = "nonlinear",
    seed: int | None = None,
) -> ABTestData:
    """Generate A/B test data with binary outcome.

    Model: P(Y=1) = sigmoid(a(X) + b(X) * T)

    Parameters
    ----------
    n : int, default=5000
        Number of observations.
    p : int, default=10
        Number of covariates.
    treatment_prob : float, default=0.5
        Probability of treatment.
    heterogeneity : str, default="nonlinear"
        Type of heterogeneity in treatment effect.
    seed : int, optional
        Random seed.

    Returns
    -------
    ABTestData
        Container with binary outcome data.

    Examples
    --------
    >>> result = make_ab_test_binary(n=1000, seed=42)
    >>> print(result.data["Y"].value_counts())
    """
    rng = np.random.default_rng(seed)

    # Generate covariates
    X = rng.standard_normal((n, p))

    # Treatment assignment
    T = rng.binomial(1, treatment_prob, n)

    # Parameter functions (scaled for logit)
    if heterogeneity == "none":

        def a_func(X):
            return 0.1 * X[:, 0] - 0.2 * X[:, 1]

        def b_func(X):
            return np.ones(len(X)) * 0.5  # Constant log-odds effect

    elif heterogeneity == "linear":

        def a_func(X):
            return 0.1 * X[:, 0] - 0.3 * X[:, 1] - 0.1 * X[:, 2]

        def b_func(X):
            return 0.5 - 0.3 * X[:, 1] + 0.2 * X[:, 0]

    elif heterogeneity == "nonlinear":

        def a_func(X):
            return 0.1 * X[:, 0] - 0.3 * X[:, 1] - 0.1 * X[:, 2]

        def b_func(X):
            return 0.5 - 0.3 * X[:, 1] + 0.1 * X[:, 0] ** 2

    else:  # complex

        def a_func(X):
            return 0.2 * np.sin(X[:, 0]) + 0.1 * X[:, 1] ** 2 - 0.2 * X[:, 2]

        def b_func(X):
            return 0.5 + 0.3 * np.sin(X[:, 0]) - 0.2 * X[:, 1] ** 2

    # Compute logits
    true_a = a_func(X)
    true_b = b_func(X)
    logit = true_a + true_b * T

    # Generate binary outcome
    prob = 1 / (1 + np.exp(-logit))
    Y = rng.binomial(1, prob).astype(float)

    # True ATE (on log-odds scale)
    true_ate = true_b.mean()

    # Create DataFrame
    columns = {f"X{i+1}": X[:, i] for i in range(p)}
    columns["T"] = T
    columns["Y"] = Y
    data = pd.DataFrame(columns)

    return ABTestData(
        data=data,
        true_ate=true_ate,
        true_ite=true_b,
        true_baseline=true_a,
        a_function=a_func,
        b_function=b_func,
    )


def make_ab_test_highdim(
    n: int = 5000,
    p: int = 50,
    p_relevant: int = 5,
    treatment_prob: float = 0.5,
    noise_scale: float = 1.0,
    seed: int | None = None,
) -> ABTestData:
    """Generate high-dimensional A/B test data.

    Only a subset of covariates affect the treatment effect,
    simulating realistic high-dimensional settings.

    Parameters
    ----------
    n : int, default=5000
        Number of observations.
    p : int, default=50
        Total number of covariates.
    p_relevant : int, default=5
        Number of covariates that actually affect treatment effect.
    treatment_prob : float, default=0.5
        Probability of treatment.
    noise_scale : float, default=1.0
        Scale of noise.
    seed : int, optional
        Random seed.

    Returns
    -------
    ABTestData
        Container with high-dimensional data.
    """
    rng = np.random.default_rng(seed)

    # Generate covariates
    X = rng.standard_normal((n, p))

    # Treatment assignment
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

    # Compute true values
    true_a = a_func(X)
    true_b = b_func(X)

    # Generate outcome
    epsilon = rng.standard_normal(n) * noise_scale
    Y = true_a + true_b * T + epsilon

    # True ATE
    true_ate = true_b.mean()

    # Create DataFrame
    columns = {f"X{i+1}": X[:, i] for i in range(p)}
    columns["T"] = T
    columns["Y"] = Y
    data = pd.DataFrame(columns)

    return ABTestData(
        data=data,
        true_ate=true_ate,
        true_ite=true_b,
        true_baseline=true_a,
        a_function=a_func,
        b_function=b_func,
    )
