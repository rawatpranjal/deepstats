"""Placebo test scenarios for statistical validity.

Placebo tests verify estimator validity by checking that when true ATE = 0:
- Estimated ATE is approximately 0
- p-value > 0.05 (fail to reject null)
- 95% CI contains 0
- Coverage is correct across simulations

These tests are critical for validating that the inference is well-calibrated
and not producing spurious significant results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .dgp import SimulationData


def make_placebo_scenario(
    seed: int,
    n: int = 2000,
    p: int = 5,
    baseline_complexity: Literal["low", "medium", "high"] = "medium",
    noise_scale: float = 1.0,
) -> SimulationData:
    """Generate data with ZERO treatment effect (placebo).

    The model is: Y = a(X) + 0*T + epsilon
    True ATE = 0, True ITE = 0 for all units.

    The baseline a(X) can still be complex, but the treatment has no effect.
    This is used to verify that the estimator doesn't produce false positives.

    Parameters
    ----------
    seed : int
        Random seed.
    n : int, default=2000
        Sample size.
    p : int, default=5
        Number of covariates.
    baseline_complexity : str, default="medium"
        Complexity of baseline function a(X): "low", "medium", "high".
    noise_scale : float, default=1.0
        Scale of noise.

    Returns
    -------
    SimulationData
        Container with data and true values (all ITEs = 0).

    Examples
    --------
    >>> data = make_placebo_scenario(seed=42)
    >>> print(f"True ATE: {data.true_ate}")  # 0.0
    >>> print(f"True ITEs all zero: {np.allclose(data.true_ite, 0)}")  # True
    """
    rng = np.random.default_rng(seed)

    # Generate covariates
    X = rng.standard_normal((n, p))

    # Random treatment (no confounding)
    T = rng.binomial(1, 0.5, n)

    # Baseline function (can be complex)
    if baseline_complexity == "low":

        def a_func(X):
            return 0.5 * X[:, 0] - 0.3 * X[:, 1]

    elif baseline_complexity == "medium":

        def a_func(X):
            return (
                0.5 * X[:, 0]
                - 0.3 * X[:, 1]
                + 0.2 * X[:, 2]
                - 0.15 * X[:, 0] * X[:, 1]
            )

    else:  # high

        def a_func(X):
            return (
                0.3 * np.sin(X[:, 0])
                + 0.2 * X[:, 1] ** 2
                - 0.3 * X[:, 2]
                + 0.15 * X[:, 0] * X[:, 1]
                + 0.1 * np.exp(-X[:, 3] ** 2)
            )

    # Treatment effect is ZERO
    def b_func(X):
        return np.zeros(len(X))

    true_a = a_func(X)
    true_b = b_func(X)  # All zeros

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
        true_ate=0.0,  # Always zero for placebo
        true_ite=true_b,  # All zeros
        true_baseline=true_a,
        scenario=f"placebo_{baseline_complexity}",
    )


def make_near_zero_scenario(
    seed: int,
    n: int = 2000,
    p: int = 5,
    ate: float = 0.01,
    noise_scale: float = 1.0,
) -> SimulationData:
    """Generate data with near-zero treatment effect.

    Useful for testing power: with small true effect, estimator should
    sometimes fail to reject (low power is expected).

    Parameters
    ----------
    seed : int
        Random seed.
    n : int, default=2000
        Sample size.
    p : int, default=5
        Number of covariates.
    ate : float, default=0.01
        Small constant treatment effect.
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

    # Random treatment
    T = rng.binomial(1, 0.5, n)

    # Baseline function
    def a_func(X):
        return 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2]

    # Small constant treatment effect
    def b_func(X):
        return np.ones(len(X)) * ate

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
        true_ate=ate,
        true_ite=true_b,
        true_baseline=true_a,
        scenario=f"near_zero_ate{ate}",
    )


@dataclass
class PlaceboTestResult:
    """Results from a placebo test.

    Attributes
    ----------
    ate_estimate : float
        Estimated ATE.
    ate_se : float
        Standard error of ATE.
    pvalue : float
        p-value for H0: ATE = 0.
    ci_lower : float
        Lower bound of 95% CI.
    ci_upper : float
        Upper bound of 95% CI.
    ci_contains_zero : bool
        Whether 95% CI contains 0.
    significant : bool
        Whether p-value < 0.05 (should be False for placebo).
    """

    ate_estimate: float
    ate_se: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    ci_contains_zero: bool
    significant: bool

    @property
    def is_valid_placebo(self) -> bool:
        """Check if placebo test passed (not significant, CI contains 0)."""
        return self.ci_contains_zero and not self.significant


def run_placebo_test(result) -> PlaceboTestResult:
    """Extract placebo test results from HTEResults.

    Parameters
    ----------
    result : HTEResults
        Fitted model results.

    Returns
    -------
    PlaceboTestResult
        Placebo test summary.
    """
    from scipy import stats

    ate = result.ate
    se = result.ate_se

    # Two-sided test for H0: ATE = 0
    z_stat = ate / se
    pvalue = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # 95% CI
    ci_lower, ci_upper = result.ate_confint(alpha=0.05)
    ci_contains_zero = ci_lower <= 0 <= ci_upper

    return PlaceboTestResult(
        ate_estimate=ate,
        ate_se=se,
        pvalue=pvalue,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_contains_zero=ci_contains_zero,
        significant=pvalue < 0.05,
    )


def run_placebo_monte_carlo(
    estimator_factory,
    n_simulations: int = 100,
    scenario_kwargs: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Run Monte Carlo placebo tests.

    Parameters
    ----------
    estimator_factory : callable
        Function that returns a DeepHTE estimator instance.
    n_simulations : int, default=100
        Number of simulation replications.
    scenario_kwargs : dict, optional
        Keyword arguments for make_placebo_scenario.
    verbose : bool, default=True
        Whether to print progress.

    Returns
    -------
    dict
        Summary statistics including:
        - rejection_rate: Proportion of false rejections (should be ~0.05)
        - coverage_rate: Proportion of CIs containing 0 (should be ~0.95)
        - mean_ate: Mean estimated ATE (should be ~0)
        - mean_se: Mean standard error
        - results: List of PlaceboTestResult objects
    """
    if scenario_kwargs is None:
        scenario_kwargs = {}

    results = []
    rejections = 0
    covered = 0
    ate_estimates = []

    for i in range(n_simulations):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Placebo simulation {i + 1}/{n_simulations}")

        # Generate placebo data
        data = make_placebo_scenario(seed=i, **scenario_kwargs)

        # Fit model
        model = estimator_factory()
        fit_result = model.fit(data.data)

        # Extract placebo test results
        placebo_result = run_placebo_test(fit_result)
        results.append(placebo_result)

        if placebo_result.significant:
            rejections += 1
        if placebo_result.ci_contains_zero:
            covered += 1
        ate_estimates.append(placebo_result.ate_estimate)

    rejection_rate = rejections / n_simulations
    coverage_rate = covered / n_simulations
    mean_ate = np.mean(ate_estimates)
    mean_se = np.mean([r.ate_se for r in results])

    if verbose:
        print(f"\nPlacebo Test Summary ({n_simulations} simulations)")
        print(f"  False rejection rate: {rejection_rate:.1%} (target: ~5%)")
        print(f"  Coverage rate: {coverage_rate:.1%} (target: ~95%)")
        print(f"  Mean ATE estimate: {mean_ate:.4f} (target: ~0)")
        print(f"  Mean SE: {mean_se:.4f}")

    return {
        "rejection_rate": rejection_rate,
        "coverage_rate": coverage_rate,
        "mean_ate": mean_ate,
        "mean_se": mean_se,
        "results": results,
    }
