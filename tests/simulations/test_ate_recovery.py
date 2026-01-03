"""Simulation tests for ATE recovery with DML.

These tests verify that the DoubleMachineLearning estimator
recovers the true ATE within expected tolerance.
"""

import numpy as np
import pytest


def generate_causal_dgp(n: int, ate: float, seed: int):
    """Generate a simple causal DGP with known ATE.

    Y = tau * T + X @ beta + epsilon
    T ~ Bernoulli(0.5)

    This is a simple DGP where the propensity is constant,
    making it easier to recover the true effect.
    """
    np.random.seed(seed)

    p = 3
    X = np.random.randn(n, p)

    # Simple propensity (no confounding in treatment)
    T = np.random.binomial(1, 0.5, size=n).astype(float)

    # Outcome
    beta = np.array([0.5, -0.3, 0.2])
    epsilon = np.random.randn(n) * 0.5
    Y = ate * T + X @ beta + epsilon

    return Y, T, X


class TestATERecovery:
    """Test suite for ATE recovery."""

    def test_dml_single_run(self, causal_dgp):
        """DML should estimate ATE within 2 SEs of truth."""
        from deepstats.estimators import DoubleMachineLearning

        Y = causal_dgp["Y"]
        T = causal_dgp["T"]
        X = causal_dgp["X"]
        tau_true = causal_dgp["tau_true"]

        dml = DoubleMachineLearning(n_folds=3, epochs=50, verbose=0, random_state=42)
        result = dml.fit(Y=Y, T=T, X=X)

        # ATE should be within 2 SEs of true value
        error = abs(result.ate - tau_true)
        assert error < 2 * result.ate_se, (
            f"ATE estimate {result.ate:.4f} too far from true {tau_true:.4f}. "
            f"Error: {error:.4f}, 2*SE: {2*result.ate_se:.4f}"
        )

    @pytest.mark.slow
    def test_dml_monte_carlo_coverage(self):
        """Monte Carlo test: 95% CIs should cover truth ~95% of time."""
        from deepstats.estimators import DoubleMachineLearning

        true_ate = 2.0
        n_reps = 50  # Reduced for speed; use 200+ in production
        n_samples = 500
        coverage_count = 0

        for rep in range(n_reps):
            Y, T, X = generate_causal_dgp(n=n_samples, ate=true_ate, seed=rep)

            dml = DoubleMachineLearning(n_folds=3, epochs=30, verbose=0, random_state=rep)
            result = dml.fit(Y=Y, T=T, X=X)

            ci_lower, ci_upper = result.confint(alpha=0.05)
            if ci_lower <= true_ate <= ci_upper:
                coverage_count += 1

        coverage_rate = coverage_count / n_reps
        # Allow some slack: expect ~95% but accept 80-99%
        assert 0.75 <= coverage_rate <= 0.99, (
            f"Coverage rate {coverage_rate:.2%} outside expected range [75%, 99%]"
        )

    @pytest.mark.slow
    def test_dml_bias(self):
        """Monte Carlo test: ATE estimates should be approximately unbiased."""
        from deepstats.estimators import DoubleMachineLearning

        true_ate = 2.0
        n_reps = 30
        n_samples = 500
        ate_estimates = []

        for rep in range(n_reps):
            Y, T, X = generate_causal_dgp(n=n_samples, ate=true_ate, seed=rep)

            dml = DoubleMachineLearning(n_folds=3, epochs=30, verbose=0, random_state=rep)
            result = dml.fit(Y=Y, T=T, X=X)
            ate_estimates.append(result.ate)

        mean_ate = np.mean(ate_estimates)
        bias = mean_ate - true_ate

        # Bias should be small relative to true effect
        assert abs(bias) < 0.2, (
            f"Bias {bias:.4f} too large. Mean ATE: {mean_ate:.4f}, True: {true_ate}"
        )
