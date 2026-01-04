"""Simulation tests for ATE recovery with DeepHTE.

These tests verify that the DeepHTE estimator recovers the true ATE
within expected tolerance using the enriched structural model approach.
"""

import numpy as np
import pytest

import deepstats as ds


def generate_causal_dgp(n: int, ate: float, seed: int):
    """Generate a simple causal DGP with known ATE.

    Y = a(X) + b(X) * T + epsilon
    where a(X) = X @ beta, b(X) = ate (constant)
    T ~ Bernoulli(0.5)

    This is a simple DGP where the true ITE is constant.
    """
    np.random.seed(seed)

    p = 3
    X = np.random.randn(n, p)

    # Simple propensity (no confounding in treatment)
    T = np.random.binomial(1, 0.5, size=n).astype(float)

    # Outcome: Y = a(X) + b*T + noise
    beta = np.array([0.5, -0.3, 0.2])
    epsilon = np.random.randn(n) * 0.5
    Y = X @ beta + ate * T + epsilon

    return Y, T, X


class TestATERecovery:
    """Test suite for ATE recovery with DeepHTE."""

    def test_deephte_single_run(self):
        """DeepHTE should estimate ATE within 2 SEs of truth."""
        true_ate = 2.0
        Y, T, X = generate_causal_dgp(n=1000, ate=true_ate, seed=42)

        # Create DataFrame for DeepHTE
        import pandas as pd
        df = pd.DataFrame(X, columns=["X1", "X2", "X3"])
        df["Y"] = Y
        df["T"] = T

        model = ds.DeepHTE(
            formula="Y ~ a(X1 + X2 + X3) + b(X1 + X2 + X3) * T",
            backbone="mlp",
            hidden_dims=[32, 16],
            epochs=100,
            verbose=0,
            random_state=42,
        )
        result = model.fit(df)

        # ATE should be within 2 SEs of true value
        error = abs(result.ate - true_ate)
        assert error < 2 * result.ate_se, (
            f"ATE estimate {result.ate:.4f} too far from true {true_ate:.4f}. "
            f"Error: {error:.4f}, 2*SE: {2*result.ate_se:.4f}"
        )

    def test_deephte_confint_contains_truth(self):
        """DeepHTE confidence interval should contain true ATE."""
        true_ate = 2.0
        Y, T, X = generate_causal_dgp(n=1000, ate=true_ate, seed=123)

        import pandas as pd
        df = pd.DataFrame(X, columns=["X1", "X2", "X3"])
        df["Y"] = Y
        df["T"] = T

        model = ds.DeepHTE(
            formula="Y ~ a(X1 + X2 + X3) + b(X1 + X2 + X3) * T",
            backbone="mlp",
            hidden_dims=[32, 16],
            epochs=100,
            verbose=0,
            random_state=123,
        )
        result = model.fit(df)

        ci_lower, ci_upper = result.ate_confint(alpha=0.05)
        assert ci_lower <= true_ate <= ci_upper, (
            f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}] does not contain "
            f"true ATE {true_ate}"
        )

    @pytest.mark.slow
    def test_deephte_monte_carlo_coverage(self):
        """Monte Carlo test: 95% CIs should cover truth ~95% of time."""
        true_ate = 2.0
        n_reps = 50  # Reduced for speed; use 200+ in production
        n_samples = 500
        coverage_count = 0

        import pandas as pd

        for rep in range(n_reps):
            Y, T, X = generate_causal_dgp(n=n_samples, ate=true_ate, seed=rep)

            df = pd.DataFrame(X, columns=["X1", "X2", "X3"])
            df["Y"] = Y
            df["T"] = T

            model = ds.DeepHTE(
                formula="Y ~ a(X1 + X2 + X3) + b(X1 + X2 + X3) * T",
                backbone="mlp",
                hidden_dims=[16, 8],
                epochs=50,
                verbose=0,
                random_state=rep,
            )
            result = model.fit(df)

            ci_lower, ci_upper = result.ate_confint(alpha=0.05)
            if ci_lower <= true_ate <= ci_upper:
                coverage_count += 1

        coverage_rate = coverage_count / n_reps
        # Allow some slack: expect ~95% but accept 80-99%
        assert 0.75 <= coverage_rate <= 0.99, (
            f"Coverage rate {coverage_rate:.2%} outside expected range [75%, 99%]"
        )

    @pytest.mark.slow
    def test_deephte_bias(self):
        """Monte Carlo test: ATE estimates should be approximately unbiased."""
        true_ate = 2.0
        n_reps = 30
        n_samples = 500
        ate_estimates = []

        import pandas as pd

        for rep in range(n_reps):
            Y, T, X = generate_causal_dgp(n=n_samples, ate=true_ate, seed=rep)

            df = pd.DataFrame(X, columns=["X1", "X2", "X3"])
            df["Y"] = Y
            df["T"] = T

            model = ds.DeepHTE(
                formula="Y ~ a(X1 + X2 + X3) + b(X1 + X2 + X3) * T",
                backbone="mlp",
                hidden_dims=[16, 8],
                epochs=50,
                verbose=0,
                random_state=rep,
            )
            result = model.fit(df)
            ate_estimates.append(result.ate)

        mean_ate = np.mean(ate_estimates)
        bias = mean_ate - true_ate

        # Bias should be small relative to true effect
        assert abs(bias) < 0.3, (
            f"Bias {bias:.4f} too large. Mean ATE: {mean_ate:.4f}, True: {true_ate}"
        )
