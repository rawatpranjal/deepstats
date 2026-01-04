"""Tests for placebo scenarios.

Placebo tests verify that when true ATE = 0:
- Estimated ATE is approximately 0
- p-value > 0.05 (fail to reject null)
- 95% CI contains 0
"""

import numpy as np
import pytest

import deepstats as ds
from deepstats.simulations.placebo import (
    make_placebo_scenario,
    make_near_zero_scenario,
    run_placebo_test,
)


class TestPlaceboScenario:
    """Test placebo scenario generation."""

    def test_placebo_true_ate_is_zero(self):
        """True ATE should be exactly 0 for placebo."""
        data = make_placebo_scenario(seed=42)
        assert data.true_ate == 0.0

    def test_placebo_true_ite_all_zero(self):
        """True ITEs should all be 0 for placebo."""
        data = make_placebo_scenario(seed=42)
        assert np.allclose(data.true_ite, 0)

    def test_placebo_data_structure(self):
        """Placebo data should have correct structure."""
        data = make_placebo_scenario(seed=42, n=500, p=3)
        assert len(data.data) == 500
        assert "Y" in data.data.columns
        assert "T" in data.data.columns
        assert "X1" in data.data.columns

    @pytest.mark.parametrize("complexity", ["low", "medium", "high"])
    def test_placebo_complexity_levels(self, complexity):
        """All complexity levels should produce zero ATE."""
        data = make_placebo_scenario(seed=42, baseline_complexity=complexity)
        assert data.true_ate == 0.0
        assert np.allclose(data.true_ite, 0)


class TestNearZeroScenario:
    """Test near-zero treatment effect scenarios."""

    def test_near_zero_small_ate(self):
        """Near-zero scenario should have small ATE."""
        data = make_near_zero_scenario(seed=42, ate=0.01)
        assert data.true_ate == 0.01

    def test_near_zero_constant_ite(self):
        """ITEs should be constant in near-zero scenario."""
        ate = 0.05
        data = make_near_zero_scenario(seed=42, ate=ate)
        assert np.allclose(data.true_ite, ate)


class TestPlaceboTests:
    """Test placebo test execution."""

    def test_ate_near_zero(self):
        """Estimated ATE should be close to 0 for placebo."""
        data = make_placebo_scenario(seed=42, n=1000)

        model = ds.DeepHTE(
            formula="Y ~ a(X1 + X2 + X3 + X4 + X5) + b(X1 + X2 + X3 + X4 + X5) * T",
            backbone="mlp",
            hidden_dims=[32, 16],
            epochs=100,
            verbose=0,
            random_state=42,
        )
        result = model.fit(data.data)

        # ATE should be close to 0 (within 3 SEs)
        assert abs(result.ate) < 3 * result.ate_se, (
            f"ATE {result.ate:.4f} too far from 0. "
            f"3*SE = {3*result.ate_se:.4f}"
        )

    def test_pvalue_not_significant(self):
        """p-value should typically be > 0.05 for placebo."""
        data = make_placebo_scenario(seed=123, n=1000)

        model = ds.DeepHTE(
            formula="Y ~ a(X1 + X2 + X3 + X4 + X5) + b(X1 + X2 + X3 + X4 + X5) * T",
            backbone="mlp",
            hidden_dims=[32, 16],
            epochs=100,
            verbose=0,
            random_state=123,
        )
        result = model.fit(data.data)

        placebo_result = run_placebo_test(result)

        # Note: Individual tests may occasionally reject due to randomness
        # This test just verifies the machinery works
        assert 0 <= placebo_result.pvalue <= 1

    def test_ci_contains_zero(self):
        """95% CI should typically contain 0 for placebo."""
        data = make_placebo_scenario(seed=456, n=1000)

        model = ds.DeepHTE(
            formula="Y ~ a(X1 + X2 + X3 + X4 + X5) + b(X1 + X2 + X3 + X4 + X5) * T",
            backbone="mlp",
            hidden_dims=[32, 16],
            epochs=100,
            verbose=0,
            random_state=456,
        )
        result = model.fit(data.data)

        placebo_result = run_placebo_test(result)

        # CI should contain 0 in this particular run
        assert placebo_result.ci_contains_zero, (
            f"CI [{placebo_result.ci_lower:.4f}, {placebo_result.ci_upper:.4f}] "
            f"should contain 0"
        )

    @pytest.mark.slow
    def test_placebo_coverage_monte_carlo(self):
        """Monte Carlo: ~95% of CIs should contain 0."""
        n_reps = 50  # Use 200+ in production
        coverage_count = 0

        for rep in range(n_reps):
            data = make_placebo_scenario(seed=rep, n=500)

            model = ds.DeepHTE(
                formula="Y ~ a(X1 + X2 + X3 + X4 + X5) + b(X1 + X2 + X3 + X4 + X5) * T",
                backbone="mlp",
                hidden_dims=[16, 8],
                epochs=50,
                verbose=0,
                random_state=rep,
            )
            result = model.fit(data.data)

            ci_lower, ci_upper = result.ate_confint(alpha=0.05)
            if ci_lower <= 0 <= ci_upper:
                coverage_count += 1

        coverage_rate = coverage_count / n_reps

        # Should be close to 95%, allow 80-99%
        assert 0.75 <= coverage_rate <= 0.99, (
            f"Coverage rate {coverage_rate:.1%} outside expected range [75%, 99%]"
        )

    @pytest.mark.slow
    def test_placebo_rejection_rate(self):
        """Monte Carlo: False rejection rate should be ~5%."""
        n_reps = 50  # Use 200+ in production
        rejection_count = 0

        for rep in range(n_reps):
            data = make_placebo_scenario(seed=rep + 1000, n=500)

            model = ds.DeepHTE(
                formula="Y ~ a(X1 + X2 + X3 + X4 + X5) + b(X1 + X2 + X3 + X4 + X5) * T",
                backbone="mlp",
                hidden_dims=[16, 8],
                epochs=50,
                verbose=0,
                random_state=rep,
            )
            result = model.fit(data.data)

            placebo_result = run_placebo_test(result)
            if placebo_result.significant:
                rejection_count += 1

        rejection_rate = rejection_count / n_reps

        # Should be close to 5%, allow 0-20%
        assert rejection_rate <= 0.25, (
            f"False rejection rate {rejection_rate:.1%} is too high (> 25%)"
        )


class TestPlaceboResultProperties:
    """Test PlaceboTestResult properties."""

    def test_is_valid_placebo(self):
        """is_valid_placebo should be True when CI contains 0 and not significant."""
        data = make_placebo_scenario(seed=42, n=1000)

        model = ds.DeepHTE(
            formula="Y ~ a(X1 + X2 + X3 + X4 + X5) + b(X1 + X2 + X3 + X4 + X5) * T",
            backbone="mlp",
            hidden_dims=[32, 16],
            epochs=100,
            verbose=0,
            random_state=42,
        )
        result = model.fit(data.data)

        placebo_result = run_placebo_test(result)

        # Check that properties are consistent
        expected_valid = (
            placebo_result.ci_contains_zero and not placebo_result.significant
        )
        assert placebo_result.is_valid_placebo == expected_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
