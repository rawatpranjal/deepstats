"""Tests for tough high-dimensional DGP patterns.

These tests verify that DeepHTE can recover treatment effects from
complex nonlinear patterns when given sufficient capacity and data.
"""

import numpy as np
import pytest

import deepstats as ds
from deepstats.simulations.tough_dgp import (
    make_tough_highdim_scenario,
    make_deep_interaction_scenario,
    make_threshold_scenario,
    make_multifreq_scenario,
    make_sparse_nonlinear_scenario,
    make_mixed_tough_scenario,
)


class TestToughDGPGeneration:
    """Test tough DGP generation functions."""

    @pytest.mark.parametrize("pattern", [
        "deep_interactions", "threshold", "multifreq",
        "sparse_nonlinear", "mixed"
    ])
    def test_pattern_generation(self, pattern):
        """All patterns should generate valid data."""
        data = make_tough_highdim_scenario(
            seed=42, n=500, p=50, pattern=pattern
        )

        assert len(data.data) == 500
        assert data.data.shape[1] == 52  # 50 covariates + Y + T
        assert isinstance(data.true_ate, float)
        assert len(data.true_ite) == 500
        assert not np.isnan(data.true_ate)

    def test_sparsity_respected(self):
        """Only sparse relevant variables should affect outcome."""
        data = make_tough_highdim_scenario(
            seed=42, n=1000, p=100, sparsity=5, pattern="sparse_nonlinear"
        )

        # Check that true effects exist
        assert np.std(data.true_ite) > 0
        assert np.std(data.true_baseline) > 0

    def test_convenience_functions(self):
        """Convenience functions should work."""
        scenarios = [
            make_deep_interaction_scenario(seed=42, n=100, p=20),
            make_threshold_scenario(seed=42, n=100, p=20),
            make_multifreq_scenario(seed=42, n=100, p=20),
            make_sparse_nonlinear_scenario(seed=42, n=100, p=50, sparsity=5),
            make_mixed_tough_scenario(seed=42, n=100, p=50),
        ]

        for data in scenarios:
            assert len(data.data) == 100
            assert isinstance(data.true_ate, float)


class TestATERecoveryToughPatterns:
    """Test ATE recovery from tough patterns."""

    def test_deep_interactions_recovery(self):
        """DeepHTE should estimate ATE from deep interactions."""
        data = make_deep_interaction_scenario(seed=42, n=2000, p=20)

        # Build formula from column names
        covs = " + ".join([f"X{i+1}" for i in range(20)])
        formula = f"Y ~ a({covs}) + b({covs}) * T"

        model = ds.DeepHTE(
            formula=formula,
            backbone="mlp",
            hidden_dims=[64, 32, 16],  # More capacity for interactions
            epochs=200,
            verbose=0,
            random_state=42,
        )
        result = model.fit(data.data)

        # ATE should be within 3 SEs of true value (wider tolerance for tough patterns)
        error = abs(result.ate - data.true_ate)
        assert error < 3 * result.ate_se, (
            f"ATE estimate {result.ate:.4f} too far from true {data.true_ate:.4f}. "
            f"Error: {error:.4f}, 3*SE: {3*result.ate_se:.4f}"
        )

    def test_threshold_recovery(self):
        """DeepHTE should estimate ATE from threshold patterns."""
        data = make_threshold_scenario(seed=42, n=2000, p=20)

        covs = " + ".join([f"X{i+1}" for i in range(20)])
        formula = f"Y ~ a({covs}) + b({covs}) * T"

        model = ds.DeepHTE(
            formula=formula,
            backbone="mlp",
            hidden_dims=[64, 32],
            epochs=200,
            verbose=0,
            random_state=42,
        )
        result = model.fit(data.data)

        # Check sign is correct at minimum
        assert np.sign(result.ate) == np.sign(data.true_ate) or abs(result.ate) < 0.5

    def test_multifreq_recovery(self):
        """DeepHTE should estimate ATE from multi-frequency patterns."""
        data = make_multifreq_scenario(seed=42, n=2000, p=20)

        covs = " + ".join([f"X{i+1}" for i in range(20)])
        formula = f"Y ~ a({covs}) + b({covs}) * T"

        model = ds.DeepHTE(
            formula=formula,
            backbone="mlp",
            hidden_dims=[64, 32],
            epochs=200,
            verbose=0,
            random_state=42,
        )
        result = model.fit(data.data)

        # ATE estimate should be reasonable (within 50% of true)
        if abs(data.true_ate) > 0.1:
            relative_error = abs(result.ate - data.true_ate) / abs(data.true_ate)
            assert relative_error < 0.5, (
                f"Relative error {relative_error:.2%} too large"
            )

    @pytest.mark.slow
    def test_sparse_nonlinear_recovery(self):
        """DeepHTE should recover ATE from sparse nonlinear patterns."""
        data = make_sparse_nonlinear_scenario(
            seed=42, n=3000, p=50, sparsity=5
        )

        covs = " + ".join([f"X{i+1}" for i in range(50)])
        formula = f"Y ~ a({covs}) + b({covs}) * T"

        # Use larger network for high-dim sparse problem
        model = ds.DeepHTE(
            formula=formula,
            backbone="mlp",
            hidden_dims=[128, 64, 32],
            epochs=300,
            verbose=0,
            random_state=42,
        )
        result = model.fit(data.data)

        # ATE should be within 4 SEs (relaxed for very tough pattern)
        error = abs(result.ate - data.true_ate)
        assert error < 4 * result.ate_se, (
            f"ATE estimate {result.ate:.4f} too far from true {data.true_ate:.4f}"
        )

    @pytest.mark.slow
    def test_mixed_tough_recovery(self):
        """DeepHTE should attempt ATE recovery from mixed patterns."""
        data = make_mixed_tough_scenario(seed=42, n=3000, p=50)

        covs = " + ".join([f"X{i+1}" for i in range(50)])
        formula = f"Y ~ a({covs}) + b({covs}) * T"

        model = ds.DeepHTE(
            formula=formula,
            backbone="transformer",  # More capacity
            hidden_dims=[64, 32],
            epochs=300,
            verbose=0,
            random_state=42,
        )
        result = model.fit(data.data)

        # For the hardest pattern, just verify we get reasonable output
        assert not np.isnan(result.ate)
        assert result.ate_se > 0


class TestITERecoveryToughPatterns:
    """Test ITE recovery correlation for tough patterns."""

    def test_ite_correlation_deep_interactions(self):
        """ITE estimates should correlate with true ITEs."""
        data = make_deep_interaction_scenario(seed=42, n=2000, p=20)

        covs = " + ".join([f"X{i+1}" for i in range(20)])
        formula = f"Y ~ a({covs}) + b({covs}) * T"

        model = ds.DeepHTE(
            formula=formula,
            backbone="mlp",
            hidden_dims=[64, 32, 16],
            epochs=200,
            verbose=0,
            random_state=42,
        )
        result = model.fit(data.data)

        # Check correlation (should be positive, even if not perfect)
        correlation = np.corrcoef(result.ite, data.true_ite)[0, 1]
        assert correlation > 0.3, f"ITE correlation {correlation:.3f} too low"

    def test_ite_correlation_threshold(self):
        """ITE estimates should capture threshold heterogeneity."""
        data = make_threshold_scenario(seed=42, n=2000, p=20)

        covs = " + ".join([f"X{i+1}" for i in range(20)])
        formula = f"Y ~ a({covs}) + b({covs}) * T"

        model = ds.DeepHTE(
            formula=formula,
            backbone="mlp",
            hidden_dims=[64, 32],
            epochs=200,
            verbose=0,
            random_state=42,
        )
        result = model.fit(data.data)

        # ITE variance should reflect heterogeneity
        if np.std(data.true_ite) > 0.1:
            ite_std_ratio = np.std(result.ite) / np.std(data.true_ite)
            # Estimated ITE should have some spread (not all identical)
            assert ite_std_ratio > 0.2, (
                f"ITE std ratio {ite_std_ratio:.3f} too low - "
                "model may not be capturing heterogeneity"
            )


class TestToughPatternHeterogeneity:
    """Test that tough patterns produce meaningful heterogeneity."""

    @pytest.mark.parametrize("pattern", [
        "deep_interactions", "threshold", "multifreq",
        "sparse_nonlinear", "mixed"
    ])
    def test_patterns_have_heterogeneity(self, pattern):
        """All patterns should produce heterogeneous effects."""
        data = make_tough_highdim_scenario(
            seed=42, n=1000, p=50, pattern=pattern
        )

        # ITEs should vary across observations
        ite_std = np.std(data.true_ite)
        ite_range = data.true_ite.max() - data.true_ite.min()

        assert ite_std > 0.1, f"Pattern {pattern} ITE std too low: {ite_std:.4f}"
        assert ite_range > 0.5, f"Pattern {pattern} ITE range too small: {ite_range:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
