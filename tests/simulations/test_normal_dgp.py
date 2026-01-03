"""Simulation tests for Normal/OLS regression.

These tests verify that DeepOLS recovers true coefficients
(as marginal effects) within expected tolerance.
"""

import numpy as np
import pytest


class TestNormalDGP:
    """Test suite for DeepOLS with Normal DGP."""

    def test_linear_recovery(self, linear_dgp):
        """DeepOLS should recover linear coefficients as marginal effects."""
        from deepstats.estimators import DeepOLS

        X = linear_dgp["X"]
        y = linear_dgp["y"]
        beta_true = linear_dgp["beta_true"]

        model = DeepOLS(
            hidden_dims=[64, 32],
            epochs=150,
            lr=1e-3,
            robust_se="HC1",
            random_state=42,
            verbose=0,
        )
        result = model.fit(X, y)

        # Marginal effects should approximate true beta
        for i, (est, true) in enumerate(zip(result.params, beta_true)):
            error = abs(est - true)
            assert error < 0.15, (
                f"Coefficient {i}: estimated {est:.4f} vs true {true:.4f}, "
                f"error {error:.4f} > 0.15"
            )

    def test_r_squared_reasonable(self, linear_dgp):
        """R-squared should be high for linear DGP with moderate noise."""
        from deepstats.estimators import DeepOLS

        X = linear_dgp["X"]
        y = linear_dgp["y"]

        model = DeepOLS(epochs=100, random_state=42, verbose=0)
        result = model.fit(X, y)

        # With sigma=0.5 and reasonable signal, R^2 should be > 0.6
        assert result.r_squared > 0.6, f"R^2 {result.r_squared:.4f} too low"

    def test_sigma_estimation(self, linear_dgp):
        """Estimated sigma should be close to true sigma."""
        from deepstats.estimators import DeepOLS

        X = linear_dgp["X"]
        y = linear_dgp["y"]
        sigma_true = linear_dgp["sigma"]

        model = DeepOLS(epochs=100, random_state=42, verbose=0)
        result = model.fit(X, y)

        # Sigma estimate should be within 20% of true value
        error = abs(result.sigma_ - sigma_true) / sigma_true
        assert error < 0.2, (
            f"Sigma estimate {result.sigma_:.4f} vs true {sigma_true:.4f}, "
            f"relative error {error:.2%}"
        )

    def test_summary_output(self, linear_dgp):
        """summary() should produce valid output."""
        from deepstats.estimators import DeepOLS

        X = linear_dgp["X"]
        y = linear_dgp["y"]

        model = DeepOLS(epochs=50, random_state=42, verbose=0)
        result = model.fit(X, y)

        summary = result.summary()

        # Check summary contains expected sections
        assert "Deep Neural Network Regression Results" in summary
        assert "R-squared" in summary
        assert "coef" in summary
        assert "std err" in summary

    def test_confint_coverage(self, linear_dgp):
        """95% CI should contain true values most of the time."""
        from deepstats.estimators import DeepOLS

        X = linear_dgp["X"]
        y = linear_dgp["y"]
        beta_true = linear_dgp["beta_true"]

        model = DeepOLS(epochs=100, robust_se="HC1", random_state=42, verbose=0)
        result = model.fit(X, y)

        ci = result.confint(alpha=0.05)
        covered = 0
        for i, true_val in enumerate(beta_true):
            if ci.iloc[i]["lower"] <= true_val <= ci.iloc[i]["upper"]:
                covered += 1

        # At least 2 out of 3 coefficients should be covered
        assert covered >= 2, f"Only {covered}/3 coefficients covered by 95% CI"

    @pytest.mark.slow
    def test_monte_carlo_unbiased(self):
        """Monte Carlo: estimates should be approximately unbiased."""
        from deepstats.estimators import DeepOLS

        n_reps = 30
        n = 500
        beta_true = np.array([0.5, -0.3])

        estimates = []
        for rep in range(n_reps):
            np.random.seed(rep)
            X = np.random.randn(n, 2)
            y = X @ beta_true + np.random.randn(n) * 0.5

            model = DeepOLS(epochs=80, random_state=rep, verbose=0)
            result = model.fit(X, y)
            estimates.append(result.params)

        mean_estimates = np.mean(estimates, axis=0)
        bias = mean_estimates - beta_true

        for i, b in enumerate(bias):
            assert abs(b) < 0.1, f"Coefficient {i} bias {b:.4f} too large"
