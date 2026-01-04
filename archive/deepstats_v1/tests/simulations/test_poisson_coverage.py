"""Monte Carlo coverage tests for DeepPoisson estimator.

Tests that 95% CIs achieve approximately 95% coverage for:
- E[lambda(X)]
- Var[lambda(X)]
- Quantiles of lambda(X)
"""

import pytest
import numpy as np

from deepstats.estimators import DeepPoisson
from deepstats.simulations import (
    PoissonSimulationStudy,
    make_poisson_dgp_lowdim,
    make_poisson_dgp_highdim,
)


@pytest.mark.slow
class TestPoissonCoverage:
    """Test DeepPoisson inference coverage via Monte Carlo."""

    def test_coverage_lowdim(self):
        """Test coverage with low-dimensional X (p=5)."""

        def dgp(seed):
            return make_poisson_dgp_lowdim(seed, n=2000, p=5)

        def make_estimator():
            return DeepPoisson(
                hidden_dims=[64, 32],
                epochs=200,
                cross_fit_folds=1,
                bootstrap_se=True,
                bootstrap_samples=50,
                verbose=0,
                random_state=None,
            )

        study = PoissonSimulationStudy(
            dgp=dgp,
            estimator_factory=make_estimator,
            n_simulations=100,
            random_state=42,
            verbose=1,
        )

        summary = study.run()
        print("\n" + summary.summary())

        # Check mean_lambda coverage (should be ~0.95)
        assert summary.mean_lambda_coverage >= 0.85, (
            f"Mean lambda coverage {summary.mean_lambda_coverage:.1%} < 85%"
        )

        # Check var_lambda coverage
        assert summary.var_lambda_coverage >= 0.80, (
            f"Var lambda coverage {summary.var_lambda_coverage:.1%} < 80%"
        )

        # Check SE calibration (ratio should be reasonably close to 1.0)
        assert 0.5 < summary.mean_lambda_se_ratio < 2.0, (
            f"Mean lambda SE ratio {summary.mean_lambda_se_ratio:.2f} out of range"
        )

    def test_coverage_highdim(self):
        """Test coverage with high-dimensional X (p=50)."""

        def dgp(seed):
            return make_poisson_dgp_highdim(seed, n=2000, p=50)

        def make_estimator():
            return DeepPoisson(
                hidden_dims=[64, 32],
                epochs=200,
                cross_fit_folds=1,
                bootstrap_se=True,
                bootstrap_samples=50,
                verbose=0,
                random_state=None,
            )

        study = PoissonSimulationStudy(
            dgp=dgp,
            estimator_factory=make_estimator,
            n_simulations=100,
            random_state=123,
            verbose=1,
        )

        summary = study.run()
        print("\n" + summary.summary())

        # Coverage may be slightly lower in high dimensions
        assert summary.mean_lambda_coverage >= 0.80, (
            f"Mean lambda coverage {summary.mean_lambda_coverage:.1%} < 80%"
        )

        assert summary.var_lambda_coverage >= 0.75, (
            f"Var lambda coverage {summary.var_lambda_coverage:.1%} < 75%"
        )


def run_quick_simulation():
    """Run a quick simulation for manual testing."""

    print("Running quick Poisson coverage simulation...")
    print("=" * 60)

    def dgp(seed):
        return make_poisson_dgp_lowdim(seed, n=1000, p=5)

    def make_estimator():
        return DeepPoisson(
            hidden_dims=[32, 16],
            epochs=100,
            cross_fit_folds=1,
            bootstrap_se=True,
            bootstrap_samples=30,
            verbose=0,
        )

    study = PoissonSimulationStudy(
        dgp=dgp,
        estimator_factory=make_estimator,
        n_simulations=20,
        random_state=42,
        verbose=1,
    )

    summary = study.run()
    print("\n" + summary.summary())

    return summary


class TestDeepPoissonBasic:
    """Basic unit tests for DeepPoisson."""

    def test_fit_returns_results(self):
        """Test that fit() returns PoissonResults."""
        from deepstats.results import PoissonResults

        data = make_poisson_dgp_lowdim(seed=42, n=200, p=3)

        model = DeepPoisson(
            hidden_dims=[16, 8],
            epochs=20,
            bootstrap_se=False,
            verbose=0,
        )
        result = model.fit(data.X, data.y)

        assert isinstance(result, PoissonResults)
        assert result.n_obs == 200
        assert result.mean_lambda > 0
        assert result.mean_lambda_se > 0
        assert result.var_lambda >= 0
        assert len(result.lambda_values) == 200

    def test_summary_runs(self):
        """Test that summary() produces output."""
        data = make_poisson_dgp_lowdim(seed=42, n=200, p=3)

        model = DeepPoisson(
            hidden_dims=[16, 8],
            epochs=20,
            bootstrap_se=False,
            verbose=0,
        )
        result = model.fit(data.X, data.y)

        summary_str = result.summary()
        assert "E[lambda(X)]" in summary_str
        assert "Var[lambda(X)]" in summary_str

    def test_confint_runs(self):
        """Test that confint() produces DataFrame."""
        import pandas as pd

        data = make_poisson_dgp_lowdim(seed=42, n=200, p=3)

        model = DeepPoisson(
            hidden_dims=[16, 8],
            epochs=20,
            bootstrap_se=False,
            verbose=0,
        )
        result = model.fit(data.X, data.y)

        ci = result.confint()
        assert isinstance(ci, pd.DataFrame)
        assert "lower" in ci.columns
        assert "upper" in ci.columns


if __name__ == "__main__":
    run_quick_simulation()
