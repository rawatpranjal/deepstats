"""Tests for HTE simulation studies."""

import numpy as np
import pytest

import deepstats as ds
from deepstats.simulations import (
    SimulationStudy,
    SimulationResult,
    SimulationSummary,
    compute_ate_metrics,
    compute_ite_metrics,
    compute_quantile_metrics,
    diagnose_fitting,
    FittingDiagnosis,
    make_overfit_scenario,
    make_underfit_scenario,
    make_balanced_scenario,
)


class TestSimulationMetrics:
    """Test simulation metrics computation."""

    def test_compute_ate_metrics(self):
        """Test ATE metrics computation."""
        true_ate = 2.0
        estimated_ates = [1.9, 2.1, 2.0, 1.8, 2.2]
        estimated_ses = [0.1, 0.1, 0.1, 0.1, 0.1]

        metrics = compute_ate_metrics(true_ate, estimated_ates, estimated_ses)

        assert abs(metrics.bias) < 0.1
        assert metrics.rmse > 0
        assert 0 <= metrics.coverage <= 1
        assert metrics.mean_se > 0
        assert metrics.empirical_se > 0

    def test_compute_ate_metrics_extended(self):
        """Test extended ATE metrics computation."""
        true_ate = 2.0
        estimated_ates = [1.9, 2.1, 2.0, 1.8, 2.2]
        estimated_ses = [0.1, 0.1, 0.1, 0.1, 0.1]

        metrics = compute_ate_metrics(true_ate, estimated_ates, estimated_ses)

        # Test new metrics
        assert not np.isnan(metrics.relative_bias)
        assert abs(metrics.relative_bias) < 0.1  # Small relative bias
        assert metrics.mae > 0
        assert metrics.ci_width > 0
        assert metrics.ci_width == pytest.approx(2 * 1.96 * 0.1, rel=0.01)  # 2 * z * SE
        # Power should be computed since true_ate=2.0 is non-zero
        assert not np.isnan(metrics.power)
        assert 0 <= metrics.power <= 1
        assert np.isnan(metrics.type1_error)  # Should be nan for non-zero true ATE

    def test_compute_ate_metrics_null_effect(self):
        """Test ATE metrics with null effect (type I error scenario)."""
        true_ate = 0.0
        # Small estimates around 0
        estimated_ates = [0.05, -0.03, 0.02, -0.01, 0.04]
        estimated_ses = [0.1, 0.1, 0.1, 0.1, 0.1]

        metrics = compute_ate_metrics(true_ate, estimated_ates, estimated_ses)

        # Relative bias should be nan when true_ate=0
        assert np.isnan(metrics.relative_bias)
        # Type I error should be computed
        assert not np.isnan(metrics.type1_error)
        assert 0 <= metrics.type1_error <= 1
        # Power should be nan when true_ate ≈ 0
        assert np.isnan(metrics.power)

    def test_compute_ite_metrics(self):
        """Test ITE metrics computation."""
        true_ites = [np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 3.5])]
        estimated_ites = [np.array([1.1, 2.1, 2.9]), np.array([1.4, 2.6, 3.4])]

        metrics = compute_ite_metrics(true_ites, estimated_ites)

        assert metrics.rmse > 0
        assert metrics.mae > 0
        assert -1 <= metrics.correlation <= 1
        assert -1 <= metrics.rank_correlation <= 1

    def test_compute_ite_metrics_extended(self):
        """Test extended ITE metrics computation."""
        np.random.seed(42)
        true_ites = [np.random.randn(100) + 2 for _ in range(5)]
        estimated_ites = [t + 0.1 * np.random.randn(100) for t in true_ites]

        metrics = compute_ite_metrics(true_ites, estimated_ites)

        # Test new metrics exist and have reasonable values
        assert not np.isnan(metrics.bias)
        assert abs(metrics.bias) < 0.5  # Bias should be small

        assert not np.isnan(metrics.relative_rmse)
        assert metrics.relative_rmse > 0

        assert not np.isnan(metrics.relative_bias)

        assert not np.isnan(metrics.empirical_se)
        assert metrics.empirical_se > 0

        assert not np.isnan(metrics.ks_statistic)
        assert 0 <= metrics.ks_statistic <= 1

        assert not np.isnan(metrics.ks_pvalue)
        assert 0 <= metrics.ks_pvalue <= 1

        assert not np.isnan(metrics.wasserstein_distance)
        assert metrics.wasserstein_distance >= 0

        # Without SEs, coverage and calibration should be nan
        assert np.isnan(metrics.coverage)
        assert np.isnan(metrics.mean_se)
        assert np.isnan(metrics.calibration_ratio)

    def test_compute_ite_metrics_with_ses(self):
        """Test ITE metrics with standard errors for coverage."""
        np.random.seed(42)
        n = 100
        true_ites = [np.random.randn(n) + 2 for _ in range(5)]
        estimated_ites = [t + 0.1 * np.random.randn(n) for t in true_ites]
        # Create SEs that approximately match the actual error scale
        estimated_ses = [np.ones(n) * 0.15 for _ in range(5)]

        metrics = compute_ite_metrics(
            true_ites, estimated_ites, estimated_ses=estimated_ses
        )

        # Coverage should now be computed
        assert not np.isnan(metrics.coverage)
        assert 0 <= metrics.coverage <= 1
        # With well-calibrated SEs, coverage should be near 95%
        assert metrics.coverage > 0.5  # At least reasonable

        # Mean SE should be computed
        assert not np.isnan(metrics.mean_se)
        assert metrics.mean_se > 0

        # Calibration ratio should be computed
        assert not np.isnan(metrics.calibration_ratio)
        assert metrics.calibration_ratio > 0

    def test_compute_quantile_metrics(self):
        """Test quantile metrics computation."""
        np.random.seed(42)
        true_ites = [np.random.randn(100) + 2 for _ in range(10)]
        estimated_ites = [t + 0.1 * np.random.randn(100) for t in true_ites]

        metrics = compute_quantile_metrics(true_ites, estimated_ites)

        assert 0.1 in metrics.biases
        assert 0.5 in metrics.biases
        assert 0.9 in metrics.biases

    def test_compute_quantile_metrics_extended(self):
        """Test extended quantile metrics computation."""
        np.random.seed(42)
        true_ites = [np.random.randn(100) + 2 for _ in range(10)]
        estimated_ites = [t + 0.1 * np.random.randn(100) for t in true_ites]

        metrics = compute_quantile_metrics(true_ites, estimated_ites)

        # Test new metrics exist for all quantiles
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            assert q in metrics.maes
            assert metrics.maes[q] > 0
            assert q in metrics.relative_biases
            assert q in metrics.empirical_ses
            assert metrics.empirical_ses[q] > 0

    def test_compute_quantile_metrics_with_bootstrap_ses(self):
        """Test quantile metrics with bootstrap SEs for SE calibration."""
        np.random.seed(42)
        true_ites = [np.random.randn(100) + 2 for _ in range(10)]
        estimated_ites = [t + 0.1 * np.random.randn(100) for t in true_ites]

        # Create mock bootstrap SEs
        bootstrap_ses = [
            {0.1: 0.05, 0.25: 0.04, 0.5: 0.03, 0.75: 0.04, 0.9: 0.05}
            for _ in range(10)
        ]

        metrics = compute_quantile_metrics(
            true_ites, estimated_ites, bootstrap_ses=bootstrap_ses
        )

        # Test SE calibration metrics
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            assert q in metrics.mean_ses
            assert not np.isnan(metrics.mean_ses[q])
            assert q in metrics.se_ratios
            assert q in metrics.ci_widths
            assert not np.isnan(metrics.ci_widths[q])


class TestFittingDiagnostics:
    """Test fitting diagnostics."""

    def test_diagnose_overfit(self):
        """Test overfitting detection."""
        # Overfitting: train loss low, val loss high and increasing
        train_loss = list(np.linspace(1.0, 0.1, 100))
        val_loss = list(np.linspace(1.0, 0.5, 50)) + list(np.linspace(0.5, 1.5, 50))

        diagnosis = diagnose_fitting(train_loss, val_loss)

        assert diagnosis in [FittingDiagnosis.OVERFIT, FittingDiagnosis.EARLY_STOPPING_NEEDED]

    def test_diagnose_underfit(self):
        """Test underfitting detection."""
        # Underfitting: both losses remain high
        train_loss = list(np.linspace(1.0, 0.9, 100))
        val_loss = list(np.linspace(1.0, 0.95, 100))

        diagnosis = diagnose_fitting(train_loss, val_loss)

        assert diagnosis in [FittingDiagnosis.UNDERFIT, FittingDiagnosis.INCONCLUSIVE]

    def test_diagnose_good_fit(self):
        """Test good fit detection."""
        # Good fit: both losses converge to similar low values
        train_loss = list(np.linspace(1.0, 0.2, 100))
        val_loss = list(np.linspace(1.0, 0.25, 100))

        diagnosis = diagnose_fitting(train_loss, val_loss)

        assert diagnosis == FittingDiagnosis.GOOD_FIT


class TestDGPs:
    """Test simulation DGPs."""

    def test_overfit_scenario(self):
        """Test overfit scenario generation."""
        data = make_overfit_scenario(seed=42, n=100, p=50)

        assert len(data.data) == 100
        assert data.data.shape[1] == 52  # 50 covariates + Y + T
        assert isinstance(data.true_ate, float)
        assert len(data.true_ite) == 100

    def test_underfit_scenario(self):
        """Test underfit scenario generation."""
        data = make_underfit_scenario(seed=42, n=5000, p=10)

        assert len(data.data) == 5000
        assert isinstance(data.true_ate, float)

    def test_balanced_scenario(self):
        """Test balanced scenario generation."""
        for complexity in ["low", "medium", "high"]:
            data = make_balanced_scenario(seed=42, complexity=complexity)

            assert len(data.data) == 2000
            assert isinstance(data.true_ate, float)


class TestSimulationStudy:
    """Test SimulationStudy class."""

    @pytest.fixture
    def simple_dgp(self):
        """Create a simple DGP for testing."""
        def dgp(seed):
            return make_balanced_scenario(seed=seed, n=200, p=3)
        return dgp

    @pytest.fixture
    def simple_estimator_factory(self):
        """Create a simple estimator factory."""
        def factory():
            return ds.DeepHTE(
                formula="Y ~ a(X1 + X2 + X3) + b(X1 + X2 + X3) * T",
                backbone="mlp",
                hidden_dims=[16, 8],
                epochs=20,
                verbose=0,
                random_state=42,
            )
        return factory

    def test_simulation_study_init(self, simple_dgp, simple_estimator_factory):
        """Test SimulationStudy initialization."""
        study = SimulationStudy(
            dgp=simple_dgp,
            estimator_factory=simple_estimator_factory,
            n_simulations=2,
            verbose=0,
        )

        assert study.n_simulations == 2
        assert study.validation_split == 0.2

    def test_simulation_study_run(self, simple_dgp, simple_estimator_factory):
        """Test running a small simulation study."""
        study = SimulationStudy(
            dgp=simple_dgp,
            estimator_factory=simple_estimator_factory,
            n_simulations=2,
            verbose=0,
        )

        summary = study.run()

        assert isinstance(summary, SimulationSummary)
        assert summary.n_simulations == 2
        assert len(summary.results) == 2
        assert summary.ate_metrics is not None
        assert summary.ite_metrics is not None

    def test_simulation_result_properties(self, simple_dgp, simple_estimator_factory):
        """Test SimulationResult properties."""
        study = SimulationStudy(
            dgp=simple_dgp,
            estimator_factory=simple_estimator_factory,
            n_simulations=1,
            verbose=0,
        )

        summary = study.run()
        result = summary.results[0]

        assert isinstance(result, SimulationResult)
        assert isinstance(result.ate_bias, float)
        # ate_covered may be numpy bool or Python bool
        assert result.ate_covered in (True, False)
        assert isinstance(result.ite_rmse, float)

    def test_simulation_summary_output(self, simple_dgp, simple_estimator_factory):
        """Test SimulationSummary output methods."""
        study = SimulationStudy(
            dgp=simple_dgp,
            estimator_factory=simple_estimator_factory,
            n_simulations=2,
            verbose=0,
        )

        summary = study.run()

        # Test summary string
        summary_str = summary.summary()
        assert "Simulation Study Summary" in summary_str
        assert "ATE Estimation" in summary_str
        assert "ITE Estimation" in summary_str

        # Test DataFrame conversion
        df = summary.to_dataframe()
        assert len(df) == 2
        assert "ate_estimate" in df.columns
        assert "ite_rmse" in df.columns


class TestDeepHTEValidation:
    """Test DeepHTE with validation split."""

    def test_deephte_validation_split(self):
        """Test DeepHTE with validation_split parameter."""
        data = ds.make_ab_test(n=500, p=3, seed=42)

        model = ds.DeepHTE(
            formula="Y ~ a(X1 + X2 + X3) + b(X1 + X2 + X3) * T",
            backbone="mlp",
            hidden_dims=[16, 8],
            epochs=30,
            verbose=0,
            random_state=42,
        )

        result = model.fit(data.data, validation_split=0.2)

        assert len(result.train_loss_history) == 30
        assert len(result.val_loss_history) == 30

    def test_deephte_validation_data(self):
        """Test DeepHTE with separate validation data."""
        train_data = ds.make_ab_test(n=400, p=3, seed=42)
        val_data = ds.make_ab_test(n=100, p=3, seed=43)

        model = ds.DeepHTE(
            formula="Y ~ a(X1 + X2 + X3) + b(X1 + X2 + X3) * T",
            backbone="mlp",
            hidden_dims=[16, 8],
            epochs=30,
            verbose=0,
            random_state=42,
        )

        result = model.fit(train_data.data, validation_data=val_data.data)

        assert len(result.train_loss_history) == 30
        assert len(result.val_loss_history) == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
