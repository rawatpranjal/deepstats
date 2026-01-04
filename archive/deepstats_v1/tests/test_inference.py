"""Tests for inference methods (bootstrap, influence, jackknife)."""

import numpy as np
import pytest


class TestBootstrap:
    """Tests for bootstrap SE methods."""

    def test_pairs_bootstrap_shape(self):
        """Bootstrap SEs should have correct shape."""
        from deepstats.inference.bootstrap import bootstrap_pairs

        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)
        y = X @ np.array([1.0, -0.5, 0.3]) + np.random.randn(n) * 0.5

        def fit_fn(X, y):
            # Simple OLS for testing
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            return beta

        result = bootstrap_pairs(X, y, fit_fn, n_bootstrap=50, random_state=42)

        assert result.se.shape == (p,)
        assert result.vcov.shape == (p, p)
        assert result.samples.shape == (50, p)

    def test_wild_bootstrap(self):
        """Wild bootstrap should work with both distributions."""
        from deepstats.inference.bootstrap import bootstrap_wild

        np.random.seed(42)
        n, p = 100, 2
        X = np.random.randn(n, p)
        beta = np.array([1.0, -0.5])
        y = X @ beta + np.random.randn(n) * 0.5
        fitted = X @ np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - fitted

        def fit_fn(X, y):
            return np.linalg.lstsq(X, y, rcond=None)[0]

        # Test Rademacher
        result_rad = bootstrap_wild(
            X, y, fitted, residuals, fit_fn,
            n_bootstrap=50, random_state=42, distribution="rademacher"
        )
        assert result_rad.se.shape == (p,)

        # Test Mammen
        result_mam = bootstrap_wild(
            X, y, fitted, residuals, fit_fn,
            n_bootstrap=50, random_state=42, distribution="mammen"
        )
        assert result_mam.se.shape == (p,)

    def test_bootstrap_se_positive(self):
        """All bootstrap SEs should be positive."""
        from deepstats.inference.bootstrap import bootstrap_pairs

        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)
        y = X @ np.array([1.0, -0.5, 0.3]) + np.random.randn(n) * 0.5

        def fit_fn(X, y):
            return np.linalg.lstsq(X, y, rcond=None)[0]

        result = bootstrap_pairs(X, y, fit_fn, n_bootstrap=50, random_state=42)
        assert (result.se > 0).all()


class TestInfluenceFunction:
    """Tests for influence function SE methods."""

    def test_influence_se_shape(self):
        """Influence SEs should have correct shape."""
        from deepstats.estimators import DeepOLS
        from deepstats.inference.influence import compute_influence_function_se

        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)
        y = X @ np.array([1.0, -0.5, 0.3]) + np.random.randn(n) * 0.5

        model = DeepOLS(epochs=20, verbose=0, random_state=42)
        result = model.fit(X, y)

        inf_result = compute_influence_function_se(
            X, y, result.network_,
            result.fitted_values, result.residuals,
            cross_fit=False,
        )

        assert inf_result.se.shape == (p,)
        assert inf_result.influence_scores.shape == (n, p)

    def test_influence_se_crossfit(self):
        """Cross-fitted influence SEs should work."""
        from deepstats.estimators import DeepOLS
        from deepstats.inference.influence import compute_influence_function_se

        np.random.seed(42)
        n, p = 100, 2
        X = np.random.randn(n, p)
        y = X @ np.array([0.5, -0.3]) + np.random.randn(n) * 0.3

        model = DeepOLS(epochs=20, verbose=0, random_state=42)
        result = model.fit(X, y)

        inf_result = compute_influence_function_se(
            X, y, result.network_,
            result.fitted_values, result.residuals,
            cross_fit=True, n_folds=3, random_state=42,
        )

        assert inf_result.se.shape == (p,)
        assert (inf_result.se > 0).all()


class TestJackknife:
    """Tests for jackknife SE methods."""

    def test_delete_d_jackknife(self):
        """Delete-d jackknife should work."""
        from deepstats.inference.jackknife import delete_d_jackknife_se

        np.random.seed(42)
        n, p = 50, 2
        X = np.random.randn(n, p)
        beta = np.array([1.0, -0.5])
        y = X @ beta + np.random.randn(n) * 0.3

        def fit_fn(X, y):
            return np.linalg.lstsq(X, y, rcond=None)[0]

        params = fit_fn(X, y)
        result = delete_d_jackknife_se(
            X, y, fit_fn, params, d=5, random_state=42
        )

        assert result.se.shape == (p,)
        assert (result.se > 0).all()

    def test_infinitesimal_jackknife(self):
        """Infinitesimal jackknife should work."""
        from deepstats.inference.jackknife import infinitesimal_jackknife_se

        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)
        beta = np.array([1.0, -0.5, 0.3])
        y = X @ beta + np.random.randn(n) * 0.5

        fitted = X @ np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - fitted

        se = infinitesimal_jackknife_se(X, residuals, X)

        assert se.shape == (p,)
        assert (se > 0).all()


class TestSEValidation:
    """Tests for SE validation utilities."""

    def test_validation_result_structure(self):
        """Validation result should have correct structure."""
        from deepstats.estimators import DeepOLS
        from deepstats.inference.validation import validate_standard_errors

        np.random.seed(42)
        n, p = 100, 2
        X = np.random.randn(n, p)
        y = X @ np.array([0.5, -0.3]) + np.random.randn(n) * 0.3

        model = DeepOLS(epochs=20, verbose=0, random_state=42)
        result = model.fit(X, y)

        validation = validate_standard_errors(
            result,
            methods=["influence"],  # Skip bootstrap for speed
            random_state=42,
        )

        assert validation.analytical_se is not None
        assert validation.influence_se is not None
        assert validation.agreement_status in ["excellent", "good", "acceptable", "poor"]

    def test_validation_summary(self):
        """Validation summary should be printable."""
        from deepstats.estimators import DeepOLS
        from deepstats.inference.validation import validate_standard_errors

        np.random.seed(42)
        n, p = 100, 2
        X = np.random.randn(n, p)
        y = X @ np.array([0.5, -0.3]) + np.random.randn(n) * 0.3

        model = DeepOLS(epochs=20, verbose=0, random_state=42)
        result = model.fit(X, y)

        validation = validate_standard_errors(
            result,
            methods=["influence"],
            random_state=42,
        )

        summary = validation.summary()
        assert isinstance(summary, str)
        assert "Standard Error Validation" in summary
