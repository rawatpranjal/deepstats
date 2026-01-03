"""Tests for estimator classes."""

import numpy as np
import pytest
from sklearn.base import clone


class TestDeepOLS:
    """Test suite for DeepOLS estimator."""

    def test_sklearn_compatibility(self, linear_dgp):
        """DeepOLS should be sklearn-compatible (clonable)."""
        from deepstats.estimators import DeepOLS

        model = DeepOLS(epochs=10, random_state=42)

        # Should be clonable
        model_clone = clone(model)
        assert model_clone.epochs == model.epochs
        assert model_clone.random_state == model.random_state

    def test_fit_returns_results(self, linear_dgp):
        """fit() should return DeepResults object."""
        from deepstats.estimators import DeepOLS
        from deepstats.results import DeepResults

        X = linear_dgp["X"]
        y = linear_dgp["y"]

        model = DeepOLS(epochs=10, verbose=0)
        result = model.fit(X, y)

        assert isinstance(result, DeepResults)
        assert hasattr(result, "params")
        assert hasattr(result, "std_errors")
        assert hasattr(result, "summary")

    def test_predict_after_fit(self, linear_dgp):
        """predict() should work after fit()."""
        from deepstats.estimators import DeepOLS

        X = linear_dgp["X"]
        y = linear_dgp["y"]

        model = DeepOLS(epochs=10, verbose=0, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_predict_before_fit_raises(self, linear_dgp):
        """predict() before fit() should raise error."""
        from deepstats.estimators import DeepOLS
        from sklearn.exceptions import NotFittedError

        X = linear_dgp["X"]

        model = DeepOLS(epochs=10)
        with pytest.raises(NotFittedError):
            model.predict(X)

    def test_formula_interface(self, linear_dgp):
        """Formula interface should work with DataFrame."""
        import pandas as pd
        from deepstats.estimators import DeepOLS

        X = linear_dgp["X"]
        y = linear_dgp["y"]

        df = pd.DataFrame(X, columns=["x0", "x1", "x2"])
        df["y"] = y

        model = DeepOLS(formula="y ~ x0 + x1 + x2", epochs=10, verbose=0)
        result = model.fit(df)

        assert len(result.feature_names) == 3
        assert "x0" in result.feature_names


class TestDoubleMachineLearning:
    """Test suite for DoubleMachineLearning estimator."""

    def test_fit_returns_causal_results(self, causal_dgp):
        """fit() should return CausalResults object."""
        from deepstats.estimators import DoubleMachineLearning
        from deepstats.estimators.dml import CausalResults

        Y = causal_dgp["Y"]
        T = causal_dgp["T"]
        X = causal_dgp["X"]

        dml = DoubleMachineLearning(n_folds=2, epochs=10, verbose=0, random_state=42)
        result = dml.fit(Y=Y, T=T, X=X)

        assert isinstance(result, CausalResults)
        assert hasattr(result, "ate")
        assert hasattr(result, "ate_se")
        assert hasattr(result, "confint")

    def test_ate_sign_correct(self, causal_dgp):
        """ATE should have correct sign."""
        from deepstats.estimators import DoubleMachineLearning

        Y = causal_dgp["Y"]
        T = causal_dgp["T"]
        X = causal_dgp["X"]
        tau_true = causal_dgp["tau_true"]

        dml = DoubleMachineLearning(n_folds=2, epochs=30, verbose=0, random_state=42)
        result = dml.fit(Y=Y, T=T, X=X)

        # ATE should be positive (true = 2.0)
        assert result.ate > 0, f"ATE {result.ate:.4f} should be positive"

    def test_custom_learners(self, causal_dgp):
        """DML should accept custom sklearn learners."""
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from deepstats.estimators import DoubleMachineLearning

        Y = causal_dgp["Y"]
        T = causal_dgp["T"]
        X = causal_dgp["X"]

        dml = DoubleMachineLearning(
            outcome_learner=RandomForestRegressor(n_estimators=10, random_state=42),
            treatment_learner=RandomForestClassifier(n_estimators=10, random_state=42),
            n_folds=2,
            verbose=0,
            random_state=42,
        )
        result = dml.fit(Y=Y, T=T, X=X)

        assert not np.isnan(result.ate)
        assert not np.isnan(result.ate_se)

    def test_summary_output(self, causal_dgp):
        """summary() should produce valid output."""
        from deepstats.estimators import DoubleMachineLearning

        Y = causal_dgp["Y"]
        T = causal_dgp["T"]
        X = causal_dgp["X"]

        dml = DoubleMachineLearning(n_folds=2, epochs=10, verbose=0, random_state=42)
        result = dml.fit(Y=Y, T=T, X=X)

        summary = result.summary()
        assert "Double Machine Learning Results" in summary
        assert "ATE" in summary
        assert "Std. Error" in summary
