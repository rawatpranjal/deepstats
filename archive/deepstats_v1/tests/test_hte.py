"""Tests for heterogeneous treatment effects estimator."""

import numpy as np
import pandas as pd
import pytest
import torch

import deepstats as ds
from deepstats.datasets.ab_test import make_ab_test, make_ab_test_binary
from deepstats.formula.parser import FormulaParser, ParsedFormula


class TestFormulaParser:
    """Test formula parsing functionality."""

    def test_basic_formula(self):
        """Test parsing a basic formula."""
        parser = FormulaParser()
        result = parser.parse("Y ~ a(X1 + X2) + b(X1 + X2) * T")

        assert result.outcome == "Y"
        assert result.treatment == "T"
        assert result.a_covariates == ["X1", "X2"]
        assert result.b_covariates == ["X1", "X2"]

    def test_formula_with_more_covariates(self):
        """Test formula with different covariates."""
        parser = FormulaParser()
        result = parser.parse("wage ~ a(edu + exp + tenure) + b(edu + exp) * training")

        assert result.outcome == "wage"
        assert result.treatment == "training"
        assert result.a_covariates == ["edu", "exp", "tenure"]
        assert result.b_covariates == ["edu", "exp"]

    def test_formula_with_different_covariates(self):
        """Test when a() and b() have different covariates."""
        parser = FormulaParser()
        result = parser.parse("Y ~ a(X1 + X2 + X3) + b(X1 + X2) * T")

        assert len(result.a_covariates) == 3
        assert len(result.b_covariates) == 2
        assert result.a_covariates == ["X1", "X2", "X3"]
        assert result.b_covariates == ["X1", "X2"]

    def test_invalid_formula_raises(self):
        """Test that invalid formulas raise ValueError."""
        parser = FormulaParser()

        with pytest.raises(ValueError):
            parser.parse("Y ~ X1 + X2")  # Missing a() and b()

        with pytest.raises(ValueError):
            parser.parse("Y ~ a(X1) + X2 * T")  # Missing b()

        with pytest.raises(ValueError):
            parser.parse("Y ~ a(X1) + b(X2)")  # Missing * T

    def test_formula_validation(self):
        """Test formula validation method."""
        assert FormulaParser.validate_formula("Y ~ a(X1 + X2) + b(X1 + X2) * T")
        assert not FormulaParser.validate_formula("Y ~ X1 + X2")

    def test_extract_data(self):
        """Test extracting data from DataFrame."""
        parser = FormulaParser()

        # Create sample data
        n = 100
        data = pd.DataFrame({
            "Y": np.random.randn(n),
            "X1": np.random.randn(n),
            "X2": np.random.randn(n),
            "T": np.random.binomial(1, 0.5, n),
        })

        result = parser.parse("Y ~ a(X1 + X2) + b(X1 + X2) * T", data)

        assert result.y is not None
        assert result.t is not None
        assert result.X_a is not None
        assert result.X_b is not None
        assert len(result.y) == n
        assert result.X_a.shape == (n, 2)

    def test_missing_variable_raises(self):
        """Test that missing variables raise KeyError."""
        parser = FormulaParser()
        data = pd.DataFrame({
            "Y": [1, 2, 3],
            "X1": [1, 2, 3],
            "T": [0, 1, 0],
        })

        result = parser.parse("Y ~ a(X1 + X2) + b(X1 + X2) * T")
        with pytest.raises(KeyError):
            result.extract_data(data)  # X2 is missing


class TestABTestData:
    """Test A/B test data generators."""

    def test_make_ab_test_basic(self):
        """Test basic A/B test data generation."""
        result = make_ab_test(n=1000, p=5, seed=42)

        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) == 1000
        assert "Y" in result.data.columns
        assert "T" in result.data.columns
        assert all(f"X{i}" in result.data.columns for i in range(1, 6))
        assert len(result.true_ite) == 1000
        assert isinstance(result.true_ate, float)

    def test_make_ab_test_heterogeneity_types(self):
        """Test different heterogeneity types."""
        for het in ["none", "linear", "nonlinear", "complex"]:
            result = make_ab_test(n=100, heterogeneity=het, seed=42)
            assert len(result.data) == 100
            assert isinstance(result.true_ate, float)

    def test_make_ab_test_binary(self):
        """Test binary outcome data generation."""
        result = make_ab_test_binary(n=1000, seed=42)

        assert len(result.data) == 1000
        assert result.data["Y"].isin([0.0, 1.0]).all()
        assert isinstance(result.true_ate, float)

    def test_treatment_balance(self):
        """Test that treatment is balanced."""
        result = make_ab_test(n=10000, treatment_prob=0.5, seed=42)
        t_mean = result.data["T"].mean()
        assert 0.48 < t_mean < 0.52  # Close to 0.5


class TestDeepHTE:
    """Test DeepHTE estimator."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return make_ab_test(n=500, p=3, heterogeneity="linear", seed=42)

    def test_deephte_init(self):
        """Test DeepHTE initialization."""
        model = ds.DeepHTE(
            formula="Y ~ a(X1 + X2) + b(X1 + X2) * T",
            family="normal",
            backbone="mlp",
        )
        assert model.formula == "Y ~ a(X1 + X2) + b(X1 + X2) * T"
        assert model.family == "normal"
        assert model.backbone == "mlp"

    def test_deephte_fit_mlp(self, sample_data):
        """Test fitting DeepHTE with MLP backbone."""
        model = ds.DeepHTE(
            formula="Y ~ a(X1 + X2 + X3) + b(X1 + X2 + X3) * T",
            family="normal",
            backbone="mlp",
            hidden_dims=[32, 16],
            epochs=50,
            verbose=0,
            random_state=42,
        )

        result = model.fit(sample_data.data)

        assert isinstance(result, ds.HTEResults)
        assert hasattr(result, "ate")
        assert hasattr(result, "ate_se")
        assert hasattr(result, "ite")
        assert hasattr(result, "quantiles")
        assert len(result.ite) == len(sample_data.data)

    def test_deephte_ate_direction(self, sample_data):
        """Test that ATE estimate is in right direction."""
        model = ds.DeepHTE(
            formula="Y ~ a(X1 + X2 + X3) + b(X1 + X2 + X3) * T",
            family="normal",
            backbone="mlp",
            hidden_dims=[32, 16],
            epochs=100,
            verbose=0,
            random_state=42,
        )

        result = model.fit(sample_data.data)

        # Check ATE is positive (true effect is positive in linear case)
        assert result.ate > 0

    def test_deephte_quantiles(self, sample_data):
        """Test that quantiles are computed."""
        model = ds.DeepHTE(
            formula="Y ~ a(X1 + X2 + X3) + b(X1 + X2 + X3) * T",
            family="normal",
            backbone="mlp",
            epochs=30,
            verbose=0,
            random_state=42,
        )

        result = model.fit(sample_data.data)

        assert 0.1 in result.quantiles
        assert 0.5 in result.quantiles
        assert 0.9 in result.quantiles
        # Quantiles should be ordered
        assert result.quantiles[0.1] <= result.quantiles[0.5] <= result.quantiles[0.9]

    def test_deephte_summary(self, sample_data):
        """Test summary output."""
        model = ds.DeepHTE(
            formula="Y ~ a(X1 + X2 + X3) + b(X1 + X2 + X3) * T",
            family="normal",
            backbone="mlp",
            epochs=30,
            verbose=0,
            random_state=42,
        )

        result = model.fit(sample_data.data)
        summary = result.summary()

        assert isinstance(summary, str)
        assert "Average Treatment Effect" in summary
        assert "Heterogeneity" in summary

    def test_deephte_binary_outcome(self):
        """Test DeepHTE with binary outcome."""
        data = make_ab_test_binary(n=500, p=3, seed=42)

        model = ds.DeepHTE(
            formula="Y ~ a(X1 + X2 + X3) + b(X1 + X2 + X3) * T",
            family="bernoulli",
            backbone="mlp",
            hidden_dims=[32, 16],
            epochs=50,
            verbose=0,
            random_state=42,
        )

        result = model.fit(data.data)

        assert isinstance(result, ds.HTEResults)
        assert result.family == "bernoulli"

    def test_deephte_predict(self, sample_data):
        """Test prediction on results."""
        model = ds.DeepHTE(
            formula="Y ~ a(X1 + X2 + X3) + b(X1 + X2 + X3) * T",
            family="normal",
            backbone="mlp",
            epochs=30,
            verbose=0,
            random_state=42,
        )

        result = model.fit(sample_data.data)

        # Test prediction
        X_new = sample_data.data[["X1", "X2", "X3"]].values[:10]
        T_new = np.ones(10)
        preds = result.predict(X_new, T_new)

        assert len(preds) == 10

    def test_deephte_cate(self, sample_data):
        """Test CATE extraction."""
        model = ds.DeepHTE(
            formula="Y ~ a(X1 + X2 + X3) + b(X1 + X2 + X3) * T",
            family="normal",
            backbone="mlp",
            epochs=30,
            verbose=0,
            random_state=42,
        )

        result = model.fit(sample_data.data)

        # Get CATE for new observations
        X_new = sample_data.data[["X1", "X2", "X3"]].values[:10]
        cate = result.cate(X_new)

        assert len(cate) == 10


class TestArchitectureRegistry:
    """Test network architecture registry."""

    def test_available_architectures(self):
        """Test listing available architectures."""
        available = ds.ArchitectureRegistry.available()
        assert "mlp" in available
        assert "transformer" in available
        assert "lstm" in available

    def test_create_mlp(self):
        """Test creating MLP backbone."""
        backbone = ds.ArchitectureRegistry.create("mlp", input_dim=10)
        assert backbone.input_dim == 10

    def test_create_transformer(self):
        """Test creating Transformer backbone."""
        backbone = ds.ArchitectureRegistry.create("transformer", input_dim=10)
        assert backbone.input_dim == 10

    def test_create_lstm(self):
        """Test creating LSTM backbone."""
        backbone = ds.ArchitectureRegistry.create("lstm", input_dim=10)
        assert backbone.input_dim == 10

    def test_unknown_architecture_raises(self):
        """Test that unknown architecture raises error."""
        with pytest.raises(ValueError):
            ds.ArchitectureRegistry.create("unknown", input_dim=10)


class TestParameterNetwork:
    """Test ParameterNetwork functionality."""

    def test_parameter_network(self):
        """Test ParameterNetwork forward pass."""
        backbone = ds.MLPBackbone(input_dim=10, hidden_dims=[32, 16])
        net = ds.ParameterNetwork(backbone, param_dim=2, param_names=["a", "b"])

        x = torch.randn(5, 10)
        params = net(x)

        assert params.shape == (5, 2)

    def test_parameter_network_with_names(self):
        """Test ParameterNetwork with parameter names."""
        backbone = ds.MLPBackbone(input_dim=10, hidden_dims=[32, 16])
        net = ds.ParameterNetwork(backbone, param_dim=2, param_names=["a", "b"])

        # Check parameter names are stored
        assert net.param_names == ["a", "b"]

        x = torch.randn(5, 10)
        params = net(x)

        # First column is a(X), second is b(X)
        assert params[:, 0].shape == (5,)
        assert params[:, 1].shape == (5,)


class TestHTEIntegration:
    """Integration tests for full HTE pipeline."""

    def test_continuous_recovery(self):
        """Test recovery of true ATE in continuous case."""
        # Generate data with known ATE
        np.random.seed(42)
        n = 2000
        X = np.random.randn(n, 3)
        T = np.random.binomial(1, 0.5, n)

        true_a = 0.5 * X[:, 0] - 0.3 * X[:, 1]
        true_b = np.ones(n) * 2.0  # Constant ATE = 2

        Y = true_a + true_b * T + np.random.randn(n) * 0.5

        data = pd.DataFrame({
            "Y": Y,
            "X1": X[:, 0],
            "X2": X[:, 1],
            "X3": X[:, 2],
            "T": T,
        })

        model = ds.DeepHTE(
            formula="Y ~ a(X1 + X2 + X3) + b(X1 + X2 + X3) * T",
            family="normal",
            backbone="mlp",
            hidden_dims=[64, 32],
            epochs=200,
            verbose=0,
            random_state=42,
        )

        result = model.fit(data)

        # ATE should be close to 2.0
        assert abs(result.ate - 2.0) < 0.3

    def test_heterogeneity_detection(self):
        """Test that heterogeneity is detected in ITE distribution."""
        # Generate data with heterogeneous effects
        data = make_ab_test(n=2000, heterogeneity="nonlinear", seed=42)

        model = ds.DeepHTE(
            formula="Y ~ a(X1 + X2 + X3) + b(X1 + X2 + X3) * T",
            family="normal",
            backbone="mlp",
            hidden_dims=[64, 32],
            epochs=100,
            verbose=0,
            random_state=42,
        )

        result = model.fit(data.data)

        # With heterogeneity, ITE range should be substantial
        ite_range = result.ite.max() - result.ite.min()
        assert ite_range > 1.0  # Should have spread

    def test_all_backbones_run(self):
        """Test that all backbones can be used."""
        data = make_ab_test(n=200, p=5, seed=42)

        for backbone in ["mlp", "transformer", "lstm"]:
            model = ds.DeepHTE(
                formula="Y ~ a(X1 + X2 + X3 + X4 + X5) + b(X1 + X2 + X3 + X4 + X5) * T",
                family="normal",
                backbone=backbone,
                epochs=10,
                verbose=0,
                random_state=42,
            )

            result = model.fit(data.data)
            assert isinstance(result.ate, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
