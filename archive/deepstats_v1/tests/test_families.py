"""Tests for distribution families."""

import numpy as np
import pytest
import torch


class TestNormalFamily:
    """Tests for Normal distribution family."""

    def test_link_inverse_link(self):
        """Identity link should be self-inverse."""
        from deepstats.families import Normal

        family = Normal()
        mu = torch.tensor([1.0, 2.0, 3.0])

        eta = family.link(mu)
        mu_back = family.inverse_link(eta)

        assert torch.allclose(mu, mu_back)

    def test_variance(self):
        """Normal variance should be constant 1."""
        from deepstats.families import Normal

        family = Normal()
        mu = torch.tensor([1.0, 10.0, 100.0])
        V = family.variance(mu)

        assert torch.allclose(V, torch.ones_like(mu))

    def test_log_likelihood_shape(self):
        """Log-likelihood should return correct shape."""
        from deepstats.families import Normal

        family = Normal()
        y = torch.tensor([1.0, 2.0, 3.0])
        mu = torch.tensor([1.1, 1.9, 3.1])

        ll = family.log_likelihood(y, mu, dispersion=1.0)
        assert ll.shape == y.shape

    def test_log_likelihood_maximum_at_true(self):
        """Log-likelihood should be maximized at y=mu."""
        from deepstats.families import Normal

        family = Normal()
        y = torch.tensor([2.0])
        mu_true = torch.tensor([2.0])
        mu_wrong = torch.tensor([5.0])

        ll_true = family.log_likelihood(y, mu_true, dispersion=1.0)
        ll_wrong = family.log_likelihood(y, mu_wrong, dispersion=1.0)

        assert ll_true > ll_wrong

    def test_unit_deviance(self):
        """Unit deviance should be (y-mu)^2 for Normal."""
        from deepstats.families import Normal

        family = Normal()
        y = torch.tensor([1.0, 2.0, 3.0])
        mu = torch.tensor([1.5, 2.0, 2.5])

        d = family.unit_deviance(y, mu)
        expected = (y - mu) ** 2

        assert torch.allclose(d, expected)


class TestPoissonFamily:
    """Tests for Poisson distribution family."""

    def test_link_inverse_link(self):
        """Log link and exp inverse should be consistent."""
        from deepstats.families import Poisson

        family = Poisson()
        mu = torch.tensor([1.0, 2.0, 3.0])

        eta = family.link(mu)
        mu_back = family.inverse_link(eta)

        assert torch.allclose(mu, mu_back, rtol=1e-5)

    def test_variance_equals_mean(self):
        """Poisson variance should equal mean."""
        from deepstats.families import Poisson

        family = Poisson()
        mu = torch.tensor([1.0, 5.0, 10.0])
        V = family.variance(mu)

        assert torch.allclose(V, mu)

    def test_validates_negative_response(self):
        """Poisson should reject negative responses."""
        from deepstats.families import Poisson

        family = Poisson()
        y = torch.tensor([-1.0, 0.0, 1.0])

        with pytest.raises(ValueError):
            family.validate_response(y)

    def test_log_likelihood_shape(self):
        """Log-likelihood should have correct shape."""
        from deepstats.families import Poisson

        family = Poisson()
        y = torch.tensor([0.0, 1.0, 5.0])
        mu = torch.tensor([1.0, 2.0, 4.0])

        ll = family.log_likelihood(y, mu)
        assert ll.shape == y.shape


class TestBernoulliFamily:
    """Tests for Bernoulli distribution family."""

    def test_sigmoid_link(self):
        """Logit and sigmoid should be inverses."""
        from deepstats.families import Bernoulli

        family = Bernoulli()
        mu = torch.tensor([0.2, 0.5, 0.8])

        eta = family.link(mu)
        mu_back = family.inverse_link(eta)

        assert torch.allclose(mu, mu_back, rtol=1e-5)

    def test_variance_binomial(self):
        """Bernoulli variance should be mu*(1-mu)."""
        from deepstats.families import Bernoulli

        family = Bernoulli()
        mu = torch.tensor([0.3, 0.5, 0.7])
        V = family.variance(mu)

        expected = mu * (1 - mu)
        assert torch.allclose(V, expected, rtol=1e-5)

    def test_validates_out_of_range(self):
        """Bernoulli should reject responses outside [0, 1]."""
        from deepstats.families import Bernoulli

        family = Bernoulli()
        y = torch.tensor([-0.5, 0.5, 1.5])

        with pytest.raises(ValueError):
            family.validate_response(y)

    def test_probit_link(self):
        """Probit link should also be invertible."""
        from deepstats.families import Bernoulli

        family = Bernoulli(link="probit")
        mu = torch.tensor([0.2, 0.5, 0.8])

        eta = family.link(mu)
        mu_back = family.inverse_link(eta)

        assert torch.allclose(mu, mu_back, rtol=1e-4)


class TestGammaFamily:
    """Tests for Gamma distribution family."""

    def test_link_inverse_link_log(self):
        """Log link and exp inverse should be consistent."""
        from deepstats.families import Gamma

        family = Gamma(link="log")
        mu = torch.tensor([1.0, 2.0, 3.0])

        eta = family.link(mu)
        mu_back = family.inverse_link(eta)

        assert torch.allclose(mu, mu_back, rtol=1e-5)

    def test_link_inverse_link_inverse(self):
        """Inverse link should work correctly."""
        from deepstats.families import Gamma

        family = Gamma(link="inverse")
        mu = torch.tensor([1.0, 2.0, 0.5])

        eta = family.link(mu)  # eta = 1/mu
        mu_back = family.inverse_link(eta)  # mu = 1/eta

        assert torch.allclose(mu, mu_back, rtol=1e-5)

    def test_variance_quadratic(self):
        """Gamma variance should be mu^2."""
        from deepstats.families import Gamma

        family = Gamma()
        mu = torch.tensor([1.0, 2.0, 3.0])
        V = family.variance(mu)

        expected = mu ** 2
        assert torch.allclose(V, expected)

    def test_validates_nonpositive_response(self):
        """Gamma should reject non-positive responses."""
        from deepstats.families import Gamma

        family = Gamma()
        y = torch.tensor([0.0, 1.0, 2.0])  # 0 is invalid

        with pytest.raises(ValueError):
            family.validate_response(y)

    def test_log_likelihood_shape(self):
        """Log-likelihood should have correct shape."""
        from deepstats.families import Gamma

        family = Gamma()
        y = torch.tensor([1.0, 2.0, 3.0])
        mu = torch.tensor([1.1, 2.1, 2.9])

        ll = family.log_likelihood(y, mu, dispersion=1.0)
        assert ll.shape == y.shape

    def test_unit_deviance_at_mu(self):
        """Unit deviance should be 0 when y=mu."""
        from deepstats.families import Gamma

        family = Gamma()
        y = torch.tensor([1.0, 2.0, 3.0])
        mu = y.clone()

        d = family.unit_deviance(y, mu)
        assert torch.allclose(d, torch.zeros_like(d), atol=1e-6)


class TestExponentialFamily:
    """Tests for Exponential distribution family."""

    def test_link_inverse_link_log(self):
        """Log link and exp inverse should be consistent."""
        from deepstats.families import Exponential

        family = Exponential(link="log")
        mu = torch.tensor([1.0, 2.0, 3.0])

        eta = family.link(mu)
        mu_back = family.inverse_link(eta)

        assert torch.allclose(mu, mu_back, rtol=1e-5)

    def test_link_inverse_link_inverse(self):
        """Inverse link should work correctly."""
        from deepstats.families import Exponential

        family = Exponential(link="inverse")
        mu = torch.tensor([1.0, 2.0, 0.5])

        eta = family.link(mu)
        mu_back = family.inverse_link(eta)

        assert torch.allclose(mu, mu_back, rtol=1e-5)

    def test_variance_quadratic(self):
        """Exponential variance should be mu^2."""
        from deepstats.families import Exponential

        family = Exponential()
        mu = torch.tensor([1.0, 2.0, 3.0])
        V = family.variance(mu)

        expected = mu ** 2
        assert torch.allclose(V, expected)

    def test_validates_nonpositive_response(self):
        """Exponential should reject non-positive responses."""
        from deepstats.families import Exponential

        family = Exponential()
        y = torch.tensor([0.0, 1.0, 2.0])

        with pytest.raises(ValueError):
            family.validate_response(y)

    def test_log_likelihood_shape(self):
        """Log-likelihood should have correct shape."""
        from deepstats.families import Exponential

        family = Exponential()
        y = torch.tensor([1.0, 2.0, 3.0])
        mu = torch.tensor([1.1, 2.1, 2.9])

        ll = family.log_likelihood(y, mu)
        assert ll.shape == y.shape

    def test_no_dispersion(self):
        """Exponential should have no dispersion parameter."""
        from deepstats.families import Exponential

        family = Exponential()
        assert family.has_dispersion is False


class TestDeepHTEFamilies:
    """Tests for DeepHTE with different families."""

    def test_deephte_gamma_family(self):
        """DeepHTE with gamma family should work."""
        import deepstats as ds
        import pandas as pd

        np.random.seed(42)
        n = 300
        X1 = np.random.randn(n)
        X2 = np.random.randn(n)
        T = np.random.binomial(1, 0.5, n)
        # Gamma outcomes (positive)
        Y = np.exp(1.0 + 0.3 * X1 + 0.5 * T + 0.2 * np.random.randn(n))

        data = pd.DataFrame({'Y': Y, 'X1': X1, 'X2': X2, 'T': T})

        model = ds.DeepHTE(
            formula='Y ~ a(X1 + X2) + b(X1 + X2) * T',
            family='gamma',
            backbone='mlp',
            epochs=30,
            verbose=0,
            random_state=42,
        )
        result = model.fit(data)

        assert result.family == 'gamma'
        assert not np.isnan(result.ate)
        assert result.ate_se > 0

    def test_deephte_poisson_family(self):
        """DeepHTE with poisson family should work."""
        import deepstats as ds
        import pandas as pd

        np.random.seed(42)
        n = 300
        X1 = np.random.randn(n)
        X2 = np.random.randn(n)
        T = np.random.binomial(1, 0.5, n)
        # Poisson outcomes (counts)
        lam = np.exp(1.0 + 0.3 * X1 + 0.5 * T)
        Y = np.random.poisson(lam).astype(float)

        data = pd.DataFrame({'Y': Y, 'X1': X1, 'X2': X2, 'T': T})

        model = ds.DeepHTE(
            formula='Y ~ a(X1 + X2) + b(X1 + X2) * T',
            family='poisson',
            backbone='mlp',
            epochs=30,
            verbose=0,
            random_state=42,
        )
        result = model.fit(data)

        assert result.family == 'poisson'
        assert not np.isnan(result.ate)
        assert result.ate_se > 0

    def test_deephte_exponential_family(self):
        """DeepHTE with exponential family should work."""
        import deepstats as ds
        import pandas as pd

        np.random.seed(42)
        n = 300
        X1 = np.random.randn(n)
        X2 = np.random.randn(n)
        T = np.random.binomial(1, 0.5, n)
        # Exponential outcomes (positive durations)
        rate = np.exp(-(1.0 + 0.3 * X1 + 0.5 * T))
        Y = np.random.exponential(1 / rate)

        data = pd.DataFrame({'Y': Y, 'X1': X1, 'X2': X2, 'T': T})

        model = ds.DeepHTE(
            formula='Y ~ a(X1 + X2) + b(X1 + X2) * T',
            family='exponential',
            backbone='mlp',
            epochs=30,
            verbose=0,
            random_state=42,
        )
        result = model.fit(data)

        assert result.family == 'exponential'
        assert not np.isnan(result.ate)
        assert result.ate_se > 0


class TestLinearOLS:
    """Tests for LinearOLS estimator."""

    def test_fit_returns_results(self):
        """LinearOLS fit should return DeepResults."""
        from deepstats.estimators import LinearOLS
        from deepstats.results import DeepResults

        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)
        y = X @ np.array([1.0, -0.5, 0.3]) + np.random.randn(n) * 0.5

        model = LinearOLS(robust_se="HC1")
        result = model.fit(X, y)

        assert isinstance(result, DeepResults)
        assert len(result.params) == p + 1  # includes intercept

    def test_recovers_true_coefficients(self):
        """LinearOLS should recover true coefficients."""
        from deepstats.estimators import LinearOLS

        np.random.seed(42)
        n = 1000
        X = np.random.randn(n, 2)
        beta_true = np.array([0.5, -0.3])
        y = 2.0 + X @ beta_true + np.random.randn(n) * 0.1

        model = LinearOLS(robust_se="HC1")
        result = model.fit(X, y)

        # First coefficient is intercept
        assert abs(result.params[0] - 2.0) < 0.1
        assert abs(result.params[1] - 0.5) < 0.1
        assert abs(result.params[2] - (-0.3)) < 0.1


class TestDeepGLM:
    """Tests for DeepGLM estimator."""

    def test_normal_family(self):
        """DeepGLM with normal family should work."""
        from deepstats.estimators import DeepGLM

        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 2)
        y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + np.random.randn(n) * 0.5

        model = DeepGLM(family="normal", epochs=30, verbose=0, random_state=42)
        result = model.fit(X, y)

        assert result.family == "normal"
        assert not np.isnan(result.params).any()
        assert not np.isnan(result.std_errors).any()

    def test_poisson_family(self):
        """DeepGLM with poisson family should work."""
        from deepstats.estimators import DeepGLM

        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        lam = np.exp(0.5 + 0.3 * X[:, 0] + 0.2 * X[:, 1])
        y = np.random.poisson(lam)

        model = DeepGLM(family="poisson", epochs=50, verbose=0, random_state=42)
        result = model.fit(X, y.astype(float))

        assert result.family == "poisson"
        assert not np.isnan(result.params).any()

    def test_bernoulli_family(self):
        """DeepGLM with bernoulli family should work."""
        from deepstats.estimators import DeepGLM

        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        p = 1 / (1 + np.exp(-(0.5 + 0.8 * X[:, 0] - 0.3 * X[:, 1])))
        y = np.random.binomial(1, p)

        model = DeepGLM(family="bernoulli", epochs=50, verbose=0, random_state=42)
        result = model.fit(X, y.astype(float))

        assert result.family == "bernoulli"
        # Predictions should be in (0, 1)
        preds = model.predict(X)
        assert (preds > 0).all() and (preds < 1).all()


class TestDatasetGenerators:
    """Tests for dataset generators."""

    def test_linear_highdim(self):
        """Linear high-dim generator should work."""
        from deepstats.datasets import make_linear_highdim

        X, y, meta = make_linear_highdim(n=100, p=20, sparsity=0.3, seed=42)

        assert X.shape == (100, 20)
        assert y.shape == (100,)
        assert "beta_true" in meta
        assert len(meta["beta_true"]) == 20

    def test_poisson_highdim(self):
        """Poisson high-dim generator should work."""
        from deepstats.datasets import make_poisson_highdim

        X, y, meta = make_poisson_highdim(n=100, p=20, seed=42)

        assert X.shape == (100, 20)
        assert y.shape == (100,)
        assert (y >= 0).all()  # counts are non-negative

    def test_binary_highdim(self):
        """Binary high-dim generator should work."""
        from deepstats.datasets import make_binary_highdim

        X, y, meta = make_binary_highdim(n=100, p=20, seed=42)

        assert X.shape == (100, 20)
        assert y.shape == (100,)
        assert set(np.unique(y)).issubset({0.0, 1.0})

    def test_nonlinear_highdim(self):
        """Non-linear high-dim generator should work."""
        from deepstats.datasets import make_nonlinear_highdim

        X, y, meta = make_nonlinear_highdim(n=100, p=20, complexity="medium", seed=42)

        assert X.shape == (100, 20)
        assert y.shape == (100,)
        assert "g_description" in meta
