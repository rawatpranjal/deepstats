"""
Quick validation tests for the new inference() API.
"""

import numpy as np
import torch


def test_linear_model_beta():
    """Test linear model with average beta target."""
    from deep_inference import inference

    # Generate simple linear data
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 3)
    T = np.random.randn(n)
    alpha_true = X[:, 0]  # heterogeneous intercept
    beta_true = 0.5  # constant effect
    Y = alpha_true + beta_true * T + np.random.randn(n) * 0.5

    # Run inference
    result = inference(
        Y=Y,
        T=T,
        X=X,
        model="linear",
        target="beta",
        n_folds=5,
        epochs=50,
        verbose=True,
    )

    print(f"\nLinear Model Beta Test:")
    print(f"  True beta: {beta_true}")
    print(f"  Estimated: {result.mu_hat:.4f} +/- {result.se:.4f}")
    print(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    print(f"  Regime: {result.diagnostics['regime']}")

    # Check if true value is in CI
    covered = result.ci_lower <= beta_true <= result.ci_upper
    print(f"  Covers true: {covered}")

    assert abs(result.mu_hat - beta_true) < 3 * result.se, "Estimate too far from truth"
    print("PASS: Linear model beta test")

    return result


def test_custom_loss_target():
    """Test with custom loss and target functions."""
    from deep_inference import inference
    import torch

    # Generate logit data
    np.random.seed(123)
    n = 500
    X = np.random.randn(n, 3)
    T = np.random.randn(n)
    alpha_true = X[:, 0] * 0.3
    beta_true = 0.4

    # Generate binary outcome
    logits = alpha_true + beta_true * T
    probs = 1 / (1 + np.exp(-logits))
    Y = (np.random.rand(n) < probs).astype(float)

    # Custom loss (logit)
    def my_loss(y, t, theta):
        logits = theta[0] + theta[1] * t
        p = torch.sigmoid(logits)
        eps = 1e-7
        p = torch.clamp(p, eps, 1 - eps)
        return -y * torch.log(p) - (1 - y) * torch.log(1 - p)

    # Custom target (just beta)
    def my_target(x, theta, t_tilde):
        return theta[1]

    # Run inference
    result = inference(
        Y=Y,
        T=T,
        X=X,
        loss=my_loss,
        target_fn=my_target,
        theta_dim=2,
        n_folds=5,
        epochs=50,
        verbose=True,
    )

    print(f"\nCustom Loss/Target Test:")
    print(f"  True beta: {beta_true}")
    print(f"  Estimated: {result.mu_hat:.4f} +/- {result.se:.4f}")
    print(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    print(f"  Regime: {result.diagnostics['regime']}")

    # Check reasonableness
    assert abs(result.mu_hat - beta_true) < 1.0, "Estimate unreasonably far from truth"
    print("PASS: Custom loss/target test")

    return result


def test_regime_detection():
    """Test regime detection logic."""
    from deep_inference.models import Linear, Logit
    from deep_inference.lambda_ import detect_regime, Regime

    # Linear model should be Regime B
    linear = Linear()
    regime_linear = detect_regime(linear, is_randomized=False, has_known_treatment_dist=False)
    print(f"\nLinear (observational): {regime_linear}")
    assert regime_linear == Regime.B, f"Expected Regime B for linear, got {regime_linear}"

    # Logit randomized with known F_T should be Regime A
    logit = Logit()
    regime_logit_rand = detect_regime(logit, is_randomized=True, has_known_treatment_dist=True)
    print(f"Logit (randomized, known F_T): {regime_logit_rand}")
    assert regime_logit_rand == Regime.A, f"Expected Regime A, got {regime_logit_rand}"

    # Logit observational should be Regime C
    regime_logit_obs = detect_regime(logit, is_randomized=False, has_known_treatment_dist=False)
    print(f"Logit (observational): {regime_logit_obs}")
    assert regime_logit_obs == Regime.C, f"Expected Regime C, got {regime_logit_obs}"

    print("PASS: Regime detection test")


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("Validating new inference() API")
    print("=" * 60)

    test_regime_detection()
    test_linear_model_beta()
    test_custom_loss_target()

    print("\n" + "=" * 60)
    print("ALL VALIDATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
