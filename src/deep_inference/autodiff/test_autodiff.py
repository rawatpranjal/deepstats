"""
Test autodiff functions against closed-form implementations.
"""

import torch
import numpy as np


def test_linear_score():
    """Test score computation for linear model against closed form."""
    from deep_inference.autodiff import compute_score_vmap, compute_score

    # Linear loss: (y - theta[0] - theta[1]*t)^2 / 2
    def linear_loss(y, t, theta):
        pred = theta[0] + theta[1] * t
        return 0.5 * (y - pred) ** 2

    # Closed-form score: (pred - y) * [1, t]
    def linear_score_closed(y, t, theta):
        pred = theta[:, 0] + theta[:, 1] * t
        residual = pred - y
        return torch.stack([residual, residual * t], dim=-1)

    # Test data
    n = 100
    torch.manual_seed(42)
    y = torch.randn(n)
    t = torch.randn(n)
    theta = torch.randn(n, 2)

    # Compute both
    score_vmap = compute_score_vmap(linear_loss, y, t, theta)
    score_closed = linear_score_closed(y, t, theta)

    # Check match
    diff = torch.abs(score_vmap - score_closed).max().item()
    print(f"Linear score max diff: {diff:.2e}")
    assert diff < 1e-5, f"Linear score mismatch: {diff}"
    print("PASS: Linear score matches closed form")


def test_logit_score():
    """Test score computation for logit model against closed form."""
    from deep_inference.autodiff import compute_score_vmap

    # Logit loss: -y*log(p) - (1-y)*log(1-p) where p = sigmoid(theta[0] + theta[1]*t)
    def logit_loss(y, t, theta):
        logits = theta[0] + theta[1] * t
        p = torch.sigmoid(logits)
        eps = 1e-7
        p = torch.clamp(p, eps, 1 - eps)
        return -y * torch.log(p) - (1 - y) * torch.log(1 - p)

    # Closed-form score: (p - y) * [1, t]
    def logit_score_closed(y, t, theta):
        logits = theta[:, 0] + theta[:, 1] * t
        p = torch.sigmoid(logits)
        residual = p - y
        return torch.stack([residual, residual * t], dim=-1)

    # Test data
    n = 100
    torch.manual_seed(42)
    y = torch.randint(0, 2, (n,)).float()
    t = torch.randn(n)
    theta = torch.randn(n, 2) * 0.5  # Keep small for numerical stability

    # Compute both
    score_vmap = compute_score_vmap(logit_loss, y, t, theta)
    score_closed = logit_score_closed(y, t, theta)

    # Check match
    diff = torch.abs(score_vmap - score_closed).max().item()
    print(f"Logit score max diff: {diff:.2e}")
    assert diff < 1e-5, f"Logit score mismatch: {diff}"
    print("PASS: Logit score matches closed form")


def test_linear_hessian():
    """Test Hessian computation for linear model against closed form."""
    from deep_inference.autodiff import compute_hessian_vmap

    # Linear loss
    def linear_loss(y, t, theta):
        pred = theta[0] + theta[1] * t
        return 0.5 * (y - pred) ** 2

    # Closed-form Hessian: [[1, t], [t, t^2]]
    def linear_hessian_closed(y, t, theta):
        n = len(y)
        H = torch.zeros(n, 2, 2)
        H[:, 0, 0] = 1.0
        H[:, 0, 1] = t
        H[:, 1, 0] = t
        H[:, 1, 1] = t ** 2
        return H

    # Test data
    n = 50
    torch.manual_seed(42)
    y = torch.randn(n)
    t = torch.randn(n)
    theta = torch.randn(n, 2)

    # Compute both
    hess_vmap = compute_hessian_vmap(linear_loss, y, t, theta)
    hess_closed = linear_hessian_closed(y, t, theta)

    # Check match
    diff = torch.abs(hess_vmap - hess_closed).max().item()
    print(f"Linear Hessian max diff: {diff:.2e}")
    assert diff < 1e-5, f"Linear Hessian mismatch: {diff}"
    print("PASS: Linear Hessian matches closed form")


def test_logit_hessian():
    """Test Hessian computation for logit model against closed form."""
    from deep_inference.autodiff import compute_hessian_vmap

    # Logit loss
    def logit_loss(y, t, theta):
        logits = theta[0] + theta[1] * t
        p = torch.sigmoid(logits)
        eps = 1e-7
        p = torch.clamp(p, eps, 1 - eps)
        return -y * torch.log(p) - (1 - y) * torch.log(1 - p)

    # Closed-form Hessian: p(1-p) * [[1, t], [t, t^2]]
    def logit_hessian_closed(y, t, theta):
        n = len(y)
        logits = theta[:, 0] + theta[:, 1] * t
        p = torch.sigmoid(logits)
        w = p * (1 - p)

        H = torch.zeros(n, 2, 2)
        H[:, 0, 0] = w
        H[:, 0, 1] = w * t
        H[:, 1, 0] = w * t
        H[:, 1, 1] = w * t ** 2
        return H

    # Test data
    n = 50
    torch.manual_seed(42)
    y = torch.randint(0, 2, (n,)).float()
    t = torch.randn(n)
    theta = torch.randn(n, 2) * 0.5

    # Compute both
    hess_vmap = compute_hessian_vmap(logit_loss, y, t, theta)
    hess_closed = logit_hessian_closed(y, t, theta)

    # Check match
    diff = torch.abs(hess_vmap - hess_closed).max().item()
    print(f"Logit Hessian max diff: {diff:.2e}")
    assert diff < 1e-4, f"Logit Hessian mismatch: {diff}"
    print("PASS: Logit Hessian matches closed form")


def test_target_jacobian_average_param():
    """Test target Jacobian for average parameter (H = theta_k)."""
    from deep_inference.autodiff import compute_target_jacobian_vmap

    # Target: H = theta[1] (the slope parameter)
    def target_fn(x, theta, t_tilde):
        return theta[1]

    # Closed-form Jacobian: [0, 1]
    def jacobian_closed(x, theta, t_tilde):
        n = theta.shape[0]
        J = torch.zeros(n, 2)
        J[:, 1] = 1.0
        return J

    # Test data
    n = 50
    torch.manual_seed(42)
    x = torch.randn(n, 3)
    theta = torch.randn(n, 2)
    t_tilde = torch.tensor(0.0)

    # Compute both
    jac_vmap = compute_target_jacobian_vmap(target_fn, x, theta, t_tilde)
    jac_closed = jacobian_closed(x, theta, t_tilde)

    # Check match
    diff = torch.abs(jac_vmap - jac_closed).max().item()
    print(f"Average param Jacobian max diff: {diff:.2e}")
    assert diff < 1e-5, f"Jacobian mismatch: {diff}"
    print("PASS: Average parameter Jacobian matches closed form")


def test_target_jacobian_ame():
    """Test target Jacobian for average marginal effect (logit)."""
    from deep_inference.autodiff import compute_target_jacobian_vmap

    # Target: H = p(1-p) * theta[1] where p = sigmoid(theta[0] + theta[1]*t_tilde)
    def target_fn(x, theta, t_tilde):
        logits = theta[0] + theta[1] * t_tilde
        p = torch.sigmoid(logits)
        return p * (1 - p) * theta[1]

    # Closed-form Jacobian (via chain rule):
    # dH/dtheta[0] = p(1-p)(1-2p) * theta[1]
    # dH/dtheta[1] = p(1-p) + p(1-p)(1-2p) * theta[1] * t_tilde
    def jacobian_closed(x, theta, t_tilde):
        n = theta.shape[0]
        logits = theta[:, 0] + theta[:, 1] * t_tilde
        p = torch.sigmoid(logits)
        pp = p * (1 - p)
        pp_prime = pp * (1 - 2 * p)

        J = torch.zeros(n, 2)
        J[:, 0] = pp_prime * theta[:, 1]
        J[:, 1] = pp + pp_prime * theta[:, 1] * t_tilde
        return J

    # Test data
    n = 50
    torch.manual_seed(42)
    x = torch.randn(n, 3)
    theta = torch.randn(n, 2) * 0.5
    t_tilde = torch.tensor(0.5)

    # Compute both
    jac_vmap = compute_target_jacobian_vmap(target_fn, x, theta, t_tilde)
    jac_closed = jacobian_closed(x, theta, t_tilde)

    # Check match
    diff = torch.abs(jac_vmap - jac_closed).max().item()
    print(f"AME Jacobian max diff: {diff:.2e}")
    assert diff < 1e-4, f"AME Jacobian mismatch: {diff}"
    print("PASS: AME Jacobian matches closed form")


def test_hessian_theta_dependence_detection():
    """Test detection of Hessian theta dependence."""
    from deep_inference.autodiff.hessian import (
        detect_hessian_theta_dependence,
        detect_hessian_y_dependence,
    )

    # Linear loss: Hessian doesn't depend on theta
    def linear_loss(y, t, theta):
        pred = theta[0] + theta[1] * t
        return 0.5 * (y - pred) ** 2

    # Logit loss: Hessian depends on theta (through p)
    def logit_loss(y, t, theta):
        logits = theta[0] + theta[1] * t
        p = torch.sigmoid(logits)
        eps = 1e-7
        p = torch.clamp(p, eps, 1 - eps)
        return -y * torch.log(p) - (1 - y) * torch.log(1 - p)

    # Test data
    n = 20
    torch.manual_seed(42)
    y = torch.randn(n)
    t = torch.randn(n)

    # Linear should NOT depend on theta
    linear_depends = detect_hessian_theta_dependence(linear_loss, y, t, theta_dim=2)
    print(f"Linear hessian depends on theta: {linear_depends}")
    assert not linear_depends, "Linear Hessian should NOT depend on theta"
    print("PASS: Linear Hessian theta-independence detected")

    # Logit SHOULD depend on theta
    y_binary = torch.randint(0, 2, (n,)).float()
    logit_depends = detect_hessian_theta_dependence(logit_loss, y_binary, t, theta_dim=2)
    print(f"Logit hessian depends on theta: {logit_depends}")
    assert logit_depends, "Logit Hessian SHOULD depend on theta"
    print("PASS: Logit Hessian theta-dependence detected")

    # Both should NOT depend on Y (for standard GLMs)
    linear_y_depends = detect_hessian_y_dependence(linear_loss, y, t, theta_dim=2)
    logit_y_depends = detect_hessian_y_dependence(logit_loss, y_binary, t, theta_dim=2)
    print(f"Linear hessian depends on Y: {linear_y_depends}")
    print(f"Logit hessian depends on Y: {logit_y_depends}")
    assert not linear_y_depends, "Linear Hessian should NOT depend on Y"
    assert not logit_y_depends, "Logit Hessian should NOT depend on Y"
    print("PASS: Hessian Y-independence detected for both models")


def run_all_tests():
    """Run all autodiff tests."""
    print("=" * 60)
    print("Testing vmap-based autodiff against closed-form implementations")
    print("=" * 60)

    test_linear_score()
    print()

    test_logit_score()
    print()

    test_linear_hessian()
    print()

    test_logit_hessian()
    print()

    test_target_jacobian_average_param()
    print()

    test_target_jacobian_ame()
    print()

    test_hessian_theta_dependence_detection()
    print()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
