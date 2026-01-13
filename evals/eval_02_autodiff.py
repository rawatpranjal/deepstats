"""
Eval 02: Autodiff vs Calculus (All Families)

Goal: Verify torch.func autodiff matches closed-form formulas.

Criteria:
    - Score: max|autodiff - oracle| < 1e-6
    - Hessian: max|autodiff - oracle| < 1e-6

Families with closed-form oracles:
    - Linear, Logit, Poisson, NegBin, Gamma, Weibull, Gumbel (theta_dim=2)

Families using autodiff only (no oracle comparison):
    - Probit, Beta, Gaussian, Tobit, ZIP (complex derivatives)
"""

import numpy as np
import torch
from scipy.special import expit
from typing import Callable, Dict, Any, Tuple
import sys
import os

# Use local source, not installed package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# =============================================================================
# Oracle Definitions (Ground Truth) - Closed-Form Formulas
# =============================================================================


def oracle_linear(y: float, t: float, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linear: L = 0.5 * (y - alpha - beta*t)^2
    Score = (mu - y) * [1, t]
    Hessian = [[1, t], [t, t^2]]
    """
    alpha, beta = theta
    resid = (alpha + beta * t) - y
    score = np.array([resid, resid * t])
    hessian = np.array([[1.0, t], [t, t**2]])
    return score, hessian


def oracle_logit(y: float, t: float, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Logit: L = -y*log(p) - (1-y)*log(1-p) where p = sigmoid(alpha + beta*t)
    Score = (p - y) * [1, t]
    Hessian = p(1-p) * [[1, t], [t, t^2]]
    """
    alpha, beta = theta
    p = expit(alpha + beta * t)
    score = np.array([p - y, t * (p - y)])
    w = p * (1 - p)
    hessian = np.array([[w, t * w], [t * w, t**2 * w]])
    return score, hessian


def oracle_poisson(y: float, t: float, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Poisson: L = lambda - y*log(lambda) where lambda = exp(alpha + beta*t)
    Score = (lambda - y) * [1, t]
    Hessian = lambda * [[1, t], [t, t^2]]
    """
    alpha, beta = theta
    lam = np.exp(alpha + beta * t)
    score = np.array([lam - y, t * (lam - y)])
    hessian = np.array([[lam, t * lam], [t * lam, t**2 * lam]])
    return score, hessian


def oracle_negbin(y: float, t: float, theta: np.ndarray, r: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    NegBin: Log-linear rate with fixed overdispersion r
    Score = r(mu - y)/(r + mu) * [1, t]
    Hessian = r*mu*(r + y)/(r + mu)^2 * [[1, t], [t, t^2]]
    """
    alpha, beta = theta
    mu = np.exp(alpha + beta * t)
    denom = r + mu
    score_scale = r * (mu - y) / denom
    score = np.array([score_scale, t * score_scale])
    w = r * mu * (r + y) / (denom ** 2)
    hessian = np.array([[w, t * w], [t * w, t**2 * w]])
    return score, hessian


def oracle_gamma(y: float, t: float, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gamma: Deviance form L = y/mu + log(mu) where mu = exp(alpha + beta*t)
    Score = (1 - y/mu) * [1, t]
    Hessian = (y/mu) * [[1, t], [t, t^2]]
    """
    alpha, beta = theta
    mu = np.exp(alpha + beta * t)
    score_scale = 1 - y / mu
    score = np.array([score_scale, t * score_scale])
    w = y / mu
    hessian = np.array([[w, t * w], [t * w, t**2 * w]])
    return score, hessian


def oracle_weibull(y: float, t: float, theta: np.ndarray, k: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weibull: L = -log(k) + k*log(lambda) - (k-1)*log(y) + (y/lambda)^k
    where lambda = exp(alpha + beta*t)
    Score = k*(1 - z) * [1, t] where z = (y/lambda)^k
    Hessian = k^2 * z * [[1, t], [t, t^2]]
    """
    alpha, beta = theta
    lam = np.exp(alpha + beta * t)
    z = (y / lam) ** k
    score_scale = k * (1 - z)
    score = np.array([score_scale, t * score_scale])
    w = k**2 * z
    hessian = np.array([[w, t * w], [t * w, t**2 * w]])
    return score, hessian


def oracle_gumbel(y: float, t: float, theta: np.ndarray, sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gumbel: L = z + exp(-z) where z = (y - mu)/sigma, mu = alpha + beta*t
    Score = (-1/sigma)*(1 - e^(-z)) * [1, t]
    Hessian = e^(-z)/sigma^2 * [[1, t], [t, t^2]]
    """
    alpha, beta = theta
    mu = alpha + beta * t
    z = (y - mu) / sigma
    e = np.exp(-z)
    score_scale = (-1 / sigma) * (1 - e)
    score = np.array([score_scale, t * score_scale])
    w = e / (sigma ** 2)
    hessian = np.array([[w, t * w], [t * w, t**2 * w]])
    return score, hessian


# =============================================================================
# PyTorch Loss Functions (for autodiff)
# =============================================================================


def linear_loss(y: torch.Tensor, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    alpha, beta = theta[0], theta[1]
    return 0.5 * (y - (alpha + beta * t)) ** 2


def logit_loss(y: torch.Tensor, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    alpha, beta = theta[0], theta[1]
    logit = alpha + beta * t
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logit.unsqueeze(0), y.unsqueeze(0), reduction='none'
    ).squeeze()


def poisson_loss(y: torch.Tensor, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    alpha, beta = theta[0], theta[1]
    eta = alpha + beta * t
    lam = torch.exp(eta)
    return lam - y * torch.log(lam + 1e-10)


def negbin_loss(y: torch.Tensor, t: torch.Tensor, theta: torch.Tensor, r: float = 2.0) -> torch.Tensor:
    alpha, beta = theta[0], theta[1]
    mu = torch.exp(alpha + beta * t)
    # NegBin NLL (ignoring constant terms)
    return (r + y) * torch.log(r + mu) - y * torch.log(mu + 1e-10)


def gamma_loss(y: torch.Tensor, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    alpha, beta = theta[0], theta[1]
    mu = torch.exp(alpha + beta * t)
    return y / mu + torch.log(mu)


def weibull_loss(y: torch.Tensor, t: torch.Tensor, theta: torch.Tensor, k: float = 2.0) -> torch.Tensor:
    alpha, beta = theta[0], theta[1]
    lam = torch.exp(alpha + beta * t)
    return -np.log(k) + k * torch.log(lam) - (k - 1) * torch.log(y) + (y / lam) ** k


def gumbel_loss(y: torch.Tensor, t: torch.Tensor, theta: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    alpha, beta = theta[0], theta[1]
    mu = alpha + beta * t
    z = (y - mu) / sigma
    return z + torch.exp(-z)


# =============================================================================
# Test Runner
# =============================================================================


def run_autodiff_test(
    name: str,
    oracle_fn: Callable,
    loss_fn: Callable,
    n_trials: int = 20,
    seed: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Test autodiff against closed-form oracle.

    Args:
        name: Family name
        oracle_fn: Oracle function returning (score, hessian)
        loss_fn: PyTorch loss function
        n_trials: Number of random test points
        seed: Random seed
        **kwargs: Extra args passed to oracle/loss (r, k, sigma)

    Returns:
        Dict with family, score_err, hess_err, passed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    max_score_err = 0.0
    max_hess_err = 0.0

    for _ in range(n_trials):
        # Generate test point
        t = np.random.randn()
        theta = np.random.randn(2) * 0.5

        # Generate appropriate y based on family
        if name == "logit":
            y = float(np.random.binomial(1, 0.5))
        elif name == "poisson":
            y = float(np.random.poisson(2.0))
        elif name in ("gamma", "weibull"):
            y = float(np.abs(np.random.randn()) + 0.1)
        elif name == "negbin":
            y = float(np.random.poisson(2.0))
        else:
            y = np.random.randn()

        # Oracle
        if kwargs:
            s_true, h_true = oracle_fn(y, t, theta, **kwargs)
        else:
            s_true, h_true = oracle_fn(y, t, theta)

        # Autodiff
        y_t = torch.tensor(y, dtype=torch.float64)
        t_t = torch.tensor(t, dtype=torch.float64)
        th_t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)

        if kwargs:
            loss_wrapper = lambda th: loss_fn(y_t, t_t, th, **kwargs)
        else:
            loss_wrapper = lambda th: loss_fn(y_t, t_t, th)

        s_auto = torch.func.grad(loss_wrapper)(th_t).detach().numpy()
        h_auto = torch.func.hessian(loss_wrapper)(th_t).detach().numpy()

        max_score_err = max(max_score_err, np.max(np.abs(s_true - s_auto)))
        max_hess_err = max(max_hess_err, np.max(np.abs(h_true - h_auto)))

    passed = max_score_err < 1e-6 and max_hess_err < 1e-6
    return {
        "family": name,
        "score_err": max_score_err,
        "hess_err": max_hess_err,
        "passed": passed,
    }


def run_autodiff_only_test(
    name: str,
    family_class,
    n_trials: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Test that autodiff produces finite values for families without closed-form.

    Note: We only check that gradients and Hessians are finite. The PSD property
    of the Hessian is only guaranteed at the optimum θ*, not at arbitrary points.
    For complex families (Gaussian, Tobit, ZIP), the Hessian can be indefinite
    at random parameter values - this is expected and fine.

    Args:
        name: Family name
        family_class: Family class from deep_inference.families
        n_trials: Number of random test points
        seed: Random seed

    Returns:
        Dict with family, grad_finite, hess_finite, passed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    family = family_class()
    theta_dim = family.theta_dim

    all_finite = True
    hess_finite = True

    for _ in range(n_trials):
        # Generate test data
        n = 10
        t = torch.randn(n, dtype=torch.float64)
        theta = torch.randn(n, theta_dim, dtype=torch.float64) * 0.3

        # Ensure log-scale parameters (gamma for Gaussian/Tobit) are reasonable
        if name in ("gaussian", "tobit") and theta_dim >= 3:
            theta[:, 2] = torch.abs(theta[:, 2])  # gamma = log(sigma) should work

        # Generate appropriate y
        if name == "probit":
            y = torch.randint(0, 2, (n,), dtype=torch.float64)
        elif name == "beta":
            y = torch.rand(n, dtype=torch.float64) * 0.8 + 0.1  # (0.1, 0.9)
        elif name == "zip":
            y = torch.poisson(torch.ones(n) * 2.0).double()
        elif name == "gaussian":
            y = torch.randn(n, dtype=torch.float64)
        elif name == "tobit":
            y = torch.clamp(torch.randn(n, dtype=torch.float64), min=0)
        else:
            y = torch.randn(n, dtype=torch.float64)

        # Compute loss
        loss = family.loss(y, t, theta)

        if not torch.isfinite(loss).all():
            all_finite = False
            continue

        # Test autodiff on single point
        y_i = y[0:1]
        t_i = t[0:1]
        theta_i = theta[0:1].requires_grad_(True)

        def single_loss(th):
            return family.loss(y_i, t_i, th).sum()

        try:
            grad = torch.func.grad(single_loss)(theta_i)
            hess = torch.func.hessian(single_loss)(theta_i)

            if not torch.isfinite(grad).all():
                all_finite = False
            if not torch.isfinite(hess).all():
                hess_finite = False

        except Exception:
            all_finite = False

    passed = all_finite and hess_finite
    return {
        "family": name,
        "grad_finite": all_finite,
        "hess_finite": hess_finite,
        "passed": passed,
    }


def run_fitted_test(
    name: str,
    loss_fn: Callable,
    oracle_fn: Callable,
    n: int = 500,
    seed: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Test autodiff at fitted parameter values.

    1. Generate data from DGP with known θ*
    2. Fit model via gradient descent
    3. Check Hessian at θ̂ matches oracle
    4. Check Hessian is PSD at optimum

    Args:
        name: Family name
        loss_fn: PyTorch loss function
        oracle_fn: Oracle function returning (score, hessian)
        n: Sample size
        seed: Random seed
        **kwargs: Extra args (r, k, sigma)

    Returns:
        Dict with theta_hat, hess_err, min_eig, is_psd, passed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # True parameters
    alpha_true = 0.5
    beta_true = 0.3

    # Generate T
    T = np.random.randn(n)

    # Generate Y from DGP
    if name == "linear":
        Y = alpha_true + beta_true * T + np.random.randn(n) * 0.5
    elif name == "logit":
        p = expit(alpha_true + beta_true * T)
        Y = np.random.binomial(1, p).astype(float)
    elif name == "poisson":
        lam = np.exp(np.clip(alpha_true + beta_true * T, -10, 10))
        Y = np.random.poisson(lam).astype(float)
    elif name == "negbin":
        r = kwargs.get("r", 2.0)
        mu = np.exp(np.clip(alpha_true + beta_true * T, -10, 10))
        # NegBin via Poisson-Gamma mixture
        Y = np.random.negative_binomial(r, r / (r + mu)).astype(float)
    elif name == "gamma":
        mu = np.exp(alpha_true + beta_true * T)
        shape = 2.0
        Y = np.random.gamma(shape, mu / shape, size=n)
    elif name == "weibull":
        k = kwargs.get("k", 2.0)
        lam = np.exp(alpha_true + beta_true * T)
        Y = lam * np.random.weibull(k, size=n)
    elif name == "gumbel":
        sigma = kwargs.get("sigma", 1.0)
        mu = alpha_true + beta_true * T
        Y = np.random.gumbel(mu, sigma, size=n)
    else:
        raise ValueError(f"Unknown family: {name}")

    # Convert to tensors
    Y_t = torch.tensor(Y, dtype=torch.float64)
    T_t = torch.tensor(T, dtype=torch.float64)

    # Initialize theta
    theta = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)

    # Define batched loss
    def total_loss(th):
        losses = torch.stack([
            loss_fn(Y_t[i], T_t[i], th, **kwargs) if kwargs else loss_fn(Y_t[i], T_t[i], th)
            for i in range(n)
        ])
        return losses.mean()

    # Fit via LBFGS
    optimizer = torch.optim.LBFGS([theta], lr=1.0, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = total_loss(theta)
        loss.backward()
        return loss

    for _ in range(10):
        optimizer.step(closure)

    # Get fitted parameters
    theta_hat = theta.detach().numpy()

    # Compute average Hessian at θ̂ via autodiff
    hess_sum = np.zeros((2, 2))
    for i in range(n):
        y_i = Y_t[i]
        t_i = T_t[i]
        th_t = torch.tensor(theta_hat, dtype=torch.float64)

        if kwargs:
            loss_wrapper = lambda th: loss_fn(y_i, t_i, th, **kwargs)
        else:
            loss_wrapper = lambda th: loss_fn(y_i, t_i, th)

        h_auto = torch.func.hessian(loss_wrapper)(th_t).detach().numpy()
        hess_sum += h_auto

    hess_autodiff = hess_sum / n

    # Compute average Hessian via oracle
    hess_oracle_sum = np.zeros((2, 2))
    for i in range(n):
        if kwargs:
            _, h_oracle = oracle_fn(Y[i], T[i], theta_hat, **kwargs)
        else:
            _, h_oracle = oracle_fn(Y[i], T[i], theta_hat)
        hess_oracle_sum += h_oracle

    hess_oracle = hess_oracle_sum / n

    # Compare
    hess_err = np.max(np.abs(hess_autodiff - hess_oracle))

    # Check PSD
    eigvals = np.linalg.eigvalsh(hess_autodiff)
    min_eig = eigvals.min()
    is_psd = min_eig >= -1e-6

    passed = is_psd and hess_err < 1e-4

    return {
        "family": name,
        "theta_true": [alpha_true, beta_true],
        "theta_hat": theta_hat.tolist(),
        "hess_err": hess_err,
        "min_eigenvalue": min_eig,
        "is_psd": is_psd,
        "passed": passed,
    }


def run_package_test(
    name: str,
    family_class,
    n_trials: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Test actual package family implementations.

    For families with closed-form Hessian:
        - Verify autodiff Hessian matches family.hessian()

    For all families:
        - Verify autodiff gradient matches family.gradient() (if provided)
        - Verify Hessian is symmetric
        - Test at multiple random points

    Args:
        name: Family name
        family_class: Family class from deep_inference.families
        n_trials: Number of random test points
        seed: Random seed

    Returns:
        Dict with max errors and pass/fail
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    family = family_class()
    theta_dim = family.theta_dim

    max_grad_err = 0.0
    max_hess_err = 0.0
    max_symmetry_err = 0.0
    has_closed_form_grad = hasattr(family, 'gradient') and family.gradient(
        torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([[0.1] * theta_dim])
    ) is not None
    has_closed_form_hess = hasattr(family, 'hessian') and family.hessian(
        torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([[0.1] * theta_dim])
    ) is not None

    for _ in range(n_trials):
        # Generate appropriate test data
        n = 5
        t = torch.randn(n, dtype=torch.float64)
        theta = torch.randn(n, theta_dim, dtype=torch.float64) * 0.3

        # Generate appropriate y
        if name in ("logit", "probit"):
            y = torch.randint(0, 2, (n,), dtype=torch.float64)
        elif name == "beta":
            y = torch.rand(n, dtype=torch.float64) * 0.8 + 0.1
        elif name in ("poisson", "negbin", "zip"):
            y = torch.poisson(torch.ones(n) * 2.0).double()
        elif name in ("gamma", "weibull"):
            y = torch.abs(torch.randn(n, dtype=torch.float64)) + 0.1
        elif name == "tobit":
            y = torch.clamp(torch.randn(n, dtype=torch.float64), min=0)
        else:
            y = torch.randn(n, dtype=torch.float64)

        # Test single observation
        y_i = y[0:1]
        t_i = t[0:1]
        theta_i = theta[0:1].clone().requires_grad_(True)

        def single_loss(th):
            return family.loss(y_i, t_i, th).sum()

        try:
            # Autodiff gradient
            grad_auto = torch.func.grad(single_loss)(theta_i).detach()

            # Compare to closed-form gradient if available
            if has_closed_form_grad:
                grad_closed = family.gradient(y_i, t_i, theta_i.detach())
                if grad_closed is not None:
                    err = torch.max(torch.abs(grad_auto - grad_closed)).item()
                    max_grad_err = max(max_grad_err, err)

            # Autodiff Hessian
            hess_auto = torch.func.hessian(single_loss)(theta_i).detach().squeeze()

            # Check symmetry
            if hess_auto.dim() == 2:
                symmetry_err = torch.max(torch.abs(hess_auto - hess_auto.T)).item()
                max_symmetry_err = max(max_symmetry_err, symmetry_err)

            # Compare to closed-form Hessian if available
            if has_closed_form_hess:
                hess_closed = family.hessian(y_i, t_i, theta_i.detach())
                if hess_closed is not None:
                    hess_closed = hess_closed.squeeze()
                    err = torch.max(torch.abs(hess_auto - hess_closed)).item()
                    max_hess_err = max(max_hess_err, err)

        except Exception as e:
            return {
                "family": name,
                "has_closed_grad": has_closed_form_grad,
                "has_closed_hess": has_closed_form_hess,
                "grad_err": float('inf'),
                "hess_err": float('inf'),
                "symmetry_err": float('inf'),
                "passed": False,
                "error": str(e),
            }

    # Pass criteria
    grad_ok = max_grad_err < 1e-6 if has_closed_form_grad else True
    hess_ok = max_hess_err < 1e-6 if has_closed_form_hess else True
    symm_ok = max_symmetry_err < 1e-10

    passed = grad_ok and hess_ok and symm_ok

    return {
        "family": name,
        "has_closed_grad": has_closed_form_grad,
        "has_closed_hess": has_closed_form_hess,
        "grad_err": max_grad_err if has_closed_form_grad else np.nan,
        "hess_err": max_hess_err if has_closed_form_hess else np.nan,
        "symmetry_err": max_symmetry_err,
        "passed": passed,
    }


def main(seed: int = 42):
    """Run all autodiff validation tests."""

    print("=" * 70)
    print("EVAL 02: AUTODIFF VS CALCULUS (ALL FAMILIES)")
    print("=" * 70)
    print()
    print(f"Config: seed={seed}, n_trials=20")
    print()

    # Families with closed-form oracles
    oracle_tests = [
        ("linear", oracle_linear, linear_loss, {}),
        ("logit", oracle_logit, logit_loss, {}),
        ("poisson", oracle_poisson, poisson_loss, {}),
        ("negbin", oracle_negbin, negbin_loss, {"r": 2.0}),
        ("gamma", oracle_gamma, gamma_loss, {}),
        ("weibull", oracle_weibull, weibull_loss, {"k": 2.0}),
        ("gumbel", oracle_gumbel, gumbel_loss, {"sigma": 1.0}),
    ]

    print("-" * 70)
    print("PART 1: ORACLE COMPARISON (Closed-Form Families)")
    print("-" * 70)
    print()
    print(f"{'Family':<12} {'Score Err':<15} {'Hessian Err':<15} {'Status':<8}")
    print("-" * 70)

    oracle_results = []
    for name, oracle_fn, loss_fn, kwargs in oracle_tests:
        result = run_autodiff_test(name, oracle_fn, loss_fn, n_trials=20, seed=seed, **kwargs)
        oracle_results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{name:<12} {result['score_err']:<15.2e} {result['hess_err']:<15.2e} {status:<8}")

    print()

    # Families without closed-form (autodiff-only validation)
    print("-" * 70)
    print("PART 2: AUTODIFF-ONLY VALIDATION (Complex Families)")
    print("-" * 70)
    print()

    # Import families
    from deep_inference.families import (
        ProbitFamily, BetaFamily, ZIPFamily, GaussianFamily, TobitFamily
    )

    autodiff_tests = [
        ("probit", ProbitFamily),
        ("beta", BetaFamily),
        ("gaussian", GaussianFamily),
        ("tobit", TobitFamily),
        ("zip", ZIPFamily),
    ]

    print(f"{'Family':<12} {'Grad Finite':<15} {'Hess Finite':<15} {'Status':<8}")
    print("-" * 70)

    autodiff_results = []
    for name, family_class in autodiff_tests:
        result = run_autodiff_only_test(name, family_class, n_trials=20, seed=seed)
        autodiff_results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        grad_ok = "Yes" if result["grad_finite"] else "No"
        hess_ok = "Yes" if result["hess_finite"] else "No"
        print(f"{name:<12} {grad_ok:<15} {hess_ok:<15} {status:<8}")

    print()

    # Part 4: Package Integration Test
    print("-" * 70)
    print("PART 4: PACKAGE INTEGRATION (Autodiff vs Closed-Form)")
    print("-" * 70)
    print()

    from deep_inference.families import (
        LinearFamily, LogitFamily, PoissonFamily, GammaFamily,
        GumbelFamily, NegBinFamily, WeibullFamily,
        ProbitFamily, BetaFamily, GaussianFamily, TobitFamily, ZIPFamily
    )

    package_tests = [
        ("linear", LinearFamily),
        ("logit", LogitFamily),
        ("poisson", PoissonFamily),
        ("negbin", NegBinFamily),
        ("gamma", GammaFamily),
        ("weibull", WeibullFamily),
        ("gumbel", GumbelFamily),
        ("probit", ProbitFamily),
        ("beta", BetaFamily),
        ("gaussian", GaussianFamily),
        ("tobit", TobitFamily),
        ("zip", ZIPFamily),
    ]

    print(f"{'Family':<12} {'Grad Err':<12} {'Hess Err':<12} {'Symm Err':<12} {'Status':<8}")
    print("-" * 70)

    package_results = []
    for name, family_class in package_tests:
        result = run_package_test(name, family_class, n_trials=20, seed=seed)
        package_results.append(result)
        status = "PASS" if result["passed"] else "FAIL"

        grad_str = f"{result['grad_err']:.2e}" if not np.isnan(result['grad_err']) else "N/A"
        hess_str = f"{result['hess_err']:.2e}" if not np.isnan(result['hess_err']) else "N/A"
        symm_str = f"{result['symmetry_err']:.2e}"

        print(f"{name:<12} {grad_str:<12} {hess_str:<12} {symm_str:<12} {status:<8}")

    print()

    # Part 3: Fitted Parameter Validation
    print("-" * 70)
    print("PART 3: FITTED PARAMETER VALIDATION (At Optimum)")
    print("-" * 70)
    print()

    fitted_tests = [
        ("linear", linear_loss, oracle_linear, {}),
        ("logit", logit_loss, oracle_logit, {}),
        ("poisson", poisson_loss, oracle_poisson, {}),
        ("negbin", negbin_loss, oracle_negbin, {"r": 2.0}),
        ("gamma", gamma_loss, oracle_gamma, {}),
        ("weibull", weibull_loss, oracle_weibull, {"k": 2.0}),
        ("gumbel", gumbel_loss, oracle_gumbel, {"sigma": 1.0}),
    ]

    print(f"{'Family':<12} {'θ̂':<20} {'Hess Err':<12} {'Min Eig':<12} {'PSD':<6} {'Status':<8}")
    print("-" * 70)

    fitted_results = []
    for name, loss_fn, oracle_fn, kwargs in fitted_tests:
        result = run_fitted_test(name, loss_fn, oracle_fn, n=500, seed=seed, **kwargs)
        fitted_results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        psd = "Yes" if result["is_psd"] else "No"
        theta_str = f"[{result['theta_hat'][0]:.2f}, {result['theta_hat'][1]:.2f}]"
        print(f"{name:<12} {theta_str:<20} {result['hess_err']:<12.2e} {result['min_eigenvalue']:<12.4f} {psd:<6} {status:<8}")

    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    oracle_passed = sum(1 for r in oracle_results if r["passed"])
    autodiff_passed = sum(1 for r in autodiff_results if r["passed"])
    package_passed = sum(1 for r in package_results if r["passed"])
    fitted_passed = sum(1 for r in fitted_results if r["passed"])
    total_passed = oracle_passed + autodiff_passed + package_passed + fitted_passed
    total_tests = len(oracle_results) + len(autodiff_results) + len(package_results) + len(fitted_results)

    print(f"Part 1 (Oracle @ Random θ):  {oracle_passed}/{len(oracle_results)} PASS")
    print(f"Part 2 (Autodiff-Only):      {autodiff_passed}/{len(autodiff_results)} PASS")
    print(f"Part 4 (Package Integration):{package_passed}/{len(package_results)} PASS")
    print(f"Part 3 (Fitted θ̂):           {fitted_passed}/{len(fitted_results)} PASS")
    print(f"Overall:                     {total_passed}/{total_tests} PASS")
    print()
    print("=" * 70)

    return total_passed == total_tests


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Eval 02: Autodiff vs Calculus")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    success = main(seed=args.seed)
    sys.exit(0 if success else 1)
