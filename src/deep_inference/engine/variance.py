"""
Variance estimation for influence function-based inference.

Standard estimator (from paper):
    Ψ̂ = (1/n) Σ (ψᵢ - μ̂)²
    SE = √(Ψ̂/n)
"""

from typing import Optional, Tuple
import torch
from torch import Tensor


def estimate_variance(
    psi: Tensor,
    mu_hat: Optional[float] = None,
) -> float:
    """
    Estimate variance of influence function.

    Ψ̂ = (1/n) Σ (ψᵢ - μ̂)²

    Args:
        psi: (n,) influence function values
        mu_hat: Point estimate (if None, computed from psi)

    Returns:
        Variance estimate Ψ̂
    """
    n = psi.shape[0]

    if mu_hat is None:
        mu_hat = psi.mean().item()

    # Sample variance
    variance = ((psi - mu_hat) ** 2).sum().item() / n

    return variance


def estimate_variance_bessel(
    psi: Tensor,
    mu_hat: Optional[float] = None,
) -> float:
    """
    Estimate variance with Bessel correction.

    Ψ̂ = (1/(n-1)) Σ (ψᵢ - μ̂)²

    Args:
        psi: (n,) influence function values
        mu_hat: Point estimate (if None, computed from psi)

    Returns:
        Variance estimate Ψ̂
    """
    n = psi.shape[0]

    if mu_hat is None:
        mu_hat = psi.mean().item()

    variance = ((psi - mu_hat) ** 2).sum().item() / (n - 1)

    return variance


def compute_se(
    psi: Tensor,
    mu_hat: Optional[float] = None,
    use_bessel: bool = True,
) -> float:
    """
    Compute standard error of the mean.

    SE = √(Ψ̂/n)

    Args:
        psi: (n,) influence function values
        mu_hat: Point estimate
        use_bessel: Whether to use Bessel correction

    Returns:
        Standard error
    """
    n = psi.shape[0]

    if use_bessel:
        variance = estimate_variance_bessel(psi, mu_hat)
    else:
        variance = estimate_variance(psi, mu_hat)

    se = (variance / n) ** 0.5

    return se


def compute_confidence_interval(
    mu_hat: float,
    se: float,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute confidence interval.

    CI = [μ̂ - z_{α/2} × SE, μ̂ + z_{α/2} × SE]

    Args:
        mu_hat: Point estimate
        se: Standard error
        alpha: Significance level (default: 0.05 for 95% CI)

    Returns:
        (lower, upper) confidence interval bounds
    """
    import scipy.stats as stats

    z = stats.norm.ppf(1 - alpha / 2)

    lower = mu_hat - z * se
    upper = mu_hat + z * se

    return lower, upper


def compute_inference_results(
    psi: Tensor,
    alpha: float = 0.05,
) -> dict:
    """
    Compute complete inference results.

    Args:
        psi: (n,) influence function values
        alpha: Significance level

    Returns:
        Dictionary with mu_hat, se, ci_lower, ci_upper
    """
    mu_hat = psi.mean().item()
    se = compute_se(psi, mu_hat)
    ci_lower, ci_upper = compute_confidence_interval(mu_hat, se, alpha)

    return {
        "mu_hat": mu_hat,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": psi.shape[0],
    }
