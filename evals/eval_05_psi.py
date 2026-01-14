"""
Eval 05: Influence Function Assembly (ψ) — RUTHLESS VERSION

Goal: Verify assembled ψ matches Oracle ψ with RUTHLESS tolerances.

Formula (Theorem 2):
    ψ_i = H(θ_i) - H_θ(θ_i) · Λ(x_i)⁻¹ · ℓ_θ(y_i, t_i, θ_i)

ROUND A: Mechanical Assembly (identical inputs)
    - Corr(ψ̂, ψ*) > 0.999
    - |Bias| < 0.001
    - Max|diff| < 0.01
    - RMSE < 0.01

ROUND B: Neyman Orthogonality
    - Perturb θ, verify bias scales as O(δ²)

ROUND C: Variance Formula
    - Verify SE estimation

ROUND D: Multi-Seed Coverage (M=50)
    - Coverage should be 88-98%
"""

import sys
import numpy as np
import torch
from torch import Tensor
from scipy.special import expit
from scipy.stats import pearsonr
from typing import Tuple, Dict, Any, List

sys.path.insert(0, "/Users/pranjal/deepest/src")

from evals.dgp import (
    generate_canonical_dgp,
    CanonicalDGP,
    oracle_score,
    oracle_hessian,
    oracle_target_jacobian,
    oracle_lambda_conditional,
    oracle_jacobian_average_parameter,
)


# ============================================================
# ORACLE COMPUTATIONS
# ============================================================

def oracle_h_ame(theta: np.ndarray, t_tilde: float = 0.0) -> float:
    """Oracle AME: H(θ) = σ(α + β·t̃)·(1 - σ(α + β·t̃))·β"""
    alpha, beta = theta[0], theta[1]
    s = expit(alpha + beta * t_tilde)
    return s * (1 - s) * beta


def oracle_h_average_param(theta: np.ndarray, t_tilde: float = 0.0) -> float:
    """Oracle AverageParameter: H(θ) = β"""
    return theta[1]


def compute_oracle_psi_batch(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    theta: np.ndarray,
    Lambda_inv: np.ndarray,
    oracle_h_fn,
    oracle_jacobian_fn,
    t_tilde: float = 0.0,
) -> np.ndarray:
    """
    Compute Oracle ψ for all observations using provided Lambda_inv.

    This ensures IDENTICAL inputs to package for fair comparison.
    """
    n = len(Y)
    psi_oracle = np.zeros(n)

    for i in range(n):
        # H(θ)
        H = oracle_h_fn(theta[i], t_tilde)

        # H_θ(θ)
        H_theta = oracle_jacobian_fn(theta[i], t_tilde)

        # ℓ_θ
        score = oracle_score(Y[i], T[i], theta[i])

        # ψ = H - H_θ · Λ⁻¹ · ℓ_θ
        correction = H_theta @ Lambda_inv @ score
        psi_oracle[i] = H - correction

    return psi_oracle


def compute_package_psi(
    Y: Tensor,
    T: Tensor,
    X: Tensor,
    theta: Tensor,
    target_name: str,
    t_tilde: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ψ using the package's assembler.

    Returns:
        psi: (n,) influence values
        Lambda_inv_mean: (2, 2) mean inverse Lambda (for oracle comparison)
    """
    from deep_inference.models import Logit
    from deep_inference.targets import AME, AverageParameter
    from deep_inference.lambda_.estimate import EstimateLambda
    from deep_inference.engine.assembler import compute_psi
    from deep_inference.utils.linalg import batch_inverse

    model = Logit()

    # Select target
    if target_name == "AME":
        target = AME(param_index=1, model_type="logit")
    elif target_name == "AverageParameter":
        target = AverageParameter(param_index=1)
    else:
        raise ValueError(f"Unknown target: {target_name}")

    # Use aggregate Lambda
    lambda_strategy = EstimateLambda(method="aggregate")
    lambda_strategy.fit(X, T, Y, theta, model)

    t_tilde_t = torch.tensor(t_tilde, dtype=torch.float32)
    lambda_matrices = lambda_strategy.predict(X, theta)

    # Get Lambda_inv for oracle comparison
    lambda_inv = batch_inverse(lambda_matrices, ridge=1e-6)
    Lambda_inv_mean = lambda_inv[0].numpy()  # All same for aggregate

    psi = compute_psi(
        Y=Y,
        T=T,
        X=X,
        theta_hat=theta,
        t_tilde=t_tilde_t,
        lambda_matrices=lambda_matrices,
        model=model,
        target=target,
        ridge=1e-6,
    )

    return psi.numpy(), Lambda_inv_mean


def compute_psi_metrics(psi_package: np.ndarray, psi_oracle: np.ndarray) -> Dict[str, float]:
    """Compute ψ assembly metrics."""
    corr, p_value = pearsonr(psi_package, psi_oracle)
    bias = psi_package.mean() - psi_oracle.mean()
    max_diff = np.max(np.abs(psi_package - psi_oracle))
    rmse = np.sqrt(np.mean((psi_package - psi_oracle) ** 2))

    return {
        "correlation": corr,
        "bias": bias,
        "max_diff": max_diff,
        "rmse": rmse,
        "mean_package": psi_package.mean(),
        "mean_oracle": psi_oracle.mean(),
        "std_package": psi_package.std(),
        "std_oracle": psi_oracle.std(),
    }


# ============================================================
# ROUND A: MECHANICAL ASSEMBLY
# ============================================================

def run_round_a(n: int, seed: int, verbose: bool = True) -> Dict[str, Any]:
    """
    Round A: Test mechanical assembly with RUTHLESS tolerances.

    Uses identical inputs (true θ*, same Λ) for oracle and package.
    """
    print("\n" + "=" * 60)
    print("ROUND A: MECHANICAL ASSEMBLY (identical inputs)")
    print("=" * 60)

    dgp = CanonicalDGP()
    t_tilde = 0.0

    # Generate data
    Y, T, X, theta_true, mu_true = generate_canonical_dgp(n=n, seed=seed, dgp=dgp)

    targets = [
        ("AME", oracle_h_ame, oracle_target_jacobian),
        ("AverageParameter", oracle_h_average_param, oracle_jacobian_average_parameter),
    ]

    results = {}
    all_pass = True

    for target_name, oracle_h_fn, oracle_jacobian_fn in targets:
        print(f"\n  Target: {target_name}")

        # Compute package ψ (also returns Lambda_inv)
        psi_package, Lambda_inv = compute_package_psi(
            Y, T, X, theta_true, target_name, t_tilde
        )

        # Compute oracle ψ with SAME Lambda_inv
        psi_oracle = compute_oracle_psi_batch(
            Y.numpy(), T.numpy(), X.numpy(), theta_true.numpy(),
            Lambda_inv, oracle_h_fn, oracle_jacobian_fn, t_tilde
        )

        # Compute metrics
        metrics = compute_psi_metrics(psi_package, psi_oracle)

        # RUTHLESS criteria
        criteria = {
            "Corr > 0.999": (metrics["correlation"] > 0.999, metrics["correlation"]),
            "|Bias| < 0.001": (abs(metrics["bias"]) < 0.001, abs(metrics["bias"])),
            "Max|diff| < 0.01": (metrics["max_diff"] < 0.01, metrics["max_diff"]),
            "RMSE < 0.01": (metrics["rmse"] < 0.01, metrics["rmse"]),
        }

        target_pass = True
        for name, (passed, value) in criteria.items():
            status = "PASS" if passed else "FAIL"
            print(f"    {name}: {value:.6f} [{status}]")
            if not passed:
                target_pass = False
                all_pass = False

        results[target_name] = {
            "metrics": metrics,
            "criteria": criteria,
            "passed": target_pass,
        }

        # Show worst discrepancies if FAIL
        if not target_pass:
            diffs = np.abs(psi_package - psi_oracle)
            worst_indices = np.argsort(diffs)[-5:][::-1]
            print(f"    Worst discrepancies:")
            for idx in worst_indices:
                print(f"      i={idx}: oracle={psi_oracle[idx]:.6f}, "
                      f"package={psi_package[idx]:.6f}, diff={diffs[idx]:.6f}")

    print(f"\n  ROUND A: {'PASS' if all_pass else 'FAIL'}")

    return {"results": results, "passed": all_pass, "mu_true": mu_true}


# ============================================================
# ROUND B: NEYMAN ORTHOGONALITY
# ============================================================

def run_round_b(n: int, seed: int, verbose: bool = True) -> Dict[str, Any]:
    """
    Round B: Test Neyman orthogonality.

    The influence function should be first-order insensitive to θ errors.
    Bias should scale as O(δ²), not O(δ).
    """
    print("\n" + "=" * 60)
    print("ROUND B: NEYMAN ORTHOGONALITY")
    print("=" * 60)

    dgp = CanonicalDGP()
    t_tilde = 0.0

    # Generate data
    Y, T, X, theta_true, mu_true = generate_canonical_dgp(n=n, seed=seed, dgp=dgp)

    # Baseline: ψ with true θ
    psi_true, Lambda_inv = compute_package_psi(Y, T, X, theta_true, "AME", t_tilde)
    mu_hat_true = psi_true.mean()

    perturbation_sizes = [0.01, 0.05, 0.1]
    results = {}
    all_pass = True

    np.random.seed(seed + 1000)  # Fixed seed for perturbations

    for delta in perturbation_sizes:
        # Perturb theta
        perturbation = delta * torch.randn_like(theta_true)
        theta_perturbed = theta_true + perturbation

        # Compute ψ with perturbed θ
        psi_perturbed, _ = compute_package_psi(Y, T, X, theta_perturbed, "AME", t_tilde)
        mu_hat_perturbed = psi_perturbed.mean()

        # Bias from true μ*
        bias = abs(mu_hat_perturbed - mu_true)

        # Orthogonality says bias ~ O(δ²)
        # We allow bias < 10 * δ² (generous constant)
        threshold = 10 * delta ** 2
        passed = bias < threshold

        status = "PASS" if passed else "FAIL"
        print(f"  δ={delta:.2f}: bias={bias:.6f}, threshold={threshold:.6f} [{status}]")

        if not passed:
            all_pass = False

        results[delta] = {
            "bias": bias,
            "threshold": threshold,
            "passed": passed,
        }

    print(f"\n  ROUND B: {'PASS' if all_pass else 'FAIL'}")

    return {"results": results, "passed": all_pass}


# ============================================================
# ROUND C: VARIANCE FORMULA
# ============================================================

def run_round_c(n: int, seed: int, verbose: bool = True) -> Dict[str, Any]:
    """
    Round C: Verify variance formula.

    Theorem 3: sqrt(n)(μ̂ - μ*) →_d N(0, V) where V = Var(ψ)

    The sample variance of ψ should give valid SE estimates.
    """
    print("\n" + "=" * 60)
    print("ROUND C: VARIANCE FORMULA")
    print("=" * 60)

    dgp = CanonicalDGP()
    t_tilde = 0.0

    # Generate data
    Y, T, X, theta_true, mu_true = generate_canonical_dgp(n=n, seed=seed, dgp=dgp)

    # Compute ψ
    psi, Lambda_inv = compute_package_psi(Y, T, X, theta_true, "AME", t_tilde)

    # Variance and SE
    empirical_var = psi.var()
    se_estimate = np.sqrt(empirical_var / n)
    mu_hat = psi.mean()

    # CI
    ci_lower = mu_hat - 1.96 * se_estimate
    ci_upper = mu_hat + 1.96 * se_estimate
    covers = ci_lower <= mu_true <= ci_upper

    print(f"  Empirical Var(ψ): {empirical_var:.6f}")
    print(f"  SE estimate: {se_estimate:.6f}")
    print(f"  μ̂: {mu_hat:.6f}")
    print(f"  μ*: {mu_true:.6f}")
    print(f"  95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"  Covers μ*: {covers}")

    # Basic sanity checks
    criteria = {
        "Var(ψ) > 0": empirical_var > 0,
        "SE > 0": se_estimate > 0,
        "SE < 1": se_estimate < 1,  # Reasonable magnitude
    }

    all_pass = all(criteria.values())

    for name, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    print(f"\n  ROUND C: {'PASS' if all_pass else 'FAIL'}")

    return {
        "empirical_var": empirical_var,
        "se_estimate": se_estimate,
        "mu_hat": mu_hat,
        "mu_true": mu_true,
        "ci": (ci_lower, ci_upper),
        "covers": covers,
        "passed": all_pass,
    }


# ============================================================
# ROUND D: MULTI-SEED COVERAGE
# ============================================================

def run_round_d(n: int, M: int = 50, verbose: bool = True) -> Dict[str, Any]:
    """
    Round D: Multi-seed coverage test.

    Run M seeds, compute ψ for each, check coverage.
    Expected: 88-98% (allowing MC noise).
    """
    print("\n" + "=" * 60)
    print(f"ROUND D: MULTI-SEED COVERAGE (M={M})")
    print("=" * 60)

    dgp = CanonicalDGP()
    t_tilde = 0.0
    mu_true = dgp.mu_true()

    covers = 0
    biases = []
    ses = []
    mu_hats = []

    for seed in range(M):
        # Generate data
        Y, T, X, theta_true, _ = generate_canonical_dgp(n=n, seed=seed, dgp=dgp)

        # Compute ψ
        psi, _ = compute_package_psi(Y, T, X, theta_true, "AME", t_tilde)

        mu_hat = psi.mean()
        se = psi.std() / np.sqrt(n)
        ci_lower = mu_hat - 1.96 * se
        ci_upper = mu_hat + 1.96 * se

        if ci_lower <= mu_true <= ci_upper:
            covers += 1

        biases.append(mu_hat - mu_true)
        ses.append(se)
        mu_hats.append(mu_hat)

        if verbose and seed < 5:
            cov_str = "✓" if ci_lower <= mu_true <= ci_upper else "✗"
            print(f"  Seed {seed}: μ̂={mu_hat:.4f}, SE={se:.4f}, "
                  f"CI=[{ci_lower:.4f}, {ci_upper:.4f}] {cov_str}")

    if verbose and M > 5:
        print(f"  ... ({M-5} more seeds)")

    coverage = covers / M
    mean_bias = np.mean(biases)
    mean_se = np.mean(ses)
    empirical_se = np.std(mu_hats)

    print(f"\n  Coverage: {covers}/{M} = {coverage*100:.1f}%")
    print(f"  Mean bias: {mean_bias:.6f}")
    print(f"  Mean SE: {mean_se:.6f}")
    print(f"  Empirical SE: {empirical_se:.6f}")
    print(f"  SE ratio (mean/empirical): {mean_se/empirical_se:.3f}")

    # Criteria: coverage between 88% and 98%
    passed = 0.88 <= coverage <= 0.98
    status = "PASS" if passed else "FAIL"
    print(f"  Coverage in [88%, 98%]: {status}")

    print(f"\n  ROUND D: {'PASS' if passed else 'FAIL'}")

    return {
        "coverage": coverage,
        "covers": covers,
        "M": M,
        "mean_bias": mean_bias,
        "mean_se": mean_se,
        "empirical_se": empirical_se,
        "se_ratio": mean_se / empirical_se,
        "passed": passed,
    }


# ============================================================
# MAIN
# ============================================================

def run_eval_05(n: int = 1000, seed: int = 42, M_coverage: int = 50, verbose: bool = True):
    """Run all rounds of eval 05."""
    print("=" * 60)
    print("EVAL 05: INFLUENCE FUNCTION ASSEMBLY (RUTHLESS)")
    print("=" * 60)
    print(f"\nFormula (Theorem 2):")
    print(f"  ψ_i = H(θ_i) - H_θ(θ_i) · Λ(x_i)⁻¹ · ℓ_θ(y_i, t_i, θ_i)")
    print(f"\nn={n}, seed={seed}, M_coverage={M_coverage}")

    # Run all rounds
    round_a = run_round_a(n, seed, verbose)
    round_b = run_round_b(n, seed, verbose)
    round_c = run_round_c(n, seed, verbose)
    round_d = run_round_d(n, M_coverage, verbose)

    # Final verdict
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)

    all_pass = True
    for name, result in [("A", round_a), ("B", round_b), ("C", round_c), ("D", round_d)]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  Round {name}: {status}")
        if not result["passed"]:
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("EVAL 05: PASS")
    else:
        print("EVAL 05: FAIL")
    print("=" * 60)

    return {
        "round_a": round_a,
        "round_b": round_b,
        "round_c": round_c,
        "round_d": round_d,
        "passed": all_pass,
    }


if __name__ == "__main__":
    result = run_eval_05(n=1000, seed=42, M_coverage=50)
