"""
Eval 04: ψ Closed Form (Regime B)

Goal: Verify the generic ψ formula simplifies to Robinson 1988 form for linear.

Generic Formula (Theorem 2):
    ψ = H - H_θ · Λ⁻¹ · ℓ_θ

For Linear ATE:
    H = β (the slope parameter)
    H_θ = [0, 1] (Jacobian of H w.r.t. θ = [α, β])
    ℓ_θ = (α + β·t - y) · [1, t]
    Λ = [[1, E[T|X]], [E[T|X], E[T²|X]]]

Robinson 1988 Closed Form:
    ψ* = β + (T - E[T|X]) · (Y - α - β·T) / Var(T|X)

This test verifies:
    1. Generic formula (matrix inversion) gives same result as Robinson form
    2. Both formulas give E[ψ] = μ* = 0 (unbiased)

Criteria:
    - |ψ_generic - ψ_robinson| < 1e-6 for all points
    - |mean(ψ) - μ*| < 0.05 (small bias)
"""

import sys
import numpy as np
import torch

sys.path.insert(0, "/Users/pranjal/deepest/src")

from evals.dgps.regime_b_linear import (
    LinearDGP,
    generate_linear_data,
    oracle_lambda_linear,
    oracle_score_linear,
    oracle_psi_robinson,
)


def compute_psi_generic(y, t, x, theta, dgp):
    """
    Compute ψ using the generic formula:
        ψ = H - H_θ · Λ⁻¹ · ℓ_θ

    For ATE target:
        H = β
        H_θ = [0, 1]
    """
    alpha, beta = theta[0], theta[1]

    # Target: H = β
    H = beta

    # Jacobian: H_θ = [0, 1]
    H_theta = np.array([0.0, 1.0])

    # Score: ℓ_θ
    score = oracle_score_linear(y, t, theta)

    # Lambda: Λ(x)
    Lambda = oracle_lambda_linear(x, dgp)

    # Inverse with regularization
    Lambda_inv = np.linalg.inv(Lambda + 1e-8 * np.eye(2))

    # Generic formula: ψ = H - H_θ · Λ⁻¹ · ℓ_θ
    psi = H - H_theta @ Lambda_inv @ score

    return psi


def run_eval_04_psi_closed_form(n: int = 1000, verbose: bool = True):
    """
    Run ψ closed form evaluation for Regime B.
    """
    print("=" * 60)
    print("EVAL 04: ψ CLOSED FORM (Regime B)")
    print("=" * 60)

    dgp = LinearDGP()
    mu_true = dgp.mu_true()

    print("\nGeneric Formula: ψ = H - H_θ · Λ⁻¹ · ℓ_θ")
    print("\nRobinson 1988:   ψ* = β + (T - E[T|X])·(Y - α - β·T) / Var(T|X)")
    print("\nThese should be IDENTICAL for linear model.")
    print(f"\nTrue μ* = {mu_true:.6f}")

    # Generate data (use TRUE parameters for oracle comparison)
    Y, T, X, theta_true, _ = generate_linear_data(n=n, seed=42, dgp=dgp)

    # Compute both forms of ψ
    psi_generic_list = []
    psi_robinson_list = []

    for i in range(n):
        y = Y[i].item()
        t = T[i].item()
        x = X[i, 0].item()
        theta = theta_true[i].numpy()

        psi_g = compute_psi_generic(y, t, x, theta, dgp)
        psi_r = oracle_psi_robinson(y, t, x, theta, dgp)

        psi_generic_list.append(psi_g)
        psi_robinson_list.append(psi_r)

    psi_generic = np.array(psi_generic_list)
    psi_robinson = np.array(psi_robinson_list)

    # Compare
    max_diff = np.abs(psi_generic - psi_robinson).max()
    mean_generic = psi_generic.mean()
    mean_robinson = psi_robinson.mean()
    bias_generic = mean_generic - mu_true
    bias_robinson = mean_robinson - mu_true

    if verbose:
        print(f"\n--- Comparison (n={n}) ---")
        print(f"  Max |ψ_generic - ψ_robinson|: {max_diff:.2e}")
        print()
        print(f"  ψ_generic:  mean={mean_generic:.6f}, std={psi_generic.std():.6f}")
        print(f"  ψ_robinson: mean={mean_robinson:.6f}, std={psi_robinson.std():.6f}")
        print()
        print(f"  Bias (generic):  {bias_generic:.6f}")
        print(f"  Bias (robinson): {bias_robinson:.6f}")

        # Show first few values
        print(f"\n--- Sample Values (first 5) ---")
        print(f"  {'i':<5} {'ψ_generic':<12} {'ψ_robinson':<12} {'Diff':<12}")
        print("-" * 45)
        for i in range(5):
            print(f"  {i:<5} {psi_generic[i]:<12.6f} {psi_robinson[i]:<12.6f} {psi_generic[i]-psi_robinson[i]:<12.2e}")

    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION CRITERIA")
    print("=" * 60)

    criteria = {
        "Generic ≈ Robinson (max diff < 1e-6)": max_diff < 1e-6,
        "|Bias| < 0.05": abs(bias_generic) < 0.05,
    }

    all_pass = True
    for name, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("EVAL 04: PASS")
    else:
        print("EVAL 04: FAIL")
    print("=" * 60)

    return {
        "max_diff": max_diff,
        "mean_generic": mean_generic,
        "mean_robinson": mean_robinson,
        "bias": bias_generic,
        "mu_true": mu_true,
        "passed": all_pass,
    }


if __name__ == "__main__":
    result = run_eval_04_psi_closed_form(n=1000, verbose=True)
