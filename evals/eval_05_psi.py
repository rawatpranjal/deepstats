"""
Eval 05: Influence Function Assembly (ψ)

Goal: Verify assembled ψ matches Oracle ψ using all ground truth objects.

Formula (Theorem 2):
    ψ_i = H(θ_i) - H_θ(θ_i) · Λ(x_i)⁻¹ · ℓ_θ(y_i, t_i, θ_i)

Oracle Assembly:
    Using true θ*(x) and Oracle formulas for H, H_θ, Λ, ℓ_θ

Package Assembly:
    Using deep_inference.engine.assembler.compute_psi()

Criteria:
    - Correlation(ψ̂, ψ*) > 0.9
    - Mean(ψ̂) ≈ Mean(ψ*) (bias < 0.1)
"""

import sys
import numpy as np
import torch
from torch import Tensor
from scipy.special import expit
from scipy.stats import pearsonr

sys.path.insert(0, "/Users/pranjal/deepest/src")

from evals.dgp import (
    generate_canonical_dgp,
    CanonicalDGP,
    oracle_score,
    oracle_hessian,
    oracle_target_jacobian,
    oracle_lambda_conditional,
)


def oracle_h(theta: np.ndarray, t_tilde: float = 0.0) -> float:
    """
    Oracle target value H(θ).

    H(θ) = σ(α + β·t̃)·(1 - σ(α + β·t̃))·β
    """
    alpha, beta = theta[0], theta[1]
    s = expit(alpha + beta * t_tilde)
    return s * (1 - s) * beta


def oracle_psi_single(
    y: float,
    t: float,
    x: float,
    theta: np.ndarray,
    dgp: CanonicalDGP,
    t_tilde: float = 0.0,
    n_mc_lambda: int = 5000,
) -> float:
    """
    Compute Oracle ψ for a single observation.

    ψ = H(θ) - H_θ(θ) · Λ(x)⁻¹ · ℓ_θ(y, t, θ)
    """
    # H(θ)
    H = oracle_h(theta, t_tilde)

    # H_θ(θ): (2,)
    H_theta = oracle_target_jacobian(theta, t_tilde)

    # Λ(x): (2, 2)
    Lambda = oracle_lambda_conditional(x, dgp, n_samples=n_mc_lambda)

    # ℓ_θ(y, t, θ): (2,)
    score = oracle_score(y, t, theta)

    # Λ⁻¹ · ℓ_θ
    Lambda_inv_score = np.linalg.solve(Lambda, score)

    # ψ = H - H_θ · Λ⁻¹ · ℓ_θ
    correction = H_theta @ Lambda_inv_score
    psi = H - correction

    return psi


def compute_oracle_psi_batch(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    theta: np.ndarray,
    dgp: CanonicalDGP,
    t_tilde: float = 0.0,
    n_mc_lambda: int = 1000,
) -> np.ndarray:
    """
    Compute Oracle ψ for all observations.

    Uses mean Lambda (aggregate) for efficiency.
    """
    n = len(Y)

    # Pre-compute mean Lambda (aggregate approach for efficiency)
    # Note: This is what EstimateLambda(aggregate) does
    Lambdas = []
    for i in range(min(n, 200)):  # Sample for mean Lambda
        x = X[i, 0] if X.ndim == 2 else X[i]
        Lambdas.append(oracle_lambda_conditional(x, dgp, n_samples=n_mc_lambda))
    Lambda_mean = np.mean(Lambdas, axis=0)
    Lambda_inv = np.linalg.inv(Lambda_mean)

    psi_oracle = np.zeros(n)
    for i in range(n):
        # H(θ)
        H = oracle_h(theta[i], t_tilde)

        # H_θ(θ)
        H_theta = oracle_target_jacobian(theta[i], t_tilde)

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
    t_tilde: float = 0.0,
) -> np.ndarray:
    """
    Compute ψ using the package's assembler.
    """
    from deep_inference.models import Logit
    from deep_inference.targets import AME
    from deep_inference.lambda_.estimate import EstimateLambda
    from deep_inference.engine.assembler import compute_psi

    model = Logit()
    target = AME(param_index=1, model_type="logit")

    # Use aggregate Lambda (same as Oracle)
    lambda_strategy = EstimateLambda(method="aggregate")
    lambda_strategy.fit(X, T, Y, theta, model)

    t_tilde_t = torch.tensor(t_tilde, dtype=torch.float32)
    lambda_matrices = lambda_strategy.predict(X, theta)

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

    return psi.numpy()


def compute_psi_metrics(psi_package: np.ndarray, psi_oracle: np.ndarray):
    """
    Compute ψ assembly metrics.
    """
    # Correlation
    corr, p_value = pearsonr(psi_package, psi_oracle)

    # Bias
    bias = psi_package.mean() - psi_oracle.mean()

    # RMSE
    rmse = np.sqrt(np.mean((psi_package - psi_oracle) ** 2))

    return {
        "correlation": corr,
        "p_value": p_value,
        "bias": bias,
        "rmse": rmse,
        "mean_package": psi_package.mean(),
        "mean_oracle": psi_oracle.mean(),
        "std_package": psi_package.std(),
        "std_oracle": psi_oracle.std(),
    }


def run_eval_05(n=1000, seed=42, verbose=True):
    """
    Run ψ assembly evaluation.
    """
    print("=" * 60)
    print("EVAL 05: INFLUENCE FUNCTION ASSEMBLY")
    print("=" * 60)

    print("\nFormula (Theorem 2):")
    print("  ψ_i = H(θ_i) - H_θ(θ_i) · Λ(x_i)⁻¹ · ℓ_θ(y_i, t_i, θ_i)")

    dgp = CanonicalDGP()
    t_tilde = 0.0

    # Generate data
    print(f"\nGenerating data (n={n}, seed={seed})...")
    Y, T, X, theta_true, mu_true = generate_canonical_dgp(n=n, seed=seed, dgp=dgp)
    print(f"  True μ* = {mu_true:.6f}")

    # Compute Oracle ψ (using true theta)
    print(f"\nComputing Oracle ψ...")
    psi_oracle = compute_oracle_psi_batch(
        Y.numpy(), T.numpy(), X.numpy(), theta_true.numpy(), dgp, t_tilde
    )
    print(f"  Mean(ψ_oracle) = {psi_oracle.mean():.6f}")
    print(f"  Std(ψ_oracle) = {psi_oracle.std():.6f}")

    # Compute Package ψ (using true theta for clean comparison)
    print(f"\nComputing Package ψ...")
    psi_package = compute_package_psi(Y, T, X, theta_true, t_tilde)
    print(f"  Mean(ψ_package) = {psi_package.mean():.6f}")
    print(f"  Std(ψ_package) = {psi_package.std():.6f}")

    # Compute metrics
    metrics = compute_psi_metrics(psi_package, psi_oracle)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n--- ψ Assembly Comparison ---")
    print(f"  {'Metric':<20} {'Package':<15} {'Oracle':<15}")
    print("-" * 50)
    print(f"  {'Mean':<20} {metrics['mean_package']:<15.6f} {metrics['mean_oracle']:<15.6f}")
    print(f"  {'Std':<20} {metrics['std_package']:<15.6f} {metrics['std_oracle']:<15.6f}")

    print(f"\n--- Assembly Quality ---")
    print(f"  Correlation(ψ̂, ψ*): {metrics['correlation']:.4f}")
    print(f"  Bias (ψ̂ - ψ*): {metrics['bias']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")

    print(f"\n--- Inference Check ---")
    print(f"  True μ*: {mu_true:.6f}")
    print(f"  Mean(ψ_oracle): {psi_oracle.mean():.6f}")
    print(f"  Mean(ψ_package): {psi_package.mean():.6f}")
    print(f"  Oracle bias from true: {psi_oracle.mean() - mu_true:.6f}")
    print(f"  Package bias from true: {psi_package.mean() - mu_true:.6f}")

    # Sample ψ values
    print(f"\n--- Sample ψ Values (first 10) ---")
    print(f"  {'i':<5} {'ψ_oracle':<15} {'ψ_package':<15} {'diff':<15}")
    print("-" * 50)
    for i in range(10):
        diff = psi_package[i] - psi_oracle[i]
        print(f"  {i:<5} {psi_oracle[i]:<15.6f} {psi_package[i]:<15.6f} {diff:<15.6f}")

    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION CRITERIA")
    print("=" * 60)

    criteria = {
        "Corr(ψ̂, ψ*) > 0.9": metrics["correlation"] > 0.9,
        "|Bias| < 0.1": abs(metrics["bias"]) < 0.1,
        "RMSE < 0.5": metrics["rmse"] < 0.5,
    }

    all_pass = True
    for name, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("EVAL 05: PASS")
    else:
        print("EVAL 05: FAIL")
    print("=" * 60)

    return {
        "metrics": metrics,
        "psi_package": psi_package,
        "psi_oracle": psi_oracle,
        "passed": all_pass,
    }


if __name__ == "__main__":
    result = run_eval_05(n=1000, seed=42)
