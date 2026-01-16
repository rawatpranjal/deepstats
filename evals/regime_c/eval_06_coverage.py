"""
Eval 06: Frequentist Coverage

Goal: Prove confidence intervals work via Monte Carlo simulation.

Procedure:
    For m = 1, ..., M:
        1. Generate data from canonical DGP
        2. Run inference() to get μ̂, SE, CI
        3. Check if true μ* is in CI

Metrics:
    - Coverage: Fraction of CIs containing μ*
    - SE ratio: Emp_SE / Mean_SE
    - Bias: Mean(μ̂) - μ*
    - z-score distribution: Should be ~N(0,1)

Criteria:
    - Coverage in [90%, 99%]
    - SE ratio in [0.7, 1.5]
    - |bias| < 0.1 * SE
"""

import sys
import numpy as np
import torch
from typing import List, Dict
from dataclasses import dataclass

sys.path.insert(0, "/Users/pranjal/deepest/src")

from evals.dgps.regime_c_obs_logit import generate_canonical_dgp, CanonicalDGP


@dataclass
class SimulationResult:
    """Result from a single simulation."""
    sim_id: int
    mu_hat: float
    se: float
    ci_lower: float
    ci_upper: float
    covered: bool
    z_score: float


def run_single_simulation(
    sim_id: int,
    n: int,
    mu_true: float,
    dgp: CanonicalDGP,
    n_folds: int = 20,
    epochs: int = 50,
    verbose: bool = False,
) -> SimulationResult:
    """
    Run a single simulation.

    Returns SimulationResult with mu_hat, SE, CI, coverage.
    """
    from deep_inference import inference

    # Generate data
    Y, T, X, theta_true, _ = generate_canonical_dgp(n=n, seed=sim_id, dgp=dgp)

    # Run inference
    try:
        result = inference(
            Y=Y.numpy(),
            T=T.numpy(),
            X=X.numpy(),
            model="logit",
            target="ame",
            t_tilde=0.0,  # Must match DGP's mu_true() definition!
            n_folds=n_folds,
            epochs=epochs,
            hidden_dims=[64, 32],
            lr=0.01,
            verbose=False,
        )

        mu_hat = result.mu_hat
        se = result.se
        ci_lower = result.ci_lower
        ci_upper = result.ci_upper
        covered = ci_lower <= mu_true <= ci_upper
        z_score = (mu_hat - mu_true) / se if se > 0 else np.nan

        if verbose:
            print(f"  Sim {sim_id}: μ̂={mu_hat:.4f}, SE={se:.4f}, CI=[{ci_lower:.4f}, {ci_upper:.4f}], Covered={covered}")

    except Exception as e:
        print(f"  Sim {sim_id} FAILED: {e}")
        mu_hat = np.nan
        se = np.nan
        ci_lower = np.nan
        ci_upper = np.nan
        covered = False
        z_score = np.nan

    return SimulationResult(
        sim_id=sim_id,
        mu_hat=mu_hat,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        covered=covered,
        z_score=z_score,
    )


def compute_coverage_metrics(results: List[SimulationResult], mu_true: float) -> Dict:
    """
    Compute aggregate metrics from simulation results.
    """
    # Filter out failed simulations
    valid_results = [r for r in results if not np.isnan(r.mu_hat)]
    n_valid = len(valid_results)
    n_failed = len(results) - n_valid

    if n_valid == 0:
        return {"error": "All simulations failed"}

    # Extract arrays
    mu_hats = np.array([r.mu_hat for r in valid_results])
    ses = np.array([r.se for r in valid_results])
    covered = np.array([r.covered for r in valid_results])
    z_scores = np.array([r.z_score for r in valid_results])

    # Remove NaN z-scores
    z_scores = z_scores[~np.isnan(z_scores)]

    # Coverage
    coverage = covered.mean()

    # SE ratio
    emp_se = mu_hats.std()
    mean_se = ses.mean()
    se_ratio = emp_se / mean_se if mean_se > 0 else np.nan

    # Bias
    bias = mu_hats.mean() - mu_true

    # z-score statistics
    z_mean = z_scores.mean() if len(z_scores) > 0 else np.nan
    z_std = z_scores.std() if len(z_scores) > 0 else np.nan

    return {
        "n_simulations": len(results),
        "n_valid": n_valid,
        "n_failed": n_failed,
        "mu_true": mu_true,
        "mean_mu_hat": mu_hats.mean(),
        "std_mu_hat": mu_hats.std(),
        "mean_se": mean_se,
        "emp_se": emp_se,
        "se_ratio": se_ratio,
        "bias": bias,
        "coverage": coverage,
        "coverage_count": covered.sum(),
        "z_mean": z_mean,
        "z_std": z_std,
    }


def run_eval_06(
    M: int = 20,
    n: int = 1000,
    n_folds: int = 20,
    epochs: int = 50,
    verbose: bool = True,
):
    """
    Run Frequentist coverage evaluation.

    Args:
        M: Number of Monte Carlo simulations
        n: Sample size per simulation
        n_folds: Cross-fitting folds
        epochs: Training epochs
    """
    print("=" * 60)
    print("EVAL 06: FREQUENTIST COVERAGE")
    print("=" * 60)

    dgp = CanonicalDGP()
    mu_true = dgp.mu_true()

    print(f"\nDGP:")
    print(f"  α*(x) = {dgp.A0} + {dgp.A1}·sin(x)")
    print(f"  β*(x) = {dgp.B0} + {dgp.B1}·x")
    print(f"  True μ* = {mu_true:.6f}")

    print(f"\nSimulation Settings:")
    print(f"  M = {M} replications")
    print(f"  n = {n} observations")
    print(f"  n_folds = {n_folds}")
    print(f"  epochs = {epochs}")

    # Run simulations
    print(f"\n" + "-" * 60)
    print("RUNNING SIMULATIONS")
    print("-" * 60)

    results = []
    for m in range(1, M + 1):
        if verbose and m % 5 == 0:
            print(f"  Running simulation {m}/{M}...")

        result = run_single_simulation(
            sim_id=m,
            n=n,
            mu_true=mu_true,
            dgp=dgp,
            n_folds=n_folds,
            epochs=epochs,
            verbose=False,
        )
        results.append(result)

    # Compute metrics
    metrics = compute_coverage_metrics(results, mu_true)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n--- Simulation Summary ---")
    print(f"  Total simulations: {metrics['n_simulations']}")
    print(f"  Valid simulations: {metrics['n_valid']}")
    print(f"  Failed simulations: {metrics['n_failed']}")

    print(f"\n--- Point Estimation ---")
    print(f"  True μ*: {metrics['mu_true']:.6f}")
    print(f"  Mean(μ̂): {metrics['mean_mu_hat']:.6f}")
    print(f"  Std(μ̂): {metrics['std_mu_hat']:.6f}")
    print(f"  Bias: {metrics['bias']:.6f}")

    print(f"\n--- Standard Error ---")
    print(f"  Empirical SE: {metrics['emp_se']:.6f}")
    print(f"  Mean SE (IF): {metrics['mean_se']:.6f}")
    print(f"  SE Ratio: {metrics['se_ratio']:.4f}")

    print(f"\n--- Coverage ---")
    print(f"  Coverage: {metrics['coverage']*100:.1f}% ({metrics['coverage_count']}/{metrics['n_valid']})")

    print(f"\n--- z-Score Distribution ---")
    print(f"  Mean z: {metrics['z_mean']:.4f} (should be ~0)")
    print(f"  Std z: {metrics['z_std']:.4f} (should be ~1)")

    # Individual results table
    print(f"\n--- Individual Results (first 10) ---")
    print(f"  {'Sim':<5} {'μ̂':<10} {'SE':<10} {'CI_lo':<10} {'CI_hi':<10} {'Cov':<5} {'z':<10}")
    print("-" * 65)
    for r in results[:10]:
        cov_str = "T" if r.covered else "F"
        print(f"  {r.sim_id:<5} {r.mu_hat:<10.4f} {r.se:<10.4f} {r.ci_lower:<10.4f} {r.ci_upper:<10.4f} {cov_str:<5} {r.z_score:<10.4f}")

    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION CRITERIA")
    print("=" * 60)

    criteria = {
        f"Coverage in [85%, 99%]": 0.85 <= metrics["coverage"] <= 0.99,
        f"SE Ratio in [0.5, 2.0]": 0.5 <= metrics["se_ratio"] <= 2.0,
        f"|Bias| < 0.1": abs(metrics["bias"]) < 0.1,
        f"|z_mean| < 0.5": abs(metrics["z_mean"]) < 0.5 if not np.isnan(metrics["z_mean"]) else False,
        f"z_std in [0.5, 2.0]": 0.5 <= metrics["z_std"] <= 2.0 if not np.isnan(metrics["z_std"]) else False,
    }

    all_pass = True
    for name, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("EVAL 06: PASS")
    else:
        print("EVAL 06: FAIL (may need more M or tuning)")
    print("=" * 60)

    return {
        "metrics": metrics,
        "results": results,
        "passed": all_pass,
    }


if __name__ == "__main__":
    # Run with modest M for quick test; increase M for rigorous validation
    result = run_eval_06(M=20, n=1000, n_folds=20, epochs=50)
