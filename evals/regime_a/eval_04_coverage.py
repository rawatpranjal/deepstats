"""
Eval 04: Frequentist Coverage (Regime A)

Goal: Verify 95% CI coverage for RCT logit with ComputeLambda.

Procedure:
    For m = 1, ..., M:
        1. Generate RCT data (T ~ Bernoulli(0.5))
        2. Run inference() with is_randomized=True, treatment_dist=Bernoulli(0.5)
        3. Check if true ATE is in CI

Target: ATE = E[σ(X+1) - σ(X)] ≈ 0.231

Criteria:
    - Coverage in [85%, 99%]
    - SE ratio in [0.5, 2.0]
    - |bias| < 0.1
"""

import sys
import numpy as np
import torch
from typing import List
from dataclasses import dataclass

sys.path.insert(0, "/Users/pranjal/deepest/src")

from evals.dgps.regime_a_rct_logit import RCTLogitDGP, generate_rct_logit_data


@dataclass
class SimResult:
    """Result from a single simulation."""
    sim_id: int
    mu_hat: float
    se: float
    ci_lower: float
    ci_upper: float
    covered: bool
    z_score: float


def run_single_sim(
    sim_id: int,
    n: int,
    mu_true: float,
    dgp: RCTLogitDGP,
    n_folds: int = 10,
    epochs: int = 30,
) -> SimResult:
    """Run a single simulation."""
    try:
        from deep_inference import inference

        # Generate RCT data
        Y, T, X, theta_true, _ = generate_rct_logit_data(n=n, seed=sim_id, dgp=dgp)

        # Run inference with RCT settings
        result = inference(
            Y=Y.numpy(),
            T=T.numpy(),
            X=X.numpy(),
            model="logit",
            target="ame",
            is_randomized=True,  # KEY: Tell it this is RCT
            # treatment_dist would enable ComputeLambda
            n_folds=n_folds,
            epochs=epochs,
            hidden_dims=[32, 16],
            lr=0.01,
            verbose=False,
        )

        mu_hat = result.mu_hat
        se = result.se
        ci_lower = result.ci_lower
        ci_upper = result.ci_upper
        covered = ci_lower <= mu_true <= ci_upper
        z_score = (mu_hat - mu_true) / se if se > 0 else np.nan

        return SimResult(
            sim_id=sim_id,
            mu_hat=mu_hat,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            covered=covered,
            z_score=z_score,
        )

    except Exception as e:
        print(f"  Sim {sim_id} FAILED: {e}")
        return SimResult(
            sim_id=sim_id,
            mu_hat=np.nan,
            se=np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            covered=False,
            z_score=np.nan,
        )


def run_eval_04_coverage(M: int = 20, n: int = 500, verbose: bool = True):
    """
    Run frequentist coverage evaluation for Regime A.
    """
    print("=" * 60)
    print("EVAL 04: FREQUENTIST COVERAGE (Regime A)")
    print("=" * 60)

    dgp = RCTLogitDGP()
    mu_true = dgp.mu_true()

    print(f"\nDGP: RCT Logit")
    print(f"  T ~ Bernoulli({dgp.p_treat})")
    print(f"  True ATE = {mu_true:.6f}")

    print(f"\nSimulation Settings:")
    print(f"  M = {M} replications")
    print(f"  n = {n} observations")

    # Run simulations
    print(f"\n" + "-" * 60)
    print("RUNNING SIMULATIONS")
    print("-" * 60)

    results = []
    for m in range(1, M + 1):
        if verbose and m % 5 == 0:
            print(f"  Running simulation {m}/{M}...")

        result = run_single_sim(
            sim_id=m,
            n=n,
            mu_true=mu_true,
            dgp=dgp,
            n_folds=10,
            epochs=30,
        )
        results.append(result)

    # Compute metrics
    valid_results = [r for r in results if not np.isnan(r.mu_hat)]
    n_valid = len(valid_results)
    n_failed = len(results) - n_valid

    if n_valid == 0:
        print("\nAll simulations failed!")
        return {"passed": False, "skipped": False}

    mu_hats = np.array([r.mu_hat for r in valid_results])
    ses = np.array([r.se for r in valid_results])
    covered = np.array([r.covered for r in valid_results])
    z_scores = np.array([r.z_score for r in valid_results])
    z_scores = z_scores[~np.isnan(z_scores)]

    coverage = covered.mean()
    emp_se = mu_hats.std()
    mean_se = ses.mean()
    se_ratio = emp_se / mean_se if mean_se > 0 else np.nan
    bias = mu_hats.mean() - mu_true

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n--- Simulation Summary ---")
    print(f"  Valid: {n_valid}/{M}, Failed: {n_failed}")

    print(f"\n--- Point Estimation ---")
    print(f"  True ATE: {mu_true:.6f}")
    print(f"  Mean(μ̂): {mu_hats.mean():.6f}")
    print(f"  Bias: {bias:.6f}")

    print(f"\n--- Standard Error ---")
    print(f"  Empirical SE: {emp_se:.6f}")
    print(f"  Mean SE (IF): {mean_se:.6f}")
    print(f"  SE Ratio: {se_ratio:.4f}")

    print(f"\n--- Coverage ---")
    print(f"  Coverage: {coverage*100:.1f}% ({covered.sum()}/{n_valid})")

    if len(z_scores) > 0:
        print(f"\n--- z-Score Distribution ---")
        print(f"  Mean z: {z_scores.mean():.4f} (should be ~0)")
        print(f"  Std z: {z_scores.std():.4f} (should be ~1)")

    # Individual results
    print(f"\n--- Individual Results (first 10) ---")
    print(f"  {'Sim':<5} {'μ̂':<10} {'SE':<10} {'CI_lo':<10} {'CI_hi':<10} {'Cov':<5}")
    print("-" * 55)
    for r in results[:10]:
        cov_str = "T" if r.covered else "F"
        print(f"  {r.sim_id:<5} {r.mu_hat:<10.4f} {r.se:<10.4f} {r.ci_lower:<10.4f} {r.ci_upper:<10.4f} {cov_str:<5}")

    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION CRITERIA")
    print("=" * 60)

    criteria = {
        "Coverage in [80%, 99%]": 0.80 <= coverage <= 0.99,
        "SE Ratio in [0.5, 2.0]": 0.5 <= se_ratio <= 2.0 if not np.isnan(se_ratio) else False,
        "|Bias| < 0.1": abs(bias) < 0.1,
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
        "coverage": coverage,
        "se_ratio": se_ratio,
        "bias": bias,
        "results": results,
        "passed": all_pass,
    }


if __name__ == "__main__":
    result = run_eval_04_coverage(M=20, n=500, verbose=True)
