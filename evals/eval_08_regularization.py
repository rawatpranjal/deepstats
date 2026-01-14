"""
Eval 08: Regularization Diagnostics

============================================================
WHAT THIS EVAL TESTS
============================================================

This eval answers two critical questions about inference validity:

1. IS CROSS-FITTING NECESSARY?

   Cross-fitting (sample splitting) prevents overfitting bias in the influence
   function. Without it, we train θ̂(x) on all data and then evaluate ψ on the
   same data, causing the model to "memorize" rather than generalize.

   Expected result: No-split should have INVALID coverage (either too high
   or too low), while cross-fitting should achieve ~95% coverage.

2. HOW SHOULD RIDGE REGULARIZATION SCALE WITH n?

   When inverting Lambda matrices (Λ⁻¹), we add ridge regularization for
   numerical stability: Λ_reg = Λ + ridge × I

   Problem: Fixed ridge causes SE ratio miscalibration across sample sizes.
   - At small n: Under-regularization → unstable Λ⁻¹ → SE ratio < 1
   - At large n: Over-regularization → biased Λ⁻¹ → SE ratio > 1

   Hypothesis: Adaptive ridge = c/√n should fix this by scaling with sample size.

============================================================
WHY THIS MATTERS
============================================================

- Cross-fitting is a core assumption of the FLM (2021, 2025) theory
- SE ratio ≠ 1 means confidence intervals are miscalibrated
- SE ratio < 1 → CIs too narrow → under-coverage (false confidence)
- SE ratio > 1 → CIs too wide → over-coverage (conservative but inefficient)

============================================================
STRUCTURE
============================================================

Part A: Cross-Fitting Necessity
    A1: No-split (n_folds=1) vs cross-fit (K=5) - coverage comparison
    A2: Minimum folds needed (K=2, 5, 10, 20)

Part B: Ridge Calibration
    B1: Fixed ridge=1e-6 across n = 500, 1000, 2000, 5000, 10000
    B2: Adaptive ridge = c/sqrt(n) with c=1e-4
    B3: Grid search c = 1e-6, 1e-5, 1e-4, 1e-3 at n=5000

============================================================
PASS CRITERIA
============================================================

    A1: Cross-fit coverage in [90%, 98%], no-split outside this range
    A2: K >= 5 sufficient for valid coverage
    B1: SE ratio U-shape confirmed (not all in [0.9, 1.1])
    B2: Adaptive ridge gives SE ratio in [0.9, 1.1] for all n
    B3: Optimal c identified

============================================================
KEY DELIVERABLE
============================================================

This eval produces a concrete recommendation:

    # Before (current)
    ridge = 1e-4  # Fixed

    # After (recommended)
    ridge = OPTIMAL_C / np.sqrt(n)  # Adaptive

Where OPTIMAL_C is determined empirically by Part B3.

Uses: Canonical DGP from dgp.py (Regime C: Confounded Logit)
"""

import sys
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm

sys.path.insert(0, "/Users/pranjal/deepest/src")

from evals.dgp import generate_canonical_dgp, CanonicalDGP


# =============================================================================
# Data Structures
# =============================================================================


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


@dataclass
class ConfigResult:
    """Aggregated results for a single configuration."""

    config_name: str
    n: int
    n_folds: int
    ridge: float
    n_simulations: int
    n_valid: int
    coverage: float
    mean_se: float
    emp_se: float
    se_ratio: float
    mean_bias: float


# =============================================================================
# Helper Functions
# =============================================================================


def run_nosplit_simulation(
    sim_id: int,
    n: int,
    mu_true: float,
    dgp: CanonicalDGP,
    ridge: float = 1e-4,
    epochs: int = 50,
    lambda_method: str = "lgbm",
    verbose: bool = False,
) -> SimulationResult:
    """
    Run a NO-SPLIT simulation: train and evaluate on the SAME data.

    This bypasses cross-fitting to demonstrate overfitting bias.
    The model "memorizes" the training data, leading to invalid inference.
    """
    from deep_inference.models import Logit, StructuralNet, train_structural_net
    from deep_inference.targets import AME
    from deep_inference.lambda_ import EstimateLambda
    from deep_inference.engine.assembler import compute_psi
    from deep_inference.engine.variance import compute_inference_results

    # Generate data
    Y, T, X, theta_true, _ = generate_canonical_dgp(n=n, seed=sim_id, dgp=dgp)

    try:
        model = Logit()
        target = AME(param_index=1, model_type="logit")
        t_tilde = torch.tensor(0.0)

        # Train theta network on ALL data
        theta_net = StructuralNet(
            input_dim=X.shape[1],
            theta_dim=model.theta_dim,
            hidden_dims=[64, 32],
        )

        def loss_fn_batched(y, t, theta):
            losses = torch.zeros(len(y))
            for i in range(len(y)):
                losses[i] = model.loss(y[i], t[i], theta[i])
            return losses

        history = train_structural_net(
            model=theta_net,
            X=X,
            T=T,
            Y=Y,
            loss_fn=loss_fn_batched,
            epochs=epochs,
            lr=0.01,
            verbose=False,
        )

        # Get theta predictions on SAME data (no held-out)
        with torch.no_grad():
            theta_hat = theta_net(X)

        # Fit Lambda on SAME data
        lambda_strategy = EstimateLambda(method=lambda_method)
        lambda_strategy.fit(X=X, T=T, Y=Y, theta_hat=theta_hat, model=model)
        lambda_matrices = lambda_strategy.predict(X, theta_hat)

        # Compute psi on SAME data (this is the overfitting!)
        psi = compute_psi(
            Y=Y,
            T=T,
            X=X,
            theta_hat=theta_hat,
            t_tilde=t_tilde,
            lambda_matrices=lambda_matrices,
            model=model,
            target=target,
            ridge=ridge,
        )

        # Compute inference
        results = compute_inference_results(psi)
        mu_hat = results["mu_hat"]
        se = results["se"]
        ci_lower = results["ci_lower"]
        ci_upper = results["ci_upper"]
        covered = ci_lower <= mu_true <= ci_upper
        z_score = (mu_hat - mu_true) / se if se > 0 else np.nan

    except Exception as e:
        if verbose:
            print(f"  Sim {sim_id} FAILED: {e}")
        return SimulationResult(sim_id, np.nan, np.nan, np.nan, np.nan, False, np.nan)

    return SimulationResult(sim_id, mu_hat, se, ci_lower, ci_upper, covered, z_score)


def run_single_simulation(
    sim_id: int,
    n: int,
    mu_true: float,
    dgp: CanonicalDGP,
    n_folds: int = 20,
    ridge: float = 1e-4,
    epochs: int = 50,
    lambda_method: str = "lgbm",
    verbose: bool = False,
) -> SimulationResult:
    """
    Run a single simulation with specified n_folds and ridge.

    Key difference from eval_06: parameterizes n_folds and ridge.
    """
    from deep_inference import inference

    # Generate data
    Y, T, X, theta_true, _ = generate_canonical_dgp(n=n, seed=sim_id, dgp=dgp)

    try:
        result = inference(
            Y=Y.numpy(),
            T=T.numpy(),
            X=X.numpy(),
            model="logit",
            target="ame",
            t_tilde=0.0,
            lambda_method=lambda_method,
            n_folds=n_folds,
            epochs=epochs,
            ridge=ridge,
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

    except Exception as e:
        if verbose:
            print(f"  Sim {sim_id} FAILED: {e}")
        return SimulationResult(sim_id, np.nan, np.nan, np.nan, np.nan, False, np.nan)

    return SimulationResult(sim_id, mu_hat, se, ci_lower, ci_upper, covered, z_score)


def compute_metrics(
    results: List[SimulationResult],
    mu_true: float,
    config_name: str,
    n: int,
    n_folds: int,
    ridge: float,
) -> ConfigResult:
    """Compute aggregate metrics from simulation results."""
    valid = [r for r in results if not np.isnan(r.mu_hat)]
    n_valid = len(valid)

    if n_valid == 0:
        return ConfigResult(
            config_name,
            n,
            n_folds,
            ridge,
            len(results),
            0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    mu_hats = np.array([r.mu_hat for r in valid])
    ses = np.array([r.se for r in valid])
    covered = np.array([r.covered for r in valid])

    coverage = covered.mean()
    mean_se = ses.mean()
    emp_se = mu_hats.std()
    se_ratio = emp_se / mean_se if mean_se > 0 else np.nan
    mean_bias = mu_hats.mean() - mu_true

    return ConfigResult(
        config_name=config_name,
        n=n,
        n_folds=n_folds,
        ridge=ridge,
        n_simulations=len(results),
        n_valid=n_valid,
        coverage=coverage,
        mean_se=mean_se,
        emp_se=emp_se,
        se_ratio=se_ratio,
        mean_bias=mean_bias,
    )


# =============================================================================
# Part A: Cross-Fitting Necessity
# =============================================================================


def run_part_a1(
    M: int = 30,
    n: int = 2000,
    epochs: int = 100,
    verbose: bool = True,
) -> dict:
    """
    A1: Compare no-split vs cross-fit (K=5).

    No-split: Train θ̂ on ALL data, evaluate ψ on SAME data (overfitting!)
    Cross-fit: Train θ̂ on K-1 folds, evaluate ψ on held-out fold

    Hypothesis: No-split will have invalid coverage due to overfitting bias.
    """
    dgp = CanonicalDGP()
    mu_true = dgp.mu_true()

    results = {}

    # Run NO-SPLIT simulations (train and evaluate on same data)
    nosplit_results = []
    for m in tqdm(range(1, M + 1), desc="no_split", ncols=80):
        r = run_nosplit_simulation(
            sim_id=m,
            n=n,
            mu_true=mu_true,
            dgp=dgp,
            ridge=1e-4,
            epochs=epochs,
        )
        nosplit_results.append(r)
    results["no_split"] = compute_metrics(nosplit_results, mu_true, "no_split", n, 1, 1e-4)

    # Run CROSS-FIT simulations (proper held-out evaluation)
    crossfit_results = []
    for m in tqdm(range(1, M + 1), desc="crossfit_5", ncols=80):
        r = run_single_simulation(
            sim_id=m,
            n=n,
            mu_true=mu_true,
            dgp=dgp,
            n_folds=5,
            ridge=1e-4,
            epochs=epochs,
        )
        crossfit_results.append(r)
    results["crossfit_5"] = compute_metrics(crossfit_results, mu_true, "crossfit_5", n, 5, 1e-4)

    # Pass criteria
    crossfit_cov = results["crossfit_5"].coverage
    nosplit_cov = results["no_split"].coverage

    # Cross-fit should be in [90%, 98%], no-split should be outside this range
    pass_a1 = (0.90 <= crossfit_cov <= 0.98) and (
        nosplit_cov < 0.90 or nosplit_cov > 0.98
    )

    return {
        "results": results,
        "crossfit_coverage": crossfit_cov,
        "nosplit_coverage": nosplit_cov,
        "passed": pass_a1,
    }


def run_part_a2(
    M: int = 30,
    n: int = 2000,
    epochs: int = 100,
    verbose: bool = True,
) -> dict:
    """
    A2: Test K = 2, 5, 10, 20 to find minimum required folds.

    Hypothesis: K >= 5 should be sufficient for valid coverage.
    """
    dgp = CanonicalDGP()
    mu_true = dgp.mu_true()

    k_values = [2, 5, 10, 20]
    results = {}

    for k in k_values:
        sim_results = []
        for m in tqdm(range(1, M + 1), desc=f"K={k}", ncols=80):
            r = run_single_simulation(
                sim_id=m,
                n=n,
                mu_true=mu_true,
                dgp=dgp,
                n_folds=k,
                ridge=1e-4,
                epochs=epochs,
            )
            sim_results.append(r)

        metrics = compute_metrics(sim_results, mu_true, f"K={k}", n, k, 1e-4)
        results[k] = metrics

    # Find minimum K with valid coverage [90%, 98%]
    min_k_valid = None
    for k in k_values:
        cov = results[k].coverage
        if 0.90 <= cov <= 0.98:
            min_k_valid = k
            break

    pass_a2 = min_k_valid is not None and min_k_valid <= 5

    return {
        "results": results,
        "min_k_valid": min_k_valid,
        "passed": pass_a2,
    }


# =============================================================================
# Part B: Ridge Calibration
# =============================================================================


def run_part_b1(
    M: int = 20,
    ridge: float = 1e-6,
    n_folds: int = 10,
    epochs: int = 50,
    verbose: bool = True,
) -> dict:
    """
    B1: Fixed ridge=1e-6 across n = 500, 1000, 2000, 5000, 10000.

    Hypothesis: SE ratio will show U-shape (miscalibration at extremes).
    """
    dgp = CanonicalDGP()
    mu_true = dgp.mu_true()

    n_values = [500, 1000, 2000, 5000, 10000]
    results = {}

    for n in n_values:
        sim_results = []
        for m in tqdm(range(1, M + 1), desc=f"n={n}", ncols=80):
            r = run_single_simulation(
                sim_id=m,
                n=n,
                mu_true=mu_true,
                dgp=dgp,
                n_folds=n_folds,
                ridge=ridge,
                epochs=epochs,
            )
            sim_results.append(r)

        metrics = compute_metrics(sim_results, mu_true, f"n={n}", n, n_folds, ridge)
        results[n] = metrics

    # Check for U-shape in SE ratio (not all values in [0.9, 1.1])
    se_ratios = [results[n].se_ratio for n in n_values]
    valid_ratios = [r for r in se_ratios if not np.isnan(r)]
    pass_b1 = not all(0.9 <= r <= 1.1 for r in valid_ratios)

    return {
        "results": results,
        "se_ratios": dict(zip(n_values, se_ratios)),
        "passed": pass_b1,
    }


def run_part_b2(
    M: int = 20,
    c: float = 1e-4,
    n_folds: int = 10,
    epochs: int = 50,
    verbose: bool = True,
) -> dict:
    """
    B2: Adaptive ridge = c / sqrt(n) with c=1e-4.

    Hypothesis: This formula should give SE ratio in [0.9, 1.1] for all n.
    """
    dgp = CanonicalDGP()
    mu_true = dgp.mu_true()

    n_values = [500, 1000, 2000, 5000, 10000]
    results = {}

    for n in n_values:
        adaptive_ridge = c / np.sqrt(n)

        sim_results = []
        for m in tqdm(range(1, M + 1), desc=f"n={n}", ncols=80):
            r = run_single_simulation(
                sim_id=m,
                n=n,
                mu_true=mu_true,
                dgp=dgp,
                n_folds=n_folds,
                ridge=adaptive_ridge,
                epochs=epochs,
            )
            sim_results.append(r)

        metrics = compute_metrics(
            sim_results, mu_true, f"n={n}", n, n_folds, adaptive_ridge
        )
        results[n] = metrics

    # Check all SE ratios in [0.9, 1.1]
    se_ratios = [results[n].se_ratio for n in n_values]
    valid_ratios = [r for r in se_ratios if not np.isnan(r)]
    pass_b2 = all(0.9 <= r <= 1.1 for r in valid_ratios) if valid_ratios else False

    return {
        "results": results,
        "c": c,
        "se_ratios": dict(zip(n_values, se_ratios)),
        "passed": pass_b2,
    }


def run_part_b3(
    M: int = 20,
    n: int = 5000,
    n_folds: int = 10,
    epochs: int = 50,
    verbose: bool = True,
) -> dict:
    """
    B3: Grid search c = 1e-6, 1e-5, 1e-4, 1e-3 at n=5000.

    Goal: Find optimal c that gives SE ratio closest to 1.0.
    """
    dgp = CanonicalDGP()
    mu_true = dgp.mu_true()

    c_values = [1e-6, 1e-5, 1e-4, 1e-3]
    results = {}

    for c in c_values:
        ridge = c / np.sqrt(n)

        sim_results = []
        for m in tqdm(range(1, M + 1), desc=f"c={c:.0e}", ncols=80):
            r = run_single_simulation(
                sim_id=m,
                n=n,
                mu_true=mu_true,
                dgp=dgp,
                n_folds=n_folds,
                ridge=ridge,
                epochs=epochs,
            )
            sim_results.append(r)

        metrics = compute_metrics(sim_results, mu_true, f"c={c:.0e}", n, n_folds, ridge)
        results[c] = metrics

    # Find optimal c (SE ratio closest to 1.0)
    best_c = None
    best_diff = float("inf")
    for c in c_values:
        ratio = results[c].se_ratio
        if not np.isnan(ratio):
            diff = abs(ratio - 1.0)
            if diff < best_diff:
                best_diff = diff
                best_c = c

    return {
        "results": results,
        "optimal_c": best_c,
        "optimal_se_ratio": results[best_c].se_ratio if best_c else None,
        "optimal_bias": results[best_c].mean_bias if best_c else None,
        "passed": best_c is not None,
    }


# =============================================================================
# Print Functions
# =============================================================================


def print_a1_results(result: dict):
    """Print Part A1 results table."""
    print(f"\n{'Config':<15} {'Coverage':>10} {'SE Ratio':>10} {'Bias':>10}")
    print("-" * 50)
    for name, metrics in result["results"].items():
        print(
            f"{name:<15} {metrics.coverage*100:>9.1f}% {metrics.se_ratio:>10.3f} {metrics.mean_bias:>10.4f}"
        )

    status = "PASS" if result["passed"] else "FAIL"
    print(f"\nA1: {status}")
    print(f"  Cross-fit coverage: {result['crossfit_coverage']*100:.1f}%")
    print(f"  No-split coverage: {result['nosplit_coverage']*100:.1f}%")


def print_a2_results(result: dict):
    """Print Part A2 results table."""
    print(f"\n{'K':<10} {'Coverage':>10} {'SE Ratio':>10} {'Bias':>10}")
    print("-" * 45)
    for k, metrics in result["results"].items():
        print(
            f"{k:<10} {metrics.coverage*100:>9.1f}% {metrics.se_ratio:>10.3f} {metrics.mean_bias:>10.4f}"
        )

    status = "PASS" if result["passed"] else "FAIL"
    print(f"\nA2: {status}")
    print(f"  Minimum K for valid coverage: {result['min_k_valid']}")


def print_b1_results(result: dict):
    """Print Part B1 results table."""
    print(
        f"\n{'n':<10} {'Coverage':>10} {'SE Ratio':>10} {'Mean SE':>10} {'Emp SE':>10}"
    )
    print("-" * 55)
    for n, metrics in result["results"].items():
        print(
            f"{n:<10} {metrics.coverage*100:>9.1f}% {metrics.se_ratio:>10.3f} "
            f"{metrics.mean_se:>10.4f} {metrics.emp_se:>10.4f}"
        )

    status = "PASS" if result["passed"] else "FAIL"
    u_shape = "Yes" if result["passed"] else "No"
    print(f"\nB1: {status} (U-shape confirmed: {u_shape})")


def print_b2_results(result: dict):
    """Print Part B2 results table."""
    print(f"\nAdaptive ridge = {result['c']:.0e} / sqrt(n)")
    print(f"\n{'n':<10} {'Ridge':>12} {'Coverage':>10} {'SE Ratio':>10}")
    print("-" * 47)
    for n, metrics in result["results"].items():
        print(
            f"{n:<10} {metrics.ridge:>12.2e} {metrics.coverage*100:>9.1f}% {metrics.se_ratio:>10.3f}"
        )

    status = "PASS" if result["passed"] else "FAIL"
    print(f"\nB2: {status}")


def print_b3_results(result: dict):
    """Print Part B3 results table."""
    print(f"\n{'c':<12} {'Ridge':>12} {'Coverage':>10} {'SE Ratio':>10} {'Bias':>10}")
    print("-" * 60)
    for c, metrics in result["results"].items():
        marker = " <-- BEST" if c == result["optimal_c"] else ""
        print(
            f"{c:<12.0e} {metrics.ridge:>12.2e} {metrics.coverage*100:>9.1f}% "
            f"{metrics.se_ratio:>10.3f} {metrics.mean_bias:>10.4f}{marker}"
        )

    status = "PASS" if result["passed"] else "FAIL"
    print(f"\nB3: {status}")
    if result["optimal_c"]:
        print(f"  Optimal c: {result['optimal_c']:.0e}")
        print(f"  Optimal SE ratio: {result['optimal_se_ratio']:.3f}")


def print_final_summary(all_results: dict):
    """Print final summary table."""
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"\n{'Test':<10} {'Description':<35} {'Status':>10}")
    print("-" * 60)

    tests = [
        ("A1", "Cross-fit vs No-split", all_results["A1"]["passed"]),
        ("A2", "Minimum folds required", all_results["A2"]["passed"]),
        ("B1", "Fixed ridge U-shape confirmed", all_results["B1"]["passed"]),
        ("B2", "Adaptive ridge calibration", all_results["B2"]["passed"]),
        ("B3", "Optimal c identified", all_results["B3"]["passed"]),
    ]

    for name, desc, passed in tests:
        status = "PASS" if passed else "FAIL"
        print(f"{name:<10} {desc:<35} {status:>10}")

    n_pass = sum(1 for _, _, p in tests if p)
    print("-" * 60)
    print(f"{'TOTAL':<10} {'':<35} {n_pass}/{len(tests)}")

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if all_results["A1"]["passed"]:
        print(f"  Cross-fitting: Required (no-split fails)")
    else:
        print(f"  Cross-fitting: Needs investigation")

    if all_results["A2"]["min_k_valid"]:
        print(f"  Minimum folds: K >= {all_results['A2']['min_k_valid']}")

    if all_results["B3"]["optimal_c"]:
        print(
            f"  Ridge formula: ridge = {all_results['B3']['optimal_c']:.0e} / sqrt(n)"
        )

    print("\n" + "=" * 60)
    if n_pass == len(tests):
        print("EVAL 08: ALL PASS")
    else:
        print(f"EVAL 08: {n_pass}/{len(tests)} PASSED")
    print("=" * 60)


# =============================================================================
# Main Orchestrator
# =============================================================================


def run_eval_08(
    M_a: int = 50,
    M_b: int = 50,
    epochs: int = 50,
    verbose: bool = True,
) -> dict:
    """
    Run Eval 08: Cross-Fitting Necessity & Ridge Calibration.

    Args:
        M_a: Number of simulations for Part A (default: 50 for rigorous MC)
        M_b: Number of simulations for Part B (default: 50 for rigorous MC)
        epochs: Training epochs per simulation
        verbose: Print detailed output
    """
    print("=" * 60)
    print("EVAL 08: CROSS-FITTING & RIDGE CALIBRATION")
    print("=" * 60)

    dgp = CanonicalDGP()
    mu_true = dgp.mu_true()
    print(f"\nTrue mu* = {mu_true:.6f}")

    all_results = {}

    # Part A: Cross-Fitting Necessity
    print("\n" + "-" * 60)
    print("PART A: CROSS-FITTING NECESSITY")
    print("-" * 60)

    print("\n--- A1: No-Split vs Cross-Fit ---")
    a1 = run_part_a1(M=M_a, epochs=epochs, verbose=verbose)
    all_results["A1"] = a1
    print_a1_results(a1)

    print("\n--- A2: Minimum Folds Required ---")
    a2 = run_part_a2(M=M_a, epochs=epochs, verbose=verbose)
    all_results["A2"] = a2
    print_a2_results(a2)

    # Part B: Ridge Calibration
    print("\n" + "-" * 60)
    print("PART B: RIDGE CALIBRATION")
    print("-" * 60)

    print("\n--- B1: Fixed Ridge Across Sample Sizes ---")
    b1 = run_part_b1(M=M_b, epochs=epochs, verbose=verbose)
    all_results["B1"] = b1
    print_b1_results(b1)

    print("\n--- B2: Adaptive Ridge Formula ---")
    b2 = run_part_b2(M=M_b, epochs=epochs, verbose=verbose)
    all_results["B2"] = b2
    print_b2_results(b2)

    print("\n--- B3: Grid Search for Optimal c ---")
    b3 = run_part_b3(M=M_b, epochs=epochs, verbose=verbose)
    all_results["B3"] = b3
    print_b3_results(b3)

    # Final Summary
    print_final_summary(all_results)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Eval 08: Cross-Fitting & Ridge")
    parser.add_argument("--quick", action="store_true", help="Quick mode (M=10)")
    parser.add_argument("--M-a", type=int, default=50, help="Simulations for Part A (default: 50)")
    parser.add_argument("--M-b", type=int, default=50, help="Simulations for Part B (default: 50)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    args = parser.parse_args()

    if args.quick:
        M_a, M_b = 10, 10  # Quick mode for testing
    else:
        M_a, M_b = args.M_a, args.M_b  # Full mode: M=50 for rigorous MC

    result = run_eval_08(M_a=M_a, M_b=M_b, epochs=args.epochs)
