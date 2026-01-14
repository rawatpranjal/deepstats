"""
Eval 07: End-to-End User Experience

This eval demonstrates the package as a real analyst would use it:
1. Generate realistic heterogeneous data (loan application scenario)
2. Run Bootstrap Oracle (gold standard for θ(x) inference)
3. Run NN inference with structural_dml()
4. Compare results side-by-side
5. Analyze heterogeneous effects
6. Show advanced targets (AME, custom)

This is NOT a Monte Carlo validation - it's a single comprehensive run
showing everything the package can do.
"""

import sys
import numpy as np
import torch
from scipy.special import expit
from scipy.stats import pearsonr
import statsmodels.api as sm
from typing import Dict, Any, Tuple
from tqdm import tqdm

sys.path.insert(0, "/Users/pranjal/deepest/src")

from deep_inference import structural_dml


# ============================================================
# PART A: DATA GENERATION
# ============================================================

def generate_loan_application_data(
    n: int = 3000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate realistic loan application data with heterogeneous effects.

    Scenario:
    - Y: Applied for loan (binary)
    - T: Interest rate offered (continuous)
    - X: Customer features (income, credit_score, age)

    DGP:
    - alpha(x) = intercept + effect of customer quality
    - beta(x) = rate sensitivity (varies by customer)
    - P(apply | X, T) = sigmoid(alpha(X) + beta(X) * T)

    Heterogeneity:
    - High credit score → more rate sensitive (higher |beta|)
    - Higher income → higher baseline application rate
    """
    np.random.seed(seed)

    # Customer features (standardized)
    income = np.random.normal(0, 1, n)  # log income, standardized
    credit_score = np.random.normal(0, 1, n)  # standardized
    age = np.random.normal(0, 1, n)  # standardized

    X = np.column_stack([income, credit_score, age])

    # DGP parameters
    # alpha(x) = a0 + a1*income + a2*credit + a3*age
    a0, a1, a2, a3 = 0.5, 0.3, 0.2, 0.1

    # beta(x) = b0 + b1*income + b2*credit + b3*age
    # Negative beta: higher rate → lower application
    # Credit score makes people more rate sensitive
    b0, b1, b2, b3 = -0.8, -0.1, -0.2, 0.05

    # Compute true parameters
    alpha_true = a0 + a1*income + a2*credit_score + a3*age
    beta_true = b0 + b1*income + b2*credit_score + b3*age

    # Treatment: interest rate (centered around 0)
    T = np.random.normal(0, 1, n)

    # Outcome
    logits = alpha_true + beta_true * T
    prob = expit(logits)
    Y = np.random.binomial(1, prob).astype(float)

    # True target: E[beta(X)] = b0 (since all X are mean 0)
    mu_true = b0

    # Also compute AME: E[p(1-p)*beta]
    p_at_zero = expit(alpha_true)  # P(apply | T=0)
    ame_true = np.mean(p_at_zero * (1 - p_at_zero) * beta_true)

    dgp_info = {
        "n": n,
        "seed": seed,
        "alpha_params": (a0, a1, a2, a3),
        "beta_params": (b0, b1, b2, b3),
        "alpha_true": alpha_true,
        "beta_true": beta_true,
        "mu_true": mu_true,  # E[beta(X)]
        "ame_true": ame_true,  # E[p(1-p)*beta]
        "mean_Y": Y.mean(),
        "heterogeneity_cv": np.std(beta_true) / np.abs(np.mean(beta_true)),
    }

    return Y, T, X, dgp_info


# ============================================================
# PART B: BOOTSTRAP ORACLE
# ============================================================

def fit_oracle_logit(Y: np.ndarray, T: np.ndarray, X: np.ndarray) -> Dict[str, Any]:
    """
    Fit Oracle logistic regression with interaction.

    Model: logit(P(Y=1)) = c0 + c1*X1 + c2*X2 + c3*X3 + c4*T + c5*X1*T + c6*X2*T + c7*X3*T

    This gives:
    - alpha(x) = c0 + c1*X1 + c2*X2 + c3*X3
    - beta(x) = c4 + c5*X1 + c6*X2 + c7*X3
    """
    n = len(Y)

    # Design matrix with all interactions
    X_design = np.column_stack([
        np.ones(n),   # intercept
        X[:, 0],      # income
        X[:, 1],      # credit
        X[:, 2],      # age
        T,            # treatment
        X[:, 0] * T,  # income * T
        X[:, 1] * T,  # credit * T
        X[:, 2] * T,  # age * T
    ])

    # Fit logit
    model = sm.Logit(Y, X_design).fit(disp=0)

    # Extract parameters
    params = model.params
    c0, c1, c2, c3, c4, c5, c6, c7 = params

    # Compute theta(x) for each observation
    alpha_hat = c0 + c1*X[:, 0] + c2*X[:, 1] + c3*X[:, 2]
    beta_hat = c4 + c5*X[:, 0] + c6*X[:, 1] + c7*X[:, 2]
    theta_hat = np.column_stack([alpha_hat, beta_hat])

    # Compute mu_hat = E[beta(X)] = mean(beta_hat)
    mu_hat = beta_hat.mean()

    # Standard error via delta method
    # mu = c4 + c5*mean(X1) + c6*mean(X2) + c7*mean(X3)
    # Since X is standardized, means ≈ 0, so mu ≈ c4
    # SE(mu) ≈ SE(c4) (naive) or use full delta method
    cov = model.cov_params()

    # Gradient of mu w.r.t. params at means
    x_means = X.mean(axis=0)
    grad = np.array([0, 0, 0, 0, 1, x_means[0], x_means[1], x_means[2]])
    se_delta = np.sqrt(grad @ cov @ grad)

    # Naive SE (just SE of c4)
    se_naive = model.bse[4]

    return {
        "model": model,
        "params": params,
        "theta_hat": theta_hat,
        "alpha_hat": alpha_hat,
        "beta_hat": beta_hat,
        "mu_hat": mu_hat,
        "se_naive": se_naive,
        "se_delta": se_delta,
        "ci_lower": mu_hat - 1.96 * se_delta,
        "ci_upper": mu_hat + 1.96 * se_delta,
    }


def bootstrap_oracle(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Bootstrap inference for heterogeneous θ(x).

    For each bootstrap sample:
    1. Resample (Y, T, X) with replacement
    2. Fit Oracle logit
    3. Predict θ̂_b(x_i) on ORIGINAL X (not resampled)

    Returns bootstrap CIs for:
    - Each θ(x_i)
    - mu* = E[beta(X)]
    """
    n = len(Y)
    np.random.seed(seed)

    # Storage for bootstrap samples
    theta_boots = np.zeros((n_bootstrap, n, 2))  # (B, n, 2)
    mu_boots = np.zeros(n_bootstrap)

    iterator = tqdm(range(n_bootstrap), desc="Bootstrap") if verbose else range(n_bootstrap)

    for b in iterator:
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        Y_b = Y[idx]
        T_b = T[idx]
        X_b = X[idx]

        # Fit on bootstrap sample
        try:
            result_b = fit_oracle_logit(Y_b, T_b, X_b)

            # Predict on ORIGINAL X (not resampled)
            params = result_b["params"]
            c0, c1, c2, c3, c4, c5, c6, c7 = params

            alpha_hat_b = c0 + c1*X[:, 0] + c2*X[:, 1] + c3*X[:, 2]
            beta_hat_b = c4 + c5*X[:, 0] + c6*X[:, 1] + c7*X[:, 2]

            theta_boots[b, :, 0] = alpha_hat_b
            theta_boots[b, :, 1] = beta_hat_b
            mu_boots[b] = beta_hat_b.mean()

        except Exception as e:
            # If fit fails, use NaN
            theta_boots[b] = np.nan
            mu_boots[b] = np.nan

    # Remove failed fits
    valid = ~np.isnan(mu_boots)
    theta_boots = theta_boots[valid]
    mu_boots = mu_boots[valid]
    n_valid = len(mu_boots)

    # Point estimates (mean of bootstrap)
    theta_mean = theta_boots.mean(axis=0)
    mu_mean = mu_boots.mean()

    # Bootstrap CIs (percentile method)
    theta_lower = np.percentile(theta_boots, 2.5, axis=0)
    theta_upper = np.percentile(theta_boots, 97.5, axis=0)
    mu_lower = np.percentile(mu_boots, 2.5)
    mu_upper = np.percentile(mu_boots, 97.5)

    # Bootstrap SE
    theta_se = theta_boots.std(axis=0)
    mu_se = mu_boots.std()

    return {
        "n_bootstrap": n_bootstrap,
        "n_valid": n_valid,
        "theta_boots": theta_boots,
        "mu_boots": mu_boots,
        "theta_mean": theta_mean,
        "theta_lower": theta_lower,
        "theta_upper": theta_upper,
        "theta_se": theta_se,
        "mu_mean": mu_mean,
        "mu_lower": mu_lower,
        "mu_upper": mu_upper,
        "mu_se": mu_se,
    }


# ============================================================
# PART C: NN INFERENCE
# ============================================================

def run_nn_inference(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    n_folds: int = 30,
    epochs: int = 100,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run NN inference using the package.
    """
    result = structural_dml(
        Y=Y,
        T=T,
        X=X,
        family="logit",
        n_folds=n_folds,
        epochs=epochs,
        lambda_method="aggregate",
        verbose=verbose,
    )

    return {
        "mu_hat": result.mu_hat,
        "mu_naive": result.mu_naive,
        "se": result.se,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "theta_hat": result.theta_hat,
        "psi_values": result.psi_values,
        "diagnostics": result.diagnostics,
    }


# ============================================================
# PART D: COMPARISON
# ============================================================

def print_comparison_table(
    dgp_info: Dict[str, Any],
    oracle_result: Dict[str, Any],
    bootstrap_result: Dict[str, Any],
    nn_result: Dict[str, Any],
):
    """Print side-by-side comparison of all methods."""
    mu_true = dgp_info["mu_true"]

    print("\n" + "=" * 80)
    print("COMPARISON TABLE: Target μ* = E[β(X)]")
    print("=" * 80)
    print(f"True μ* = {mu_true:.6f}")
    print()

    print(f"{'Method':<25} {'Estimate':>12} {'SE':>10} {'CI_lo':>10} {'CI_hi':>10} {'Covers':>8}")
    print("-" * 80)

    methods = [
        ("Oracle (Naive SE)", oracle_result["mu_hat"], oracle_result["se_naive"],
         oracle_result["mu_hat"] - 1.96*oracle_result["se_naive"],
         oracle_result["mu_hat"] + 1.96*oracle_result["se_naive"]),

        ("Oracle (Delta SE)", oracle_result["mu_hat"], oracle_result["se_delta"],
         oracle_result["ci_lower"], oracle_result["ci_upper"]),

        ("Bootstrap Oracle", bootstrap_result["mu_mean"], bootstrap_result["mu_se"],
         bootstrap_result["mu_lower"], bootstrap_result["mu_upper"]),

        ("NN (Naive)", nn_result["mu_naive"],
         nn_result["theta_hat"][:, 1].std() / np.sqrt(len(nn_result["theta_hat"])),
         nn_result["mu_naive"] - 1.96 * nn_result["theta_hat"][:, 1].std() / np.sqrt(len(nn_result["theta_hat"])),
         nn_result["mu_naive"] + 1.96 * nn_result["theta_hat"][:, 1].std() / np.sqrt(len(nn_result["theta_hat"]))),

        ("NN (IF Corrected)", nn_result["mu_hat"], nn_result["se"],
         nn_result["ci_lower"], nn_result["ci_upper"]),
    ]

    for name, est, se, ci_lo, ci_hi in methods:
        covers = ci_lo <= mu_true <= ci_hi
        print(f"{name:<25} {est:>12.6f} {se:>10.6f} {ci_lo:>10.4f} {ci_hi:>10.4f} {'YES' if covers else 'NO':>8}")

    print("=" * 80)


# ============================================================
# PART E: HETEROGENEITY ANALYSIS
# ============================================================

def analyze_heterogeneity(
    dgp_info: Dict[str, Any],
    oracle_result: Dict[str, Any],
    bootstrap_result: Dict[str, Any],
    nn_result: Dict[str, Any],
):
    """Analyze heterogeneity in treatment effects."""
    beta_true = dgp_info["beta_true"]
    beta_oracle = oracle_result["beta_hat"]
    beta_nn = nn_result["theta_hat"][:, 1]

    print("\n" + "=" * 80)
    print("HETEROGENEITY ANALYSIS: β(x) = Rate Sensitivity")
    print("=" * 80)

    # Correlation with true effects
    corr_oracle, _ = pearsonr(beta_oracle, beta_true)
    corr_nn, _ = pearsonr(beta_nn, beta_true)

    print("\n--- Correlation with True β(x) ---")
    print(f"  Oracle β̂(x): {corr_oracle:.4f}")
    print(f"  NN β̂(x):     {corr_nn:.4f}")

    # Distribution statistics
    print("\n--- Distribution of β̂(x) ---")
    print(f"{'Statistic':<15} {'True':>12} {'Oracle':>12} {'NN':>12}")
    print("-" * 55)

    stats = [
        ("Mean", beta_true.mean(), beta_oracle.mean(), beta_nn.mean()),
        ("Std", beta_true.std(), beta_oracle.std(), beta_nn.std()),
        ("Min", beta_true.min(), beta_oracle.min(), beta_nn.min()),
        ("Q10", np.percentile(beta_true, 10), np.percentile(beta_oracle, 10), np.percentile(beta_nn, 10)),
        ("Median", np.median(beta_true), np.median(beta_oracle), np.median(beta_nn)),
        ("Q90", np.percentile(beta_true, 90), np.percentile(beta_oracle, 90), np.percentile(beta_nn, 90)),
        ("Max", beta_true.max(), beta_oracle.max(), beta_nn.max()),
    ]

    for name, true, oracle, nn in stats:
        print(f"{name:<15} {true:>12.4f} {oracle:>12.4f} {nn:>12.4f}")

    # CV (coefficient of variation)
    cv_true = beta_true.std() / np.abs(beta_true.mean())
    cv_oracle = beta_oracle.std() / np.abs(beta_oracle.mean())
    cv_nn = beta_nn.std() / np.abs(beta_nn.mean())

    print(f"\n{'CV':<15} {cv_true:>12.4f} {cv_oracle:>12.4f} {cv_nn:>12.4f}")

    # Bootstrap CIs for individual β(x)
    print("\n--- Bootstrap CIs for β(x_i) (sample of 10) ---")
    print(f"{'i':>5} {'True':>10} {'Oracle':>10} {'Boot_lo':>10} {'Boot_hi':>10} {'NN':>10} {'Covers':>8}")
    print("-" * 70)

    np.random.seed(42)
    sample_idx = np.random.choice(len(beta_true), size=10, replace=False)

    n_covers = 0
    for i in sample_idx:
        boot_lo = bootstrap_result["theta_lower"][i, 1]
        boot_hi = bootstrap_result["theta_upper"][i, 1]
        covers = boot_lo <= beta_true[i] <= boot_hi
        if covers:
            n_covers += 1
        print(f"{i:>5} {beta_true[i]:>10.4f} {beta_oracle[i]:>10.4f} "
              f"{boot_lo:>10.4f} {boot_hi:>10.4f} {beta_nn[i]:>10.4f} "
              f"{'YES' if covers else 'NO':>8}")

    # Overall bootstrap coverage for θ(x)
    beta_boot_lower = bootstrap_result["theta_lower"][:, 1]
    beta_boot_upper = bootstrap_result["theta_upper"][:, 1]
    coverage_theta = np.mean((beta_boot_lower <= beta_true) & (beta_true <= beta_boot_upper))

    print(f"\nBootstrap coverage for β(x): {coverage_theta*100:.1f}%")
    print("=" * 80)

    return {
        "corr_oracle": corr_oracle,
        "corr_nn": corr_nn,
        "cv_true": cv_true,
        "cv_oracle": cv_oracle,
        "cv_nn": cv_nn,
        "coverage_theta": coverage_theta,
    }


# ============================================================
# ROUND G: SE CALIBRATION (Multi-Seed)
# ============================================================

def run_round_g_se_validation(
    n: int = 1000,
    M: int = 100,
    n_folds: int = 30,
    epochs: int = 50,
    lambda_methods: list = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Round G: Multi-seed SE validation.

    This confirms SE calibration on the realistic heterogeneous DGP.
    Runs M seeds and checks:
    - Coverage: 93-97%
    - SE Ratio: 0.9-1.1

    Args:
        lambda_methods: List of methods to compare (default: ["aggregate"])
                       Options: "aggregate", "lgbm", "mlp", "rf", "ridge"

    Returns dict with results.
    """
    import warnings
    warnings.filterwarnings("ignore")

    if lambda_methods is None:
        lambda_methods = ["aggregate"]

    print("\n" + "=" * 80)
    print("ROUND G: SE CALIBRATION (Multi-Seed)")
    print("=" * 80)
    print(f"\nSettings: n={n}, M={M}, n_folds={n_folds}, epochs={epochs}")
    print(f"Lambda methods: {lambda_methods}")

    # Get true mu from seed 0
    _, _, _, dgp_info = generate_loan_application_data(n=n, seed=0)
    mu_true = dgp_info["mu_true"]
    print(f"True μ* = {mu_true:.6f}")

    all_results = {}

    for lambda_method in lambda_methods:
        print(f"\n--- Lambda method: {lambda_method} ---")
        print(f"Running {M} seeds...")

        mu_hats = []
        ses = []
        covers = 0

        iterator = tqdm(range(M), desc=f"{lambda_method}") if verbose else range(M)

        for seed in iterator:
            # Generate data with this seed
            Y, T, X, _ = generate_loan_application_data(n=n, seed=seed)

            # Run NN inference
            try:
                result = structural_dml(
                    Y=Y,
                    T=T,
                    X=X,
                    family="logit",
                    n_folds=n_folds,
                    epochs=epochs,
                    lambda_method=lambda_method,
                    verbose=False,
                )

                mu_hat = result.mu_hat
                se = result.se

                mu_hats.append(mu_hat)
                ses.append(se)

                # Check coverage
                ci_lo = mu_hat - 1.96 * se
                ci_hi = mu_hat + 1.96 * se
                if ci_lo <= mu_true <= ci_hi:
                    covers += 1

            except Exception as e:
                # If fit fails, skip
                pass

        # Compute statistics
        n_valid = len(mu_hats)
        if n_valid == 0:
            print(f"ERROR: All fits failed for {lambda_method}!")
            all_results[lambda_method] = {"status": "FAIL", "error": "All fits failed"}
            continue

        mu_hats = np.array(mu_hats)
        ses = np.array(ses)

        coverage = covers / n_valid
        mean_se = np.mean(ses)
        empirical_se = np.std(mu_hats)
        se_ratio = mean_se / empirical_se if empirical_se > 0 else np.inf
        mean_bias = np.mean(mu_hats) - mu_true

        # Checks
        coverage_pass = 0.93 <= coverage <= 0.97
        se_ratio_pass = 0.9 <= se_ratio <= 1.1
        status = "PASS" if (coverage_pass and se_ratio_pass) else "FAIL"

        all_results[lambda_method] = {
            "n_valid": n_valid,
            "mean_mu_hat": np.mean(mu_hats),
            "bias": mean_bias,
            "coverage": coverage,
            "mean_se": mean_se,
            "empirical_se": empirical_se,
            "se_ratio": se_ratio,
            "coverage_pass": coverage_pass,
            "se_ratio_pass": se_ratio_pass,
            "status": status,
        }

        print(f"  Valid: {n_valid}/{M}")
        print(f"  Bias: {mean_bias:.6f}")
        print(f"  Coverage: {coverage*100:.1f}%")
        print(f"  SE Ratio: {se_ratio:.3f}")
        print(f"  Status: {status}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"\n{'Method':<12} {'Coverage':>10} {'SE Ratio':>10} {'Bias':>10} {'Status':>8}")
    print("-" * 55)

    for method, res in all_results.items():
        if "error" in res:
            print(f"{method:<12} {'FAILED':>10} {'-':>10} {'-':>10} {'FAIL':>8}")
        else:
            print(f"{method:<12} {res['coverage']*100:>9.1f}% {res['se_ratio']:>10.3f} "
                  f"{res['bias']:>10.4f} {res['status']:>8}")

    print("=" * 80)

    # Overall status
    any_pass = any(r.get("status") == "PASS" for r in all_results.values())
    all_pass = all(r.get("status") == "PASS" for r in all_results.values())

    return {
        "n": n,
        "M": M,
        "mu_true": mu_true,
        "results": all_results,
        "any_pass": any_pass,
        "all_pass": all_pass,
    }


# ============================================================
# PART F: VALIDATION
# ============================================================

def run_validation(
    dgp_info: Dict[str, Any],
    oracle_result: Dict[str, Any],
    bootstrap_result: Dict[str, Any],
    nn_result: Dict[str, Any],
    hetero_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Run validation checks."""
    mu_true = dgp_info["mu_true"]

    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)

    checks = {}

    # 1. Oracle covers true μ*
    oracle_covers = oracle_result["ci_lower"] <= mu_true <= oracle_result["ci_upper"]
    checks["Oracle CI covers μ*"] = oracle_covers

    # 2. Bootstrap covers true μ*
    boot_covers = bootstrap_result["mu_lower"] <= mu_true <= bootstrap_result["mu_upper"]
    checks["Bootstrap CI covers μ*"] = boot_covers

    # 3. NN IF covers true μ*
    nn_covers = nn_result["ci_lower"] <= mu_true <= nn_result["ci_upper"]
    checks["NN IF CI covers μ*"] = nn_covers

    # 4. NN IF bias < 0.1
    nn_bias = abs(nn_result["mu_hat"] - mu_true)
    checks["NN IF |bias| < 0.1"] = nn_bias < 0.1

    # 5. Oracle-NN correlation > 0.5 (heterogeneity recovery is hard with limited data)
    corr = hetero_result["corr_nn"]
    checks["Corr(β̂_oracle, β_true) > 0.5"] = hetero_result["corr_oracle"] > 0.5
    checks["Corr(β̂_nn, β_true) > 0.3"] = corr > 0.3

    # 6. Bootstrap θ(x) coverage > 80%
    checks["Bootstrap θ(x) coverage > 80%"] = hetero_result["coverage_theta"] > 0.8

    print()
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_pass = all(checks.values())
    n_pass = sum(checks.values())
    n_total = len(checks)

    print()
    print(f"  Result: {n_pass}/{n_total} checks passed")
    print("=" * 80)

    return {"checks": checks, "all_pass": all_pass, "n_pass": n_pass, "n_total": n_total}


# ============================================================
# MAIN
# ============================================================

def run_eval_07(
    n: int = 3000,
    seed: int = 42,
    n_bootstrap: int = 200,
    n_folds: int = 30,
    epochs: int = 100,
    verbose: bool = True,
):
    """Run eval 07: End-to-End User Experience."""

    print("=" * 80)
    print("EVAL 07: END-TO-END USER EXPERIENCE")
    print("=" * 80)
    print()
    print("This eval demonstrates the package as a real analyst would use it.")
    print("Scenario: Loan application with heterogeneous rate sensitivity.")
    print()
    print(f"Settings: n={n}, seed={seed}, n_bootstrap={n_bootstrap}, n_folds={n_folds}")

    # PART A: Generate data
    print("\n" + "-" * 80)
    print("PART A: DATA GENERATION")
    print("-" * 80)

    Y, T, X, dgp_info = generate_loan_application_data(n=n, seed=seed)

    print(f"\nDGP:")
    print(f"  α(x) = {dgp_info['alpha_params'][0]:.2f} + {dgp_info['alpha_params'][1]:.2f}*income + "
          f"{dgp_info['alpha_params'][2]:.2f}*credit + {dgp_info['alpha_params'][3]:.2f}*age")
    print(f"  β(x) = {dgp_info['beta_params'][0]:.2f} + {dgp_info['beta_params'][1]:.2f}*income + "
          f"{dgp_info['beta_params'][2]:.2f}*credit + {dgp_info['beta_params'][3]:.2f}*age")
    print(f"  P(Y=1|X,T) = sigmoid(α(X) + β(X)*T)")
    print(f"\n  True μ* = E[β(X)] = {dgp_info['mu_true']:.6f}")
    print(f"  True AME = E[p(1-p)·β] = {dgp_info['ame_true']:.6f}")
    print(f"  Mean(Y) = {dgp_info['mean_Y']:.3f}")
    print(f"  Heterogeneity CV = {dgp_info['heterogeneity_cv']:.3f}")

    # PART B: Oracle inference
    print("\n" + "-" * 80)
    print("PART B: ORACLE INFERENCE (Bootstrap)")
    print("-" * 80)

    print("\nFitting Oracle logit with interactions...")
    oracle_result = fit_oracle_logit(Y, T, X)
    print(f"  Oracle μ̂ = {oracle_result['mu_hat']:.6f}")
    print(f"  Oracle SE (delta) = {oracle_result['se_delta']:.6f}")

    print(f"\nRunning bootstrap (B={n_bootstrap})...")
    bootstrap_result = bootstrap_oracle(Y, T, X, n_bootstrap=n_bootstrap, seed=seed, verbose=verbose)
    print(f"  Bootstrap μ̂ = {bootstrap_result['mu_mean']:.6f}")
    print(f"  Bootstrap SE = {bootstrap_result['mu_se']:.6f}")
    print(f"  Bootstrap 95% CI = [{bootstrap_result['mu_lower']:.4f}, {bootstrap_result['mu_upper']:.4f}]")

    # PART C: NN inference
    print("\n" + "-" * 80)
    print("PART C: NN INFERENCE (structural_dml)")
    print("-" * 80)

    print(f"\nRunning structural_dml(family='logit', n_folds={n_folds}, epochs={epochs})...")
    nn_result = run_nn_inference(Y, T, X, n_folds=n_folds, epochs=epochs, verbose=False)
    print(f"  NN μ̂ (naive) = {nn_result['mu_naive']:.6f}")
    print(f"  NN μ̂ (IF) = {nn_result['mu_hat']:.6f}")
    print(f"  NN SE = {nn_result['se']:.6f}")
    print(f"  NN 95% CI = [{nn_result['ci_lower']:.4f}, {nn_result['ci_upper']:.4f}]")

    # PART D: Comparison
    print("\n" + "-" * 80)
    print("PART D: COMPARISON")
    print("-" * 80)

    print_comparison_table(dgp_info, oracle_result, bootstrap_result, nn_result)

    # PART E: Heterogeneity
    print("\n" + "-" * 80)
    print("PART E: HETEROGENEITY ANALYSIS")
    print("-" * 80)

    hetero_result = analyze_heterogeneity(dgp_info, oracle_result, bootstrap_result, nn_result)

    # PART F: Validation
    print("\n" + "-" * 80)
    print("PART F: VALIDATION")
    print("-" * 80)

    validation = run_validation(dgp_info, oracle_result, bootstrap_result, nn_result, hetero_result)

    # Final verdict
    print("\n" + "=" * 80)
    if validation["all_pass"]:
        print("EVAL 07: PASS")
    else:
        print(f"EVAL 07: {validation['n_pass']}/{validation['n_total']} CHECKS PASSED")
    print("=" * 80)

    return {
        "dgp_info": dgp_info,
        "oracle_result": oracle_result,
        "bootstrap_result": bootstrap_result,
        "nn_result": nn_result,
        "hetero_result": hetero_result,
        "validation": validation,
    }


if __name__ == "__main__":
    import sys

    # Parse arguments
    args = sys.argv[1:]

    # Check for Round G modes
    if "--compare" in args:
        # Compare Lambda methods (M=50 for speed)
        # Note: structural_dml supports: aggregate, mlp, rf, ridge, lgbm
        result = run_round_g_se_validation(
            n=1000, M=50, n_folds=20, epochs=30,
            lambda_methods=["aggregate", "lgbm"]
        )
    elif "--round-g" in args:
        # Full Round G: M=100 seeds
        result = run_round_g_se_validation(n=1000, M=100, n_folds=30, epochs=50)
    elif "--quick-g" in args:
        # Quick Round G: M=20 seeds
        result = run_round_g_se_validation(n=1000, M=20, n_folds=20, epochs=30)
    elif "--quick" in args:
        # Quick mode for Parts A-F
        result = run_eval_07(n=1000, n_bootstrap=50, n_folds=20, epochs=50)
    else:
        # Full mode for Parts A-F
        result = run_eval_07(n=3000, n_bootstrap=200, n_folds=30, epochs=100)
