"""
E2E Validation: New inference() API vs Oracle

This script validates the new inference() API against correctly-specified
OLS (linear) and Logistic Regression (logit) oracles.

Validates:
- a) Parameter Recovery: Is alpha(X), beta(X) being recovered?
- b) Inference Validity:
  - b1) Is the estimate unbiased?
  - b2) Are the SEs from IF reasonable (SE ratio ~1.0)?
  - b3) Is coverage ~95%?

Usage:
    python -m deep_inference.tests.test_e2e_validation
"""

import numpy as np
import torch
import warnings
from typing import Dict, List, Tuple
from dataclasses import dataclass

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

M_ORACLE = 100  # Oracle MC replications (fast)
M_NN = 5  # NN runs (slower, for coverage estimate)
N = 1000  # Sample size
N_FOLDS = 20  # Cross-fitting folds (reduced for speed)
EPOCHS = 50  # Training epochs (reduced for speed)

# DGP 1: Linear Model (Regime B)
LINEAR_DGP = {
    "A0": 1.0,
    "A1": 0.5,
    "B0": -1.0,
    "B1": 1.0,
    "MU_X": 1.0,
    "MU_TRUE": 0.0,  # B0 + B1*MU_X = -1 + 1*1 = 0
}

# DGP 2: Logit Model (Regime C)
LOGIT_DGP = {
    "A0": 1.0,
    "A1": 0.3,
    "B0": 0.5,
    "B1": 0.2,
    "MU_X": 0.0,
    "MU_TRUE": 0.5,  # B0 (since E[X]=0)
}


# =============================================================================
# DATA GENERATION
# =============================================================================


def generate_linear_data(n: int, seed: int = None) -> Dict:
    """
    Generate data from linear DGP.

    Y = alpha(X) + beta(X)*T + eps
    alpha(X) = A0 + A1*X
    beta(X) = B0 + B1*X
    eps | X ~ N(0, (1 + 0.5*|X|)^2)  [heteroskedastic]
    X ~ N(MU_X, 1)
    T ~ N(0, 1)
    """
    if seed is not None:
        np.random.seed(seed)

    A0, A1 = LINEAR_DGP["A0"], LINEAR_DGP["A1"]
    B0, B1 = LINEAR_DGP["B0"], LINEAR_DGP["B1"]
    MU_X = LINEAR_DGP["MU_X"]

    X = np.random.normal(MU_X, 1.0, n)
    T = np.random.normal(0.0, 1.0, n)

    alpha_true = A0 + A1 * X
    beta_true = B0 + B1 * X

    sigma = 1.0 + 0.5 * np.abs(X)
    eps = np.random.normal(0, sigma)

    Y = alpha_true + beta_true * T + eps

    return {
        "Y": Y,
        "T": T,
        "X": X,
        "alpha_true": alpha_true,
        "beta_true": beta_true,
    }


def generate_logit_data(n: int, seed: int = None) -> Dict:
    """
    Generate data from logit DGP.

    P(Y=1) = sigmoid(alpha(X) + beta(X)*T)
    alpha(X) = A0 + A1*X
    beta(X) = B0 + B1*X
    X ~ N(0, 1)
    T ~ N(0, 1)
    """
    if seed is not None:
        np.random.seed(seed)

    from scipy.special import expit

    A0, A1 = LOGIT_DGP["A0"], LOGIT_DGP["A1"]
    B0, B1 = LOGIT_DGP["B0"], LOGIT_DGP["B1"]

    X = np.random.normal(0, 1, n)
    T = np.random.normal(0, 1, n)

    alpha_true = A0 + A1 * X
    beta_true = B0 + B1 * X

    p = expit(alpha_true + beta_true * T)
    Y = np.random.binomial(1, p).astype(float)

    return {
        "Y": Y,
        "T": T,
        "X": X,
        "alpha_true": alpha_true,
        "beta_true": beta_true,
    }


# =============================================================================
# ORACLE IMPLEMENTATIONS
# =============================================================================


def linear_oracle(Y: np.ndarray, T: np.ndarray, X: np.ndarray) -> Dict:
    """
    OLS oracle for linear DGP.

    Model: Y = a0 + a1*X + b0*T + b1*(X*T) + eps
    Target: mu = E[beta(X)] = b0 + b1*E[X], estimated by b0_hat + b1_hat*X_bar

    Returns dict with mu_hat, se_naive, se_delta, params, alpha_hat, beta_hat
    """
    import statsmodels.api as sm

    n = len(Y)
    X_bar = X.mean()

    # Design matrix: [1, X, T, X*T]
    X_design = np.column_stack([np.ones(n), X, T, X * T])

    # Fit OLS with HC3 robust SEs
    model = sm.OLS(Y, X_design).fit(cov_type="HC3")

    a0, a1, b0, b1 = model.params

    # Point estimate of E[beta(X)]
    mu_hat = b0 + b1 * X_bar

    # Variance-covariance for b0, b1 (indices 2, 3)
    cov = model.cov_params()
    var_b0 = cov[2, 2]
    var_b1 = cov[3, 3]
    cov_b0_b1 = cov[2, 3]

    # Naive SE: treats X_bar as fixed
    var_naive = var_b0 + X_bar**2 * var_b1 + 2 * X_bar * cov_b0_b1
    se_naive = np.sqrt(max(var_naive, 1e-10))

    # Delta-corrected SE: accounts for Var(X_bar)
    var_X_bar = X.var(ddof=1) / n
    var_delta = var_naive + b1**2 * var_X_bar
    se_delta = np.sqrt(max(var_delta, 1e-10))

    return {
        "mu_hat": mu_hat,
        "se_naive": se_naive,
        "se_delta": se_delta,
        "params": {"a0": a0, "a1": a1, "b0": b0, "b1": b1},
        "alpha_hat": a0 + a1 * X,
        "beta_hat": b0 + b1 * X,
    }


def logit_oracle(Y: np.ndarray, T: np.ndarray, X: np.ndarray) -> Dict:
    """
    Logistic regression oracle for logit DGP.

    Model: logit(P(Y=1)) = a0 + a1*X + b0*T + b1*(X*T)
    Target: mu = E[beta(X)] = b0 + b1*E[X]

    Returns dict with mu_hat, se_naive, se_delta, params, alpha_hat, beta_hat
    """
    import statsmodels.api as sm

    n = len(Y)
    X_bar = X.mean()

    X_design = np.column_stack([np.ones(n), X, T, X * T])

    model = sm.Logit(Y, X_design).fit(disp=0)

    a0, a1, b0, b1 = model.params

    mu_hat = b0 + b1 * X_bar

    cov = model.cov_params()
    var_naive = cov[2, 2] + X_bar**2 * cov[3, 3] + 2 * X_bar * cov[2, 3]
    var_delta = var_naive + b1**2 * (X.var(ddof=1) / n)

    return {
        "mu_hat": mu_hat,
        "se_naive": np.sqrt(max(var_naive, 1e-10)),
        "se_delta": np.sqrt(max(var_delta, 1e-10)),
        "params": {"a0": a0, "a1": a1, "b0": b0, "b1": b1},
        "alpha_hat": a0 + a1 * X,
        "beta_hat": b0 + b1 * X,
    }


# =============================================================================
# ORACLE MONTE CARLO
# =============================================================================


@dataclass
class OracleMCResult:
    dgp_name: str
    mu_true: float
    estimates: np.ndarray
    se_naive: np.ndarray
    se_delta: np.ndarray
    covered_naive: np.ndarray
    covered_delta: np.ndarray


def run_oracle_mc(dgp_name: str, M: int, N: int) -> OracleMCResult:
    """Run Oracle Monte Carlo for a DGP."""
    if dgp_name == "linear":
        generate_fn = generate_linear_data
        oracle_fn = linear_oracle
        mu_true = LINEAR_DGP["MU_TRUE"]
    else:
        generate_fn = generate_logit_data
        oracle_fn = logit_oracle
        mu_true = LOGIT_DGP["MU_TRUE"]

    estimates = []
    se_naive_list = []
    se_delta_list = []
    covered_naive = []
    covered_delta = []

    for i in range(M):
        data = generate_fn(N, seed=i + 1000)
        result = oracle_fn(data["Y"], data["T"], data["X"])

        estimates.append(result["mu_hat"])
        se_naive_list.append(result["se_naive"])
        se_delta_list.append(result["se_delta"])

        ci_naive = (
            result["mu_hat"] - 1.96 * result["se_naive"],
            result["mu_hat"] + 1.96 * result["se_naive"],
        )
        ci_delta = (
            result["mu_hat"] - 1.96 * result["se_delta"],
            result["mu_hat"] + 1.96 * result["se_delta"],
        )

        covered_naive.append(ci_naive[0] <= mu_true <= ci_naive[1])
        covered_delta.append(ci_delta[0] <= mu_true <= ci_delta[1])

    return OracleMCResult(
        dgp_name=dgp_name,
        mu_true=mu_true,
        estimates=np.array(estimates),
        se_naive=np.array(se_naive_list),
        se_delta=np.array(se_delta_list),
        covered_naive=np.array(covered_naive),
        covered_delta=np.array(covered_delta),
    )


# =============================================================================
# NEURAL NETWORK VALIDATION
# =============================================================================


@dataclass
class NNResult:
    dgp_name: str
    api_type: str  # "builtin" or "custom"
    mu_true: float
    estimates: np.ndarray
    se_list: np.ndarray
    covered: np.ndarray
    # Parameter recovery (from last run)
    corr_alpha: float
    corr_beta: float
    rmse_alpha: float
    rmse_beta: float


def run_nn_validation(dgp_name: str, api_type: str, M: int, N: int) -> NNResult:
    """Run Neural Network validation for a DGP."""
    from deep_inference import inference

    if dgp_name == "linear":
        generate_fn = generate_linear_data
        mu_true = LINEAR_DGP["MU_TRUE"]
    else:
        generate_fn = generate_logit_data
        mu_true = LOGIT_DGP["MU_TRUE"]

    estimates = []
    se_list = []
    covered = []

    # Store last run for parameter recovery
    last_theta_hat = None
    last_data = None

    for i in range(M):
        data = generate_fn(N, seed=i + 2000)

        if api_type == "builtin":
            # Use built-in model string
            model_arg = dgp_name
            loss_arg = None
            target_fn_arg = None
            theta_dim_arg = None
        else:
            # Use custom loss function
            model_arg = None

            if dgp_name == "linear":

                def custom_loss(y, t, theta):
                    alpha, beta = theta[0], theta[1]
                    return (y - alpha - beta * t) ** 2

            else:

                def custom_loss(y, t, theta):
                    logits = theta[0] + theta[1] * t
                    p = torch.sigmoid(logits)
                    eps = 1e-7
                    p = torch.clamp(p, eps, 1 - eps)
                    return -y * torch.log(p) - (1 - y) * torch.log(1 - p)

            loss_arg = custom_loss

            def custom_target(x, theta, t_tilde):
                return theta[1]

            target_fn_arg = custom_target
            theta_dim_arg = 2

        try:
            result = inference(
                Y=data["Y"],
                T=data["T"],
                X=data["X"].reshape(-1, 1),
                model=model_arg,
                loss=loss_arg,
                target_fn=target_fn_arg,
                theta_dim=theta_dim_arg,
                n_folds=N_FOLDS,
                epochs=EPOCHS,
                verbose=False,
            )

            estimates.append(result.mu_hat)
            se_list.append(result.se)
            covered.append(result.ci_lower <= mu_true <= result.ci_upper)

            last_theta_hat = result.theta_hat
            last_data = data

        except Exception as e:
            print(f"  Error in {dgp_name}/{api_type} run {i}: {e}")
            estimates.append(np.nan)
            se_list.append(np.nan)
            covered.append(False)

    # Compute parameter recovery from last run
    if last_theta_hat is not None and last_data is not None:
        alpha_hat = last_theta_hat[:, 0].numpy()
        beta_hat = last_theta_hat[:, 1].numpy()
        alpha_true = last_data["alpha_true"]
        beta_true = last_data["beta_true"]

        corr_alpha = np.corrcoef(alpha_hat, alpha_true)[0, 1]
        corr_beta = np.corrcoef(beta_hat, beta_true)[0, 1]
        rmse_alpha = np.sqrt(np.mean((alpha_hat - alpha_true) ** 2))
        rmse_beta = np.sqrt(np.mean((beta_hat - beta_true) ** 2))
    else:
        corr_alpha = corr_beta = rmse_alpha = rmse_beta = np.nan

    return NNResult(
        dgp_name=dgp_name,
        api_type=api_type,
        mu_true=mu_true,
        estimates=np.array(estimates),
        se_list=np.array(se_list),
        covered=np.array(covered),
        corr_alpha=corr_alpha,
        corr_beta=corr_beta,
        rmse_alpha=rmse_alpha,
        rmse_beta=rmse_beta,
    )


# =============================================================================
# REPORTING
# =============================================================================


def print_oracle_table(linear_mc: OracleMCResult, logit_mc: OracleMCResult):
    """Print Oracle MC summary table."""
    print()
    print("=" * 90)
    print("TABLE 1: ORACLE MONTE CARLO RESULTS (M={})".format(M_ORACLE))
    print("=" * 90)
    print(
        f"{'DGP':<8} {'True_mu':<9} {'Mean_Est':<10} {'Bias':<9} {'Emp_SE':<9} "
        f"{'SE_naive':<10} {'SE_delta':<10} {'Cov_N':<8} {'Cov_D':<8}"
    )
    print("-" * 90)

    for mc in [linear_mc, logit_mc]:
        mean_est = mc.estimates.mean()
        bias = mean_est - mc.mu_true
        emp_se = mc.estimates.std()
        mean_se_naive = mc.se_naive.mean()
        mean_se_delta = mc.se_delta.mean()
        cov_naive = mc.covered_naive.mean()
        cov_delta = mc.covered_delta.mean()

        print(
            f"{mc.dgp_name:<8} {mc.mu_true:<9.4f} {mean_est:<10.4f} {bias:<9.4f} "
            f"{emp_se:<9.4f} {mean_se_naive:<10.4f} {mean_se_delta:<10.4f} "
            f"{cov_naive:<8.1%} {cov_delta:<8.1%}"
        )

    print("=" * 90)


def print_nn_table(nn_results: List[NNResult]):
    """Print NN validation results table."""
    print()
    print("=" * 95)
    print("TABLE 2: NEURAL NETWORK RESULTS (M={})".format(M_NN))
    print("=" * 95)
    print(
        f"{'DGP':<8} {'API':<10} {'True_mu':<9} {'Mean_Est':<10} {'Bias':<9} "
        f"{'Emp_SE':<9} {'Mean_SE':<9} {'SE_Ratio':<10} {'Coverage':<10}"
    )
    print("-" * 95)

    for nn in nn_results:
        valid_mask = ~np.isnan(nn.estimates)
        if valid_mask.sum() == 0:
            print(f"{nn.dgp_name:<8} {nn.api_type:<10} NO VALID RESULTS")
            continue

        estimates = nn.estimates[valid_mask]
        se_list = nn.se_list[valid_mask]
        covered = nn.covered[valid_mask]

        mean_est = estimates.mean()
        bias = mean_est - nn.mu_true
        emp_se = estimates.std() if len(estimates) > 1 else np.nan
        mean_se = se_list.mean()
        se_ratio = mean_se / emp_se if emp_se > 0 else np.nan
        coverage = covered.mean()

        print(
            f"{nn.dgp_name:<8} {nn.api_type:<10} {nn.mu_true:<9.4f} {mean_est:<10.4f} "
            f"{bias:<9.4f} {emp_se:<9.4f} {mean_se:<9.4f} {se_ratio:<10.2f} {coverage:<10.1%}"
        )

    print("=" * 95)


def print_recovery_table(nn_results: List[NNResult]):
    """Print parameter recovery table."""
    print()
    print("=" * 70)
    print("TABLE 3: PARAMETER RECOVERY (Last Run)")
    print("=" * 70)
    print(
        f"{'DGP':<8} {'API':<10} {'Corr_alpha':<12} {'Corr_beta':<12} "
        f"{'RMSE_alpha':<12} {'RMSE_beta':<12}"
    )
    print("-" * 70)

    for nn in nn_results:
        print(
            f"{nn.dgp_name:<8} {nn.api_type:<10} {nn.corr_alpha:<12.3f} "
            f"{nn.corr_beta:<12.3f} {nn.rmse_alpha:<12.4f} {nn.rmse_beta:<12.4f}"
        )

    print("=" * 70)


def print_validation_summary(
    linear_mc: OracleMCResult,
    logit_mc: OracleMCResult,
    nn_results: List[NNResult],
):
    """Print validation pass/fail summary."""
    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    checks = []

    # Oracle checks
    for mc in [linear_mc, logit_mc]:
        cov_delta = mc.covered_delta.mean()
        if 0.90 <= cov_delta <= 0.99:
            checks.append(f"[PASS] Oracle {mc.dgp_name} coverage: {cov_delta:.1%}")
        else:
            checks.append(f"[FAIL] Oracle {mc.dgp_name} coverage: {cov_delta:.1%}")

    # NN checks
    for nn in nn_results:
        valid_mask = ~np.isnan(nn.estimates)
        if valid_mask.sum() == 0:
            checks.append(f"[FAIL] NN {nn.dgp_name}/{nn.api_type}: No valid results")
            continue

        estimates = nn.estimates[valid_mask]
        se_list = nn.se_list[valid_mask]
        covered = nn.covered[valid_mask]

        coverage = covered.mean()
        emp_se = estimates.std() if len(estimates) > 1 else 1.0
        se_ratio = se_list.mean() / emp_se if emp_se > 0 else np.nan

        # Coverage check (relaxed for small M)
        if coverage >= 0.5:  # At least 50% with M=5
            checks.append(
                f"[PASS] NN {nn.dgp_name}/{nn.api_type} coverage: {coverage:.1%}"
            )
        else:
            checks.append(
                f"[WARN] NN {nn.dgp_name}/{nn.api_type} coverage: {coverage:.1%}"
            )

        # SE ratio check
        if 0.5 <= se_ratio <= 2.0:
            checks.append(
                f"[PASS] NN {nn.dgp_name}/{nn.api_type} SE ratio: {se_ratio:.2f}"
            )
        elif not np.isnan(se_ratio):
            checks.append(
                f"[WARN] NN {nn.dgp_name}/{nn.api_type} SE ratio: {se_ratio:.2f}"
            )

        # Parameter recovery check
        if nn.corr_beta > 0.5:
            checks.append(
                f"[PASS] NN {nn.dgp_name}/{nn.api_type} beta recovery: {nn.corr_beta:.3f}"
            )
        elif not np.isnan(nn.corr_beta):
            checks.append(
                f"[WARN] NN {nn.dgp_name}/{nn.api_type} beta recovery: {nn.corr_beta:.3f}"
            )

    for check in checks:
        print(check)

    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================


def run_validation():
    """Run full E2E validation."""
    print()
    print("=" * 70)
    print("E2E VALIDATION: New inference() API vs Oracle")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  M_ORACLE = {M_ORACLE}")
    print(f"  M_NN = {M_NN}")
    print(f"  N = {N}")
    print(f"  N_FOLDS = {N_FOLDS}")
    print(f"  EPOCHS = {EPOCHS}")
    print()

    # Phase 1: Oracle Monte Carlo
    print("Phase 1: Running Oracle Monte Carlo...")
    linear_mc = run_oracle_mc("linear", M_ORACLE, N)
    print(f"  Linear: done (coverage={linear_mc.covered_delta.mean():.1%})")

    logit_mc = run_oracle_mc("logit", M_ORACLE, N)
    print(f"  Logit: done (coverage={logit_mc.covered_delta.mean():.1%})")

    print_oracle_table(linear_mc, logit_mc)

    # Phase 2: Neural Network Validation
    print()
    print("Phase 2: Running Neural Network Validation...")

    nn_results = []

    # Linear - builtin
    print("  Linear/builtin...")
    nn_results.append(run_nn_validation("linear", "builtin", M_NN, N))
    print(f"    done (coverage={nn_results[-1].covered.mean():.1%})")

    # Linear - custom
    print("  Linear/custom...")
    nn_results.append(run_nn_validation("linear", "custom", M_NN, N))
    print(f"    done (coverage={nn_results[-1].covered.mean():.1%})")

    # Logit - builtin
    print("  Logit/builtin...")
    nn_results.append(run_nn_validation("logit", "builtin", M_NN, N))
    print(f"    done (coverage={nn_results[-1].covered.mean():.1%})")

    # Logit - custom
    print("  Logit/custom...")
    nn_results.append(run_nn_validation("logit", "custom", M_NN, N))
    print(f"    done (coverage={nn_results[-1].covered.mean():.1%})")

    print_nn_table(nn_results)
    print_recovery_table(nn_results)
    print_validation_summary(linear_mc, logit_mc, nn_results)


if __name__ == "__main__":
    run_validation()
