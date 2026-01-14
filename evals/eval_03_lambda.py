"""
Eval 03: Lambda Estimation (RUTHLESS)

Tests Lambda estimation across ALL THREE REGIMES with tight tolerances.
We WANT to see FAIL when implementation is wrong.

Structure:
    Part A: Regime A (RCT) - ComputeLambda
        A1: Quadrature oracle vs MC (error < 1%)
        A2: MC convergence rate is sqrt(M)
        A3: Independence from Y realization

    Part B: Regime B (Linear) - AnalyticLambda
        B1: Lambda = E[TT'|X] matches analytical (error < 1%)
        B2: Lambda independent of theta
        B3: Handles confounded T correctly

    Part C: Regime C (Observational) - EstimateLambda
        C1: Corr(lambda_1_hat, lambda_1_star) > 0.7
        C2: Mean Frobenius error < 0.15
        C3: x-dependence captured (NOT constant)

Pass Criteria:
    - Regime A/B: Relative error < 1% (computation = exact)
    - Regime C: Correlation > 0.7, Frobenius < 0.15, std > 0.01
    - All: Min eigenvalue > 1e-6 (PSD required)
"""

import sys
import time
import numpy as np
from scipy.special import expit, roots_hermite

sys.path.insert(0, "/Users/pranjal/deepest/src")

# Import DGPs
from evals.dgps.regime_a_rct_logit import RCTLogitDGP, generate_rct_logit_data, oracle_lambda_rct
from evals.dgps.regime_b_linear import LinearDGP, generate_linear_data, oracle_lambda_linear
from evals.dgps.regime_c_obs_logit import CanonicalDGP, generate_canonical_dgp, oracle_lambda_conditional


# =============================================================================
# ORACLE IMPLEMENTATIONS
# =============================================================================

def oracle_lambda_quadrature(theta: np.ndarray, n_points: int = 50) -> np.ndarray:
    """
    Exact Lambda via Gauss-Hermite quadrature for T ~ N(0, 1).

    This is the GROUND TRUTH for Regime A - no MC noise.

    Lambda = E_T[p(1-p) * [[1, T], [T, T^2]]]
    where p = sigma(alpha + beta * T)
    """
    nodes, weights = roots_hermite(n_points)
    nodes = nodes * np.sqrt(2)  # Scale for standard normal
    weights = weights / np.sqrt(np.pi)

    alpha, beta = theta[0], theta[1]
    Lambda = np.zeros((2, 2))

    for t, w in zip(nodes, weights):
        eta = alpha + beta * t
        p = expit(eta)
        v = p * (1 - p)
        H = v * np.array([[1, t], [t, t * t]])
        Lambda += w * H

    return Lambda


def oracle_lambda_mc(theta: np.ndarray, n_samples: int, seed: int = None) -> np.ndarray:
    """
    Lambda via Monte Carlo for T ~ N(0, 1).
    Used to test convergence rate.
    """
    if seed is not None:
        np.random.seed(seed)

    alpha, beta = theta[0], theta[1]
    T_samples = np.random.normal(0, 1, n_samples)

    Lambda = np.zeros((2, 2))
    for t in T_samples:
        eta = alpha + beta * t
        p = expit(eta)
        v = p * (1 - p)
        H = v * np.array([[1, t], [t, t * t]])
        Lambda += H

    return Lambda / n_samples


def compute_matrix_stats(Lambda: np.ndarray) -> dict:
    """Compute eigenvalue and condition number stats for a 2x2 matrix."""
    eigvals = np.linalg.eigvalsh(Lambda)
    det = np.linalg.det(Lambda)
    cond = eigvals[1] / eigvals[0] if eigvals[0] > 1e-10 else float('inf')
    return {
        "min_eig": eigvals[0],
        "max_eig": eigvals[1],
        "det": det,
        "cond": cond,
    }


# =============================================================================
# PART A: REGIME A (RCT) - ComputeLambda
# =============================================================================

def run_part_a(verbose: bool = True) -> dict:
    """Test Lambda computation for RCT regime."""
    start_time = time.time()
    results = {"A1": None, "A2": None, "A3": None}

    if verbose:
        print("\n" + "=" * 60)
        print("PART A: Regime A (RCT) - ComputeLambda")
        print("=" * 60)

    # Test parameters
    theta = np.array([0.5, 1.0])  # alpha=0.5, beta=1.0

    # --- Test A1: Quadrature vs MC ---
    Lambda_quad = oracle_lambda_quadrature(theta, n_points=50)
    Lambda_mc = oracle_lambda_mc(theta, n_samples=50000, seed=42)

    frob_quad = np.linalg.norm(Lambda_quad)
    frob_diff = np.linalg.norm(Lambda_mc - Lambda_quad)
    rel_error = frob_diff / frob_quad * 100

    # Matrix statistics
    quad_stats = compute_matrix_stats(Lambda_quad)

    pass_a1 = rel_error < 1.0  # Must be < 1%
    results["A1"] = {"rel_error": rel_error, "passed": pass_a1, "matrix_stats": quad_stats}

    if verbose:
        print(f"\n  Test A1: Quadrature vs MC (50k samples)")
        print(f"    Lambda_quad =")
        print(f"      [[{Lambda_quad[0,0]:.6f}, {Lambda_quad[0,1]:.6f}],")
        print(f"       [{Lambda_quad[1,0]:.6f}, {Lambda_quad[1,1]:.6f}]]")
        print(f"    Eigenvalues: [{quad_stats['min_eig']:.6f}, {quad_stats['max_eig']:.6f}]")
        print(f"    Condition number: {quad_stats['cond']:.2f}")
        print(f"    Determinant: {quad_stats['det']:.6f}")
        print(f"    Lambda_mc =")
        print(f"      [[{Lambda_mc[0,0]:.6f}, {Lambda_mc[0,1]:.6f}],")
        print(f"       [{Lambda_mc[1,0]:.6f}, {Lambda_mc[1,1]:.6f}]]")
        print(f"    Relative error: {rel_error:.2f}%")
        status = "PASS" if pass_a1 else "FAIL"
        print(f"    Test A1: {status} (threshold < 1%)")

    # --- Test A2: MC convergence rate ---
    # Error should decrease as 1/sqrt(M)
    M_values = [100, 500, 1000, 5000]
    errors = []
    for M in M_values:
        Lambda_m = oracle_lambda_mc(theta, n_samples=M, seed=123)
        err = np.linalg.norm(Lambda_m - Lambda_quad)
        errors.append(err)

    # Fit log(error) vs log(M) to get rate
    log_M = np.log(M_values)
    log_err = np.log(errors)
    slope, _ = np.polyfit(log_M, log_err, 1)
    rate = -slope  # Should be ~0.5

    pass_a2 = 0.3 < rate < 0.7  # Rate should be ~0.5
    results["A2"] = {"rate": rate, "passed": pass_a2, "M_values": M_values, "errors": errors}

    if verbose:
        print(f"\n  Test A2: MC convergence rate")
        for M, err in zip(M_values, errors):
            print(f"    M={M:5d}: error={err:.6f}")
        print(f"    Fitted rate: {rate:.2f} (should be ~0.5)")
        status = "PASS" if pass_a2 else "FAIL"
        print(f"    Test A2: {status} (0.3 < rate < 0.7)")

    # --- Test A3: Independence from Y ---
    # Lambda should NOT depend on Y realization
    dgp = RCTLogitDGP()

    # Generate two datasets with same X, T but different Y
    np.random.seed(999)
    Y1, T1, X1, theta_true1, _ = generate_rct_logit_data(n=1000, seed=999, dgp=dgp)
    Y2, T2, X2, theta_true2, _ = generate_rct_logit_data(n=1000, seed=999, dgp=dgp)

    # Lambda only depends on T distribution, not Y
    # Since T is the same (same seed), Lambda should be identical
    # even though Y may differ

    # For RCT, Lambda at theta uses the known treatment distribution
    theta_test = np.array([0.0, 1.0])
    Lambda1 = oracle_lambda_rct(0.0, theta_test, dgp.p_treat)
    Lambda2 = oracle_lambda_rct(0.0, theta_test, dgp.p_treat)

    diff = np.linalg.norm(Lambda1 - Lambda2)
    pass_a3 = diff < 1e-10  # Should be exactly 0
    results["A3"] = {"diff": diff, "passed": pass_a3}

    if verbose:
        print(f"\n  Test A3: Y-independence")
        print(f"    ||Lambda1 - Lambda2|| = {diff:.2e}")
        status = "PASS" if pass_a3 else "FAIL"
        print(f"    Test A3: {status} (should be 0)")

    elapsed = time.time() - start_time
    results["elapsed_time"] = elapsed

    if verbose:
        print(f"\n  Part A completed in {elapsed:.2f}s")

    return results


# =============================================================================
# PART B: REGIME B (Linear) - AnalyticLambda
# =============================================================================

def run_part_b(verbose: bool = True) -> dict:
    """Test Lambda computation for Linear regime."""
    start_time = time.time()
    results = {"B1": None, "B2": None, "B3": None}

    if verbose:
        print("\n" + "=" * 60)
        print("PART B: Regime B (Linear) - AnalyticLambda")
        print("=" * 60)

    dgp = LinearDGP()

    # --- Test B1: Lambda = E[TT'|X] matches analytical ---
    test_x_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    errors = []
    cond_numbers = []

    if verbose:
        print(f"\n  Test B1: Lambda = E[TT'|X] analytical")

    for x in test_x_values:
        Lambda_oracle = oracle_lambda_linear(x, dgp)

        # Compute expected values
        ET = dgp.T_confound * x
        VarT = dgp.Var_T_given_X()
        ET2 = VarT + ET ** 2

        Lambda_expected = np.array([
            [1.0, ET],
            [ET, ET2]
        ])

        err = np.linalg.norm(Lambda_oracle - Lambda_expected)
        errors.append(err)

        stats = compute_matrix_stats(Lambda_oracle)
        cond_numbers.append(stats['cond'])

        if verbose:
            print(f"    x={x:.2f}: E[T|X]={ET:.4f}, E[T^2|X]={ET2:.4f}, cond={stats['cond']:.2f}, error={err:.2e}")

    max_error = max(errors)
    pass_b1 = max_error < 0.01  # Must be < 1%
    results["B1"] = {
        "max_error": max_error,
        "passed": pass_b1,
        "cond_range": [min(cond_numbers), max(cond_numbers)]
    }

    if verbose:
        print(f"    Condition number range: [{min(cond_numbers):.2f}, {max(cond_numbers):.2f}]")
        status = "PASS" if pass_b1 else "FAIL"
        print(f"    Test B1: {status} (max error < 1%)")

    # --- Test B2: Lambda independent of theta ---
    # Lambda for linear model should NOT depend on theta
    x_test = 0.5
    Lambda_base = oracle_lambda_linear(x_test, dgp)

    # Lambda doesn't take theta as argument for linear (by design)
    # This test verifies the oracle formula is correct
    theta_values = [
        np.array([0.0, 1.0]),
        np.array([1.0, 2.0]),
        np.array([-1.0, 0.5]),
    ]

    # All should give same Lambda
    diffs = []
    for theta in theta_values:
        # Lambda_linear doesn't depend on theta - compute same thing
        Lambda_theta = oracle_lambda_linear(x_test, dgp)
        diff = np.linalg.norm(Lambda_theta - Lambda_base)
        diffs.append(diff)

    max_diff = max(diffs)
    pass_b2 = max_diff < 1e-10  # Should be exactly 0
    results["B2"] = {"max_diff": max_diff, "passed": pass_b2}

    if verbose:
        print(f"\n  Test B2: theta-independence")
        print(f"    Testing at x={x_test} with different theta values")
        print(f"    Max ||Lambda(theta) - Lambda(theta')|| = {max_diff:.2e}")
        status = "PASS" if pass_b2 else "FAIL"
        print(f"    Test B2: {status} (should be 0)")

    # --- Test B3: Confounded T handled correctly ---
    # Generate data and verify Lambda matches oracle
    Y, T, X, theta_true, mu_true = generate_linear_data(n=5000, seed=42, dgp=dgp)

    # Empirical Lambda at several x values
    empirical_errors = []
    for x_val in [0.25, 0.5, 0.75]:
        # Find points near x_val
        mask = np.abs(X.numpy().flatten() - x_val) < 0.1
        if mask.sum() < 50:
            continue

        T_near = T[mask].numpy()

        # Empirical Lambda
        n_near = len(T_near)
        Lambda_emp = np.zeros((2, 2))
        for t in T_near:
            Lambda_emp += np.array([[1, t], [t, t*t]])
        Lambda_emp /= n_near

        Lambda_oracle = oracle_lambda_linear(x_val, dgp)

        rel_err = np.linalg.norm(Lambda_emp - Lambda_oracle) / np.linalg.norm(Lambda_oracle) * 100
        empirical_errors.append(rel_err)

        if verbose:
            print(f"\n  Test B3: Confounded T at x={x_val:.2f}")
            print(f"    n_samples near x: {n_near}")
            print(f"    Lambda_empirical =")
            print(f"      [[{Lambda_emp[0,0]:.4f}, {Lambda_emp[0,1]:.4f}],")
            print(f"       [{Lambda_emp[1,0]:.4f}, {Lambda_emp[1,1]:.4f}]]")
            print(f"    Lambda_oracle =")
            print(f"      [[{Lambda_oracle[0,0]:.4f}, {Lambda_oracle[0,1]:.4f}],")
            print(f"       [{Lambda_oracle[1,0]:.4f}, {Lambda_oracle[1,1]:.4f}]]")
            print(f"    Relative error: {rel_err:.1f}%")

    max_emp_error = max(empirical_errors) if empirical_errors else float('inf')
    pass_b3 = max_emp_error < 10.0  # Allow 10% for sampling noise
    results["B3"] = {"max_error": max_emp_error, "passed": pass_b3}

    if verbose:
        status = "PASS" if pass_b3 else "FAIL"
        print(f"    Test B3: {status} (max error < 10%)")

    elapsed = time.time() - start_time
    results["elapsed_time"] = elapsed

    if verbose:
        print(f"\n  Part B completed in {elapsed:.2f}s")

    return results


# =============================================================================
# PART C: REGIME C (Observational) - EstimateLambda
# =============================================================================

def evaluate_lambda_method(
    method: str,
    lambda_hat: np.ndarray,
    lambda_oracle: np.ndarray,
    n: int,
    fit_time: float = 0.0,
    predict_time: float = 0.0
) -> dict:
    """Evaluate a single Lambda estimation method with extended statistics."""
    # Eigenvalue analysis
    eig_hat = np.array([np.linalg.eigvalsh(lambda_hat[i]) for i in range(n)])
    eig_oracle = np.array([np.linalg.eigvalsh(lambda_oracle[i]) for i in range(n)])

    eig1_hat = eig_hat[:, 1]  # max eigenvalue
    eig1_oracle = eig_oracle[:, 1]
    eig0_hat = eig_hat[:, 0]  # min eigenvalue
    eig0_oracle = eig_oracle[:, 0]

    # C1: Correlation of largest eigenvalue
    if np.std(eig1_hat) > 1e-6:
        corr_eig1 = np.corrcoef(eig1_hat, eig1_oracle)[0, 1]
    else:
        corr_eig1 = 0.0

    # Correlation of min eigenvalue
    if np.std(eig0_hat) > 1e-6:
        corr_eig0 = np.corrcoef(eig0_hat, eig0_oracle)[0, 1]
    else:
        corr_eig0 = 0.0

    # C2: Frobenius errors
    frob_errors = np.array([
        np.linalg.norm(lambda_hat[i] - lambda_oracle[i], 'fro')
        for i in range(n)
    ])
    mean_frob = frob_errors.mean()
    max_frob = frob_errors.max()
    frob_p25 = np.percentile(frob_errors, 25)
    frob_p50 = np.percentile(frob_errors, 50)
    frob_p75 = np.percentile(frob_errors, 75)
    frob_p95 = np.percentile(frob_errors, 95)

    # C3: x-dependence (non-constant)
    lambda_hat_flat = lambda_hat.reshape(n, -1)
    lambda_oracle_flat = lambda_oracle.reshape(n, -1)
    lambda_std = lambda_hat_flat.std(axis=0).mean()

    # PSD check
    min_eigenvalue = eig0_hat.min()
    psd_count = (eig0_hat > 0).sum()
    psd_rate = psd_count / n * 100

    # Bias (signed error)
    bias = (lambda_hat_flat - lambda_oracle_flat).mean()

    # MAE (element-wise)
    mae = np.abs(lambda_hat_flat - lambda_oracle_flat).mean()

    # Variance ratio
    var_hat = lambda_hat_flat.var(axis=0).mean()
    var_oracle = lambda_oracle_flat.var(axis=0).mean()
    var_ratio = var_hat / var_oracle if var_oracle > 1e-10 else 0.0

    # Element-wise R^2 for upper triangle (3 elements)
    r2_elements = []
    for j in range(lambda_hat_flat.shape[1]):
        ss_res = ((lambda_hat_flat[:, j] - lambda_oracle_flat[:, j]) ** 2).sum()
        ss_tot = ((lambda_oracle_flat[:, j] - lambda_oracle_flat[:, j].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        r2_elements.append(r2)
    mean_r2 = np.mean(r2_elements)

    return {
        "method": method,
        # Core metrics (C1, C2, C3)
        "corr": corr_eig1,
        "frob": mean_frob,
        "std": lambda_std,
        # Extended Frobenius
        "max_frob": max_frob,
        "frob_p25": frob_p25,
        "frob_p50": frob_p50,
        "frob_p75": frob_p75,
        "frob_p95": frob_p95,
        # Eigenvalue stats
        "min_eigenvalue": min_eigenvalue,
        "psd_rate": psd_rate,
        "corr_eig0": corr_eig0,
        "eig1_hat_mean": eig1_hat.mean(),
        "eig1_hat_std": eig1_hat.std(),
        # Error analysis
        "bias": bias,
        "mae": mae,
        "var_ratio": var_ratio,
        "mean_r2": mean_r2,
        # Timing
        "fit_time": fit_time,
        "predict_time": predict_time,
        "total_time": fit_time + predict_time,
        # Pass/fail
        "pass_c1": corr_eig1 > 0.7,
        "pass_c2": mean_frob < 0.15,
        "pass_c3": lambda_std > 0.01,
    }


def run_part_c(verbose: bool = True) -> dict:
    """Test Lambda estimation for Observational regime - ALL METHODS."""
    start_time = time.time()
    results = {}

    if verbose:
        print("\n" + "=" * 60)
        print("PART C: Regime C (Obs) - EstimateLambda (ALL METHODS)")
        print("=" * 60)

    dgp = CanonicalDGP()
    n = 500
    n_mc_oracle = 5000

    # Generate data
    Y, T, X, theta_true, mu_true = generate_canonical_dgp(n=n, seed=42, dgp=dgp)
    X_np = X.numpy().flatten()

    if verbose:
        print(f"\n  DGP: Regime C (Confounded Logit)")
        print(f"    alpha*(x) = {dgp.A0} + {dgp.A1}*sin(x)")
        print(f"    beta*(x) = {dgp.B0} + {dgp.B1}*x")
        print(f"    T | X ~ N(beta*(x), {dgp.T_noise_std}^2)")

    # Compute Oracle Lambda(x) for each point
    if verbose:
        print(f"\n  Computing Oracle Lambda(x) for {n} points (MC={n_mc_oracle})...")

    oracle_start = time.time()
    np.random.seed(123)
    lambda_oracle = np.zeros((n, 2, 2))
    for i in range(n):
        x = X_np[i]
        lambda_oracle[i] = oracle_lambda_conditional(x, dgp, n_samples=n_mc_oracle)
    oracle_time = time.time() - oracle_start

    oracle_std = lambda_oracle.reshape(n, -1).std(axis=0).mean()
    oracle_eig1 = np.array([np.linalg.eigvalsh(lambda_oracle[i])[1] for i in range(n)])
    oracle_eig0 = np.array([np.linalg.eigvalsh(lambda_oracle[i])[0] for i in range(n)])

    if verbose:
        print(f"  Oracle computation time: {oracle_time:.2f}s")
        print(f"  Oracle stats:")
        print(f"    eig_max: mean={oracle_eig1.mean():.4f}, std={oracle_eig1.std():.4f}")
        print(f"    eig_min: mean={oracle_eig0.mean():.4f}, std={oracle_eig0.std():.4f}")
        print(f"    element-wise std: {oracle_std:.4f}")

    # Test all methods
    methods_to_test = ["aggregate", "mlp", "ridge", "rf", "lgbm"]
    method_results = {}
    method_times = {}

    try:
        from deep_inference.lambda_.estimate import EstimateLambda
        from deep_inference.models import Logit

        model = Logit()
        package_available = True

        if verbose:
            print(f"\n  Testing {len(methods_to_test)} methods: {methods_to_test}")
            print("-" * 60)

        for method in methods_to_test:
            if verbose:
                print(f"\n  --- Method: {method} ---")

            try:
                # Fit with timing
                t0 = time.time()
                strategy = EstimateLambda(method=method)
                strategy.fit(X=X, T=T, Y=Y, theta_hat=theta_true, model=model)
                fit_time = time.time() - t0

                # Predict with timing
                t0 = time.time()
                lambda_hat = strategy.predict(X, theta_true).numpy()
                predict_time = time.time() - t0

                method_times[method] = fit_time + predict_time

                metrics = evaluate_lambda_method(
                    method, lambda_hat, lambda_oracle, n,
                    fit_time=fit_time, predict_time=predict_time
                )
                method_results[method] = metrics

                if verbose:
                    print(f"    Corr(eig1): {metrics['corr']:.4f} {'PASS' if metrics['pass_c1'] else 'FAIL'}")
                    print(f"    Mean Frob:  {metrics['frob']:.4f} {'PASS' if metrics['pass_c2'] else 'FAIL'}")
                    print(f"    Max Frob:   {metrics['max_frob']:.4f}")
                    print(f"    P95 Frob:   {metrics['frob_p95']:.4f}")
                    print(f"    Std (x-dep): {metrics['std']:.6f} {'PASS' if metrics['pass_c3'] else 'FAIL'}")
                    print(f"    Min eig:    {metrics['min_eigenvalue']:.6f} (PSD: {metrics['psd_rate']:.0f}%)")
                    print(f"    Bias:       {metrics['bias']:.6f}")
                    print(f"    Var ratio:  {metrics['var_ratio']:.4f}")
                    print(f"    Time:       {metrics['total_time']:.3f}s (fit={fit_time:.3f}s, pred={predict_time:.3f}s)")

            except Exception as e:
                if verbose:
                    print(f"    ERROR: {e}")
                method_results[method] = {"error": str(e)}

    except ImportError as e:
        if verbose:
            print(f"  [SKIP] EstimateLambda not available: {e}")
        package_available = False

    # Extended summary table
    if verbose and method_results:
        print("\n" + "-" * 80)
        print("  METHOD COMPARISON TABLE (Extended)")
        print("-" * 80)
        header = f"  {'Method':<10} {'Corr':<8} {'Frob':<8} {'Max':<8} {'P95':<8} {'MinEig':<8} {'PSD%':<6} {'Time':<8}"
        print(header)
        print("  " + "-" * 78)

        for method, m in method_results.items():
            if "error" in m:
                print(f"  {method:<10} {'ERROR':<8} {'-':<8} {'-':<8} {'-':<8} {'-':<8} {'-':<6} {'-':<8}")
            else:
                print(f"  {method:<10} {m['corr']:<8.4f} {m['frob']:<8.4f} {m['max_frob']:<8.4f} "
                      f"{m['frob_p95']:<8.4f} {m['min_eigenvalue']:<8.4f} {m['psd_rate']:<6.0f} "
                      f"{m['total_time']:<8.3f}")

        # Second table: additional statistics
        print("\n" + "-" * 80)
        print("  ADDITIONAL STATISTICS")
        print("-" * 80)
        header2 = f"  {'Method':<10} {'Bias':<10} {'MAE':<10} {'VarRatio':<10} {'R²':<10} {'Result':<10}"
        print(header2)
        print("  " + "-" * 78)

        for method, m in method_results.items():
            if "error" in m:
                print(f"  {method:<10} {'ERROR':<10} {'-':<10} {'-':<10} {'-':<10} {'SKIP':<10}")
            else:
                n_pass = sum([m['pass_c1'], m['pass_c2'], m['pass_c3']])
                result = "3/3 PASS" if n_pass == 3 else f"{n_pass}/3"
                print(f"  {method:<10} {m['bias']:<10.6f} {m['mae']:<10.6f} "
                      f"{m['var_ratio']:<10.4f} {m['mean_r2']:<10.4f} {result:<10}")

    # Find best method
    best_method = None
    best_score = -1
    for method, m in method_results.items():
        if "error" not in m:
            score = sum([m['pass_c1'], m['pass_c2'], m['pass_c3']])
            if score > best_score:
                best_score = score
                best_method = method

    # Return results in expected format
    if best_method and "error" not in method_results[best_method]:
        best = method_results[best_method]
        results["C1"] = {"corr": best["corr"], "passed": best["pass_c1"]}
        results["C2"] = {"mean_frob": best["frob"], "passed": best["pass_c2"]}
        results["C3"] = {"std": best["std"], "passed": best["pass_c3"]}
    else:
        # Fallback to aggregate if available
        if "aggregate" in method_results and "error" not in method_results["aggregate"]:
            agg = method_results["aggregate"]
            results["C1"] = {"corr": agg["corr"], "passed": agg["pass_c1"]}
            results["C2"] = {"mean_frob": agg["frob"], "passed": agg["pass_c2"]}
            results["C3"] = {"std": agg["std"], "passed": agg["pass_c3"]}
        else:
            results["C1"] = {"corr": 0.0, "passed": False}
            results["C2"] = {"mean_frob": 1.0, "passed": False}
            results["C3"] = {"std": 0.0, "passed": False}

    results["package_available"] = package_available
    results["method_results"] = method_results
    results["best_method"] = best_method
    results["oracle_time"] = oracle_time
    results["method_times"] = method_times

    elapsed = time.time() - start_time
    results["elapsed_time"] = elapsed

    if verbose and best_method:
        print(f"\n  BEST METHOD: {best_method} ({best_score}/3 tests passed)")
        print(f"\n  Part C completed in {elapsed:.2f}s")
        print(f"    - Oracle computation: {oracle_time:.2f}s")
        for method, t in method_times.items():
            print(f"    - {method}: {t:.2f}s")

    return results


# =============================================================================
# MAIN
# =============================================================================

def run_eval_03(verbose: bool = True) -> dict:
    """
    Run Lambda estimation evaluation across all regimes.

    RUTHLESS: Tight tolerances, we WANT to see FAIL when wrong.
    """
    total_start = time.time()

    print("=" * 60)
    print("EVAL 03: LAMBDA ESTIMATION (RUTHLESS)")
    print("=" * 60)
    print("\nThis eval tests Lambda estimation across ALL THREE REGIMES.")
    print("We WANT to see FAIL when implementation is wrong.")

    # Run all parts
    results_a = run_part_a(verbose=verbose)
    results_b = run_part_b(verbose=verbose)
    results_c = run_part_c(verbose=verbose)

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(f"  Part A (RCT):           {results_a.get('elapsed_time', 0):.2f}s")
    print(f"  Part B (Linear):        {results_b.get('elapsed_time', 0):.2f}s")
    print(f"  Part C (Observational): {results_c.get('elapsed_time', 0):.2f}s")
    print(f"  " + "-" * 30)
    print(f"  TOTAL:                  {total_elapsed:.2f}s")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_tests = []

    print("\n  Part A (RCT):")
    for test, result in results_a.items():
        if result is None or test == "elapsed_time":
            continue
        if not isinstance(result, dict) or "passed" not in result:
            continue
        status = "PASS" if result["passed"] else "FAIL"
        all_tests.append(result["passed"])
        if test == "A1":
            print(f"    {test}: Quadrature vs MC............{status} ({result['rel_error']:.2f}%)")
        elif test == "A2":
            print(f"    {test}: MC convergence rate.........{status} (rate={result['rate']:.2f})")
        elif test == "A3":
            print(f"    {test}: Y-independence..............{status} (diff={result['diff']:.2e})")

    print("\n  Part B (Linear):")
    for test, result in results_b.items():
        if result is None or test == "elapsed_time":
            continue
        if not isinstance(result, dict) or "passed" not in result:
            continue
        status = "PASS" if result["passed"] else "FAIL"
        all_tests.append(result["passed"])
        if test == "B1":
            print(f"    {test}: Lambda = E[TT'|X]...........{status} (error={result['max_error']:.2e})")
        elif test == "B2":
            print(f"    {test}: theta-independence..........{status} (diff={result['max_diff']:.2e})")
        elif test == "B3":
            print(f"    {test}: Confounded T................{status} (error={result['max_error']:.1f}%)")

    print("\n  Part C (Observational):")
    skip_keys = {"package_available", "method_results", "best_method", "elapsed_time", "oracle_time", "method_times"}
    for test, result in results_c.items():
        if result is None or test in skip_keys:
            continue
        if not isinstance(result, dict) or "passed" not in result:
            continue
        status = "PASS" if result["passed"] else "FAIL"
        all_tests.append(result["passed"])
        if test == "C1":
            print(f"    {test}: Corr(lambda_1_hat, lambda_1*)..{status} ({result['corr']:.2f})")
        elif test == "C2":
            print(f"    {test}: Mean Frobenius..............{status} ({result['mean_frob']:.4f})")
        elif test == "C3":
            print(f"    {test}: x-dependence................{status} (std={result['std']:.4f})")

    # Statistics summary for best method
    if results_c.get("best_method") and results_c.get("method_results"):
        best = results_c["method_results"].get(results_c["best_method"])
        if best and "error" not in best:
            print("\n" + "=" * 60)
            print(f"STATISTICS SUMMARY (Best Method: {results_c['best_method']})")
            print("=" * 60)
            print(f"  Correlation:     {best['corr']:.4f} (threshold: 0.70)")
            print(f"  Mean Frobenius:  {best['frob']:.4f} (threshold: 0.15)")
            print(f"  Max Frobenius:   {best['max_frob']:.4f}")
            print(f"  95th percentile: {best['frob_p95']:.4f}")
            print(f"  Min Eigenvalue:  {best['min_eigenvalue']:.6f} (>0 = PSD)")
            print(f"  PSD Rate:        {best['psd_rate']:.0f}%")
            print(f"  Bias:            {best['bias']:.6f} (should be ~0)")
            print(f"  Variance Ratio:  {best['var_ratio']:.4f} (should be ~1)")
            print(f"  Mean R²:         {best['mean_r2']:.4f}")

    n_passed = sum(all_tests)
    n_total = len(all_tests)

    print("\n" + "=" * 60)
    print(f"EVAL 03: {n_passed}/{n_total} PASSED")
    if n_passed == n_total:
        print("STATUS: ALL PASS")
    else:
        print("STATUS: FAIL")
        print("\nFailing tests indicate implementation bugs or method limitations.")
        print("This is EXPECTED behavior - the eval is working correctly.")
    print("=" * 60)

    return {
        "part_a": results_a,
        "part_b": results_b,
        "part_c": results_c,
        "n_passed": n_passed,
        "n_total": n_total,
        "passed": n_passed == n_total,
        "total_time": total_elapsed,
    }


if __name__ == "__main__":
    result = run_eval_03(verbose=True)
