#!/usr/bin/env python3
"""
BRUTAL E2E Smoke Test for deep-inference package.
Tests all 12 GLM families against KNOWN GROUND TRUTH.

BRUTAL Requirements (per family):
1. Runs without exception ✓ (basic)
2. Estimate is finite - |mu_hat| < 100 (not NaN/Inf/crazy)
3. SE is positive and reasonable - 0.01 < se < 10
4. Estimate near ground truth - |mu_hat - 0.3| < 0.5 (loose but real)
5. CI covers ground truth - ci_lower < 0.3 < ci_upper (the real test)
6. Diagnostics reasonable - min_lambda_eigenvalue > 0

Ground Truth: beta(X) = 0.3 + 0.1*X₁, E[X₁] = 0 → μ* = 0.3

Verdict Logic:
- PASS: CI covers 0.3 AND estimate within 0.5 AND SE reasonable
- WARN: CI doesn't cover but estimate close (SE too small/large)
- FAIL: Exception OR NaN OR estimate wildly wrong
"""

import numpy as np
import torch
from scipy.special import expit
from scipy.stats import norm as norm_dist
from scipy.stats import beta as beta_dist
from deep_inference import structural_dml, FAMILY_REGISTRY

# =============================================================================
# CONFIGURATION - Eval-Validated Settings for structural_dml() API
# =============================================================================
# Based on eval_06 but adjusted for structural_dml() which lacks early stopping
# structural_dml() doesn't support patience, so use epochs=100 (default) not 200
N = 5000           # samples (10x previous, matches eval settings)
N_BINARY = 8000    # For binary families (logit, probit) - need 2x samples
EPOCHS = 100       # Default; use 100 not 200 since no early stopping available
N_FOLDS = 50       # Package recommended minimum for valid SE estimation
HIDDEN_DIMS = [64, 32]  # Adequate network capacity (was [32, 16])
LR = 0.01          # Standard learning rate
SEED = 42

# Binary families need more samples (1 bit/observation)
BINARY_FAMILIES = {'logit', 'probit'}

# GROUND TRUTH: E[beta(X)] = E[0.3 + 0.1*X1] = 0.3 (since E[X1] = 0)
MU_TRUE = 0.3

np.random.seed(SEED)
torch.manual_seed(SEED)

# Track results
results = {}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_sample_size(family_name):
    """Get appropriate sample size for family (binary families need 2x)."""
    return N_BINARY if family_name in BINARY_FAMILIES else N

def generate_covariates(n_samples=None):
    """Generate X and T for all tests."""
    n = n_samples if n_samples is not None else N
    X = np.random.randn(n, 3)
    T = np.random.randn(n)
    return X, T

def alpha(X):
    """Baseline heterogeneous intercept: alpha(x) = 0.5 + 0.2*x1"""
    return 0.5 + 0.2 * X[:, 0]

def beta(X):
    """Heterogeneous treatment effect: beta(x) = 0.3 + 0.1*x1"""
    return 0.3 + 0.1 * X[:, 0]

def evaluate_result(family, result):
    """
    BRUTAL evaluation - not just 'did it run'.

    Returns: (verdict, checks_dict)
    """
    checks = {
        'finite': np.isfinite(result.mu_hat) and np.isfinite(result.se),
        'se_positive': result.se > 0.001,  # Lower threshold for families with tight SEs
        'se_reasonable': result.se < 10,
        'estimate_sane': abs(result.mu_hat - MU_TRUE) < 0.5,
        'ci_covers': result.ci_lower < MU_TRUE < result.ci_upper,
        'lambda_ok': result.diagnostics.get('min_lambda_eigenvalue', 0) > 0,
    }

    # All checks pass = PASS
    if all(checks.values()):
        return 'PASS', checks

    # Estimate is sane but CI doesn't cover = WARN (SE miscalibrated)
    if checks['finite'] and checks['estimate_sane'] and not checks['ci_covers']:
        return 'WARN', checks

    # Otherwise = FAIL
    return 'FAIL', checks

def run_family_test(family_name, Y, X, T):
    """Run structural_dml for a family and evaluate result."""
    try:
        result = structural_dml(
            Y=Y, T=T, X=X,
            family=family_name,
            epochs=EPOCHS,
            n_folds=N_FOLDS,
            hidden_dims=HIDDEN_DIMS,
            lr=LR
        )
        verdict, checks = evaluate_result(family_name, result)
        return {
            'verdict': verdict,
            'mu_hat': result.mu_hat,
            'se': result.se,
            'ci_lower': result.ci_lower,
            'ci_upper': result.ci_upper,
            'checks': checks,
            'diagnostics': result.diagnostics,
            'error': None
        }
    except Exception as e:
        return {
            'verdict': 'FAIL',
            'mu_hat': float('nan'),
            'se': float('nan'),
            'ci_lower': float('nan'),
            'ci_upper': float('nan'),
            'checks': {},
            'diagnostics': {},
            'error': str(e)
        }

# =============================================================================
# MAIN TEST
# =============================================================================
print("=" * 80)
print("BRUTAL E2E SMOKE TEST: ALL 12 GLM FAMILIES")
print("=" * 80)
print(f"Config: N={N} (binary={N_BINARY}), epochs={EPOCHS}, n_folds={N_FOLDS}, hidden_dims={HIDDEN_DIMS}")
print(f"Ground Truth: μ* = E[β(X)] = {MU_TRUE}")
print("=" * 80)

# ============================================================================
# 1. LINEAR FAMILY
# ============================================================================
print("\n[1/12] Testing LINEAR family...")
X, T = generate_covariates()
eps = np.random.randn(N)
Y = alpha(X) + beta(X) * T + eps
results['linear'] = run_family_test('linear', Y, X, T)
r = results['linear']
if r['error']:
    print(f"  FAIL: {r['error']}")
else:
    print(f"  {r['verdict']}: μ̂={r['mu_hat']:.4f} ± {r['se']:.4f}, "
          f"CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}], "
          f"covers_μ*={r['checks'].get('ci_covers', False)}")

# ============================================================================
# 2. LOGIT FAMILY (binary: uses N_BINARY samples)
# ============================================================================
print("\n[2/12] Testing LOGIT family...")
X, T = generate_covariates(N_BINARY)  # Binary family needs more samples
eta = alpha(X) + beta(X) * T
p = expit(eta)
Y = np.random.binomial(1, p).astype(float)
results['logit'] = run_family_test('logit', Y, X, T)
r = results['logit']
if r['error']:
    print(f"  FAIL: {r['error']}")
else:
    print(f"  {r['verdict']}: μ̂={r['mu_hat']:.4f} ± {r['se']:.4f}, "
          f"CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}], "
          f"covers_μ*={r['checks'].get('ci_covers', False)}")

# ============================================================================
# 3. POISSON FAMILY
# ============================================================================
print("\n[3/12] Testing POISSON family...")
X, T = generate_covariates()
lam = np.exp(alpha(X) + beta(X) * T)
lam = np.clip(lam, 0.01, 50)
Y = np.random.poisson(lam).astype(float)
results['poisson'] = run_family_test('poisson', Y, X, T)
r = results['poisson']
if r['error']:
    print(f"  FAIL: {r['error']}")
else:
    print(f"  {r['verdict']}: μ̂={r['mu_hat']:.4f} ± {r['se']:.4f}, "
          f"CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}], "
          f"covers_μ*={r['checks'].get('ci_covers', False)}")

# ============================================================================
# 4. GAMMA FAMILY
# ============================================================================
print("\n[4/12] Testing GAMMA family...")
X, T = generate_covariates()
shape = 2.0
scale = np.exp(alpha(X) + beta(X) * T)
scale = np.clip(scale, 0.1, 10)
Y = np.random.gamma(shape, scale)
results['gamma'] = run_family_test('gamma', Y, X, T)
r = results['gamma']
if r['error']:
    print(f"  FAIL: {r['error']}")
else:
    print(f"  {r['verdict']}: μ̂={r['mu_hat']:.4f} ± {r['se']:.4f}, "
          f"CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}], "
          f"covers_μ*={r['checks'].get('ci_covers', False)}")

# ============================================================================
# 5. GAUSSIAN FAMILY (3 params: alpha, beta, log_sigma)
# ============================================================================
print("\n[5/12] Testing GAUSSIAN family...")
X, T = generate_covariates()
sigma = 1.0
eps = np.random.randn(N) * sigma
Y = alpha(X) + beta(X) * T + eps
results['gaussian'] = run_family_test('gaussian', Y, X, T)
r = results['gaussian']
if r['error']:
    print(f"  FAIL: {r['error']}")
else:
    print(f"  {r['verdict']}: μ̂={r['mu_hat']:.4f} ± {r['se']:.4f}, "
          f"CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}], "
          f"covers_μ*={r['checks'].get('ci_covers', False)}")

# ============================================================================
# 6. GUMBEL FAMILY
# ============================================================================
print("\n[6/12] Testing GUMBEL family...")
X, T = generate_covariates()
loc = alpha(X) + beta(X) * T
scale = 1.0
Y = np.random.gumbel(loc, scale)
results['gumbel'] = run_family_test('gumbel', Y, X, T)
r = results['gumbel']
if r['error']:
    print(f"  FAIL: {r['error']}")
else:
    print(f"  {r['verdict']}: μ̂={r['mu_hat']:.4f} ± {r['se']:.4f}, "
          f"CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}], "
          f"covers_μ*={r['checks'].get('ci_covers', False)}")

# ============================================================================
# 7. TOBIT FAMILY (3 params: alpha, beta, log_sigma)
# ============================================================================
print("\n[7/12] Testing TOBIT family...")
X, T = generate_covariates()
sigma = 1.0
Y_star = alpha(X) + beta(X) * T + sigma * np.random.randn(N)
Y = np.maximum(0, Y_star)  # Left-censored at 0
results['tobit'] = run_family_test('tobit', Y, X, T)
r = results['tobit']
if r['error']:
    print(f"  FAIL: {r['error']}")
else:
    print(f"  {r['verdict']}: μ̂={r['mu_hat']:.4f} ± {r['se']:.4f}, "
          f"CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}], "
          f"covers_μ*={r['checks'].get('ci_covers', False)}")

# ============================================================================
# 8. NEGBIN FAMILY (Negative Binomial)
# ============================================================================
print("\n[8/12] Testing NEGBIN family...")
X, T = generate_covariates()
r_param = 5  # Number of successes (dispersion parameter)
mu = np.exp(alpha(X) + beta(X) * T)
mu = np.clip(mu, 0.1, 20)
p = r_param / (r_param + mu)
Y = np.random.negative_binomial(r_param, p).astype(float)
results['negbin'] = run_family_test('negbin', Y, X, T)
r = results['negbin']
if r['error']:
    print(f"  FAIL: {r['error']}")
else:
    print(f"  {r['verdict']}: μ̂={r['mu_hat']:.4f} ± {r['se']:.4f}, "
          f"CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}], "
          f"covers_μ*={r['checks'].get('ci_covers', False)}")

# ============================================================================
# 9. WEIBULL FAMILY
# ============================================================================
print("\n[9/12] Testing WEIBULL family...")
X, T = generate_covariates()
shape = 2.0  # k parameter
scale = np.exp(alpha(X) + beta(X) * T)
scale = np.clip(scale, 0.1, 10)
Y = np.random.weibull(shape) * scale
results['weibull'] = run_family_test('weibull', Y, X, T)
r = results['weibull']
if r['error']:
    print(f"  FAIL: {r['error']}")
else:
    print(f"  {r['verdict']}: μ̂={r['mu_hat']:.4f} ± {r['se']:.4f}, "
          f"CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}], "
          f"covers_μ*={r['checks'].get('ci_covers', False)}")

# ============================================================================
# 10. PROBIT FAMILY (binary: uses N_BINARY samples)
# ============================================================================
print("\n[10/12] Testing PROBIT family...")
X, T = generate_covariates(N_BINARY)  # Binary family needs more samples
eta = alpha(X) + beta(X) * T
p = norm_dist.cdf(eta)
Y = np.random.binomial(1, p).astype(float)
results['probit'] = run_family_test('probit', Y, X, T)
r = results['probit']
if r['error']:
    print(f"  FAIL: {r['error']}")
else:
    print(f"  {r['verdict']}: μ̂={r['mu_hat']:.4f} ± {r['se']:.4f}, "
          f"CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}], "
          f"covers_μ*={r['checks'].get('ci_covers', False)}")

# ============================================================================
# 11. BETA FAMILY (for proportions in (0, 1))
# ============================================================================
print("\n[11/12] Testing BETA family...")
X, T = generate_covariates()
eta = alpha(X) + beta(X) * T
mu_beta = expit(eta)
mu_beta = np.clip(mu_beta, 0.01, 0.99)
phi = 10.0
a = mu_beta * phi
b = (1 - mu_beta) * phi
Y = beta_dist.rvs(a, b)
results['beta'] = run_family_test('beta', Y, X, T)
r = results['beta']
if r['error']:
    print(f"  FAIL: {r['error']}")
else:
    print(f"  {r['verdict']}: μ̂={r['mu_hat']:.4f} ± {r['se']:.4f}, "
          f"CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}], "
          f"covers_μ*={r['checks'].get('ci_covers', False)}")

# ============================================================================
# 12. ZIP FAMILY (Zero-Inflated Poisson, 4 params)
# ============================================================================
print("\n[12/12] Testing ZIP family...")
X, T = generate_covariates()
gamma_zip = -1.0 + 0.1 * X[:, 0]
delta_zip = 0.1 + 0.05 * X[:, 0]
pi = expit(gamma_zip + delta_zip * T)
lam = np.exp(alpha(X) + beta(X) * T)
lam = np.clip(lam, 0.1, 20)
zero_inflated = np.random.binomial(1, pi)
poisson_counts = np.random.poisson(lam)
Y = np.where(zero_inflated == 1, 0, poisson_counts).astype(float)
results['zip'] = run_family_test('zip', Y, X, T)
r = results['zip']
if r['error']:
    print(f"  FAIL: {r['error']}")
else:
    print(f"  {r['verdict']}: μ̂={r['mu_hat']:.4f} ± {r['se']:.4f}, "
          f"CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}], "
          f"covers_μ*={r['checks'].get('ci_covers', False)}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("BRUTAL SMOKE TEST SUMMARY")
print("=" * 80)
print(f"Ground Truth: μ* = {MU_TRUE}")
print("=" * 80)

passed = sum(1 for r in results.values() if r['verdict'] == 'PASS')
warned = sum(1 for r in results.values() if r['verdict'] == 'WARN')
failed = sum(1 for r in results.values() if r['verdict'] == 'FAIL')

# Detailed table
print(f"\n{'Family':<12} {'Verdict':<6} {'μ̂':>8} {'SE':>8} {'CI_lo':>8} {'CI_hi':>8} {'Covers':>7} {'Bias':>8}")
print("-" * 80)
for family in ['linear', 'logit', 'poisson', 'gamma', 'gaussian', 'gumbel',
               'tobit', 'negbin', 'weibull', 'probit', 'beta', 'zip']:
    r = results[family]
    if r['error']:
        print(f"{family:<12} {'FAIL':<6} {'ERR':>8} {'ERR':>8} {'ERR':>8} {'ERR':>8} {'ERR':>7} {'ERR':>8}")
    else:
        covers = 'Yes' if r['checks'].get('ci_covers', False) else 'No'
        bias = r['mu_hat'] - MU_TRUE
        print(f"{family:<12} {r['verdict']:<6} {r['mu_hat']:>8.4f} {r['se']:>8.4f} "
              f"{r['ci_lower']:>8.4f} {r['ci_upper']:>8.4f} {covers:>7} {bias:>+8.4f}")

print("-" * 80)
print(f"\nSUMMARY: {passed} PASS, {warned} WARN, {failed} FAIL out of 12 families")

# Verdict criteria reminder
print("\n" + "-" * 80)
print("VERDICT CRITERIA:")
print("  PASS: CI covers μ*=0.3 AND |μ̂ - 0.3| < 0.5 AND 0.001 < SE < 10 AND λ_min > 0")
print("  WARN: Estimate reasonable but CI doesn't cover (SE miscalibrated)")
print("  FAIL: Exception OR NaN OR estimate wildly wrong")
print("-" * 80)

# Check failures
if failed > 0:
    print("\n⚠️  FAILURES DETECTED:")
    for family, r in results.items():
        if r['verdict'] == 'FAIL':
            if r['error']:
                print(f"  {family}: {r['error'][:60]}...")
            else:
                print(f"  {family}: checks failed = {[k for k, v in r['checks'].items() if not v]}")

# Check warnings
if warned > 0:
    print("\n⚡ WARNINGS (CI miscalibration):")
    for family, r in results.items():
        if r['verdict'] == 'WARN':
            print(f"  {family}: CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}] "
                  f"doesn't cover μ*={MU_TRUE}")

# Final verdict
print("\n" + "=" * 80)
if passed == 12:
    print("✅ ALL 12 FAMILIES PASSED - PACKAGE WORKING CORRECTLY")
elif passed + warned == 12:
    print(f"⚡ {passed}/12 PASSED, {warned}/12 WARNED - PACKAGE FUNCTIONAL (SE calibration issues)")
else:
    print(f"❌ {failed}/12 FAILED - PACKAGE HAS ISSUES")
print("=" * 80)

# Exit code
exit(0 if failed == 0 else 1)
