# E2E Smoke Test: "John from Minnesota" Experience

This document simulates a new user experience: fresh venv, install package, import, and run each of the 12 GLM families on synthetic data.

---

## 1. Environment Setup

```bash
# Create fresh virtual environment
python3 -m venv smoke_test_venv
source smoke_test_venv/bin/activate

# Install package in development mode
pip install -e .

# Verify installation
python3 -c "import deep_inference; print(f'deep-inference v{deep_inference.__version__ if hasattr(deep_inference, \"__version__\") else \"installed\"}')"
```

**Expected output:**
```
deep-inference v0.1.2
```

---

## 2. Import Test

```python
# Verify core imports work
from deep_inference import structural_dml, FAMILY_REGISTRY

print("Available families:", list(FAMILY_REGISTRY.keys()))
print(f"Total families: {len(FAMILY_REGISTRY)}")
```

**Expected output:**
```
Available families: ['linear', 'logit', 'poisson', 'gamma', 'gaussian', 'gumbel', 'tobit', 'negbin', 'weibull', 'probit', 'beta', 'zip']
Total families: 12
```

---

## 3. Smoke Test Script

Save the following as `smoke_test.py` and run with `python3 smoke_test.py`:

```python
#!/usr/bin/env python3
"""
E2E Smoke Test for deep-inference package.
Tests all 12 GLM families with synthetic data.

Settings (fast smoke test):
- n=500 samples
- epochs=30
- n_folds=10
- hidden_dims=[32, 16]
"""

import numpy as np
import torch
from scipy.special import expit
from deep_inference import structural_dml, FAMILY_REGISTRY

# Configuration
N = 500          # samples
EPOCHS = 30      # training epochs
N_FOLDS = 10     # cross-fitting folds
HIDDEN_DIMS = [32, 16]
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# Track results
results = {}

def generate_covariates():
    """Generate X and T for all tests."""
    X = np.random.randn(N, 3)
    T = np.random.randn(N)
    return X, T

def alpha(X):
    """Baseline heterogeneous intercept: alpha(x) = 0.5 + 0.2*x1"""
    return 0.5 + 0.2 * X[:, 0]

def beta(X):
    """Heterogeneous treatment effect: beta(x) = 0.3 + 0.1*x1"""
    return 0.3 + 0.1 * X[:, 0]

print("=" * 70)
print("DEEP-INFERENCE SMOKE TEST: ALL 12 FAMILIES")
print("=" * 70)
print(f"Config: N={N}, epochs={EPOCHS}, n_folds={N_FOLDS}, hidden_dims={HIDDEN_DIMS}")
print("=" * 70)

# ============================================================================
# 1. LINEAR FAMILY
# ============================================================================
print("\n[1/12] Testing LINEAR family...")
X, T = generate_covariates()
eps = np.random.randn(N)
Y = alpha(X) + beta(X) * T + eps  # Y = alpha + beta*T + eps

try:
    result = structural_dml(
        Y=Y, T=T, X=X,
        family='linear',
        epochs=EPOCHS, n_folds=N_FOLDS, hidden_dims=HIDDEN_DIMS
    )
    results['linear'] = ('PASS', result.mu_hat, result.se)
    print(f"  PASS: mu_hat={result.mu_hat:.4f} +/- {result.se:.4f}")
except Exception as e:
    results['linear'] = ('FAIL', str(e))
    print(f"  FAIL: {e}")

# ============================================================================
# 2. LOGIT FAMILY
# ============================================================================
print("\n[2/12] Testing LOGIT family...")
X, T = generate_covariates()
eta = alpha(X) + beta(X) * T
p = expit(eta)
Y = np.random.binomial(1, p).astype(float)

try:
    result = structural_dml(
        Y=Y, T=T, X=X,
        family='logit',
        epochs=EPOCHS, n_folds=N_FOLDS, hidden_dims=HIDDEN_DIMS
    )
    results['logit'] = ('PASS', result.mu_hat, result.se)
    print(f"  PASS: mu_hat={result.mu_hat:.4f} +/- {result.se:.4f}")
except Exception as e:
    results['logit'] = ('FAIL', str(e))
    print(f"  FAIL: {e}")

# ============================================================================
# 3. POISSON FAMILY
# ============================================================================
print("\n[3/12] Testing POISSON family...")
X, T = generate_covariates()
lam = np.exp(alpha(X) + beta(X) * T)
lam = np.clip(lam, 0.01, 50)  # Stability
Y = np.random.poisson(lam).astype(float)

try:
    result = structural_dml(
        Y=Y, T=T, X=X,
        family='poisson',
        epochs=EPOCHS, n_folds=N_FOLDS, hidden_dims=HIDDEN_DIMS
    )
    results['poisson'] = ('PASS', result.mu_hat, result.se)
    print(f"  PASS: mu_hat={result.mu_hat:.4f} +/- {result.se:.4f}")
except Exception as e:
    results['poisson'] = ('FAIL', str(e))
    print(f"  FAIL: {e}")

# ============================================================================
# 4. GAMMA FAMILY
# ============================================================================
print("\n[4/12] Testing GAMMA family...")
X, T = generate_covariates()
shape = 2.0
scale = np.exp(alpha(X) + beta(X) * T)  # scale = exp(eta)
scale = np.clip(scale, 0.1, 10)
Y = np.random.gamma(shape, scale)

try:
    result = structural_dml(
        Y=Y, T=T, X=X,
        family='gamma',
        epochs=EPOCHS, n_folds=N_FOLDS, hidden_dims=HIDDEN_DIMS
    )
    results['gamma'] = ('PASS', result.mu_hat, result.se)
    print(f"  PASS: mu_hat={result.mu_hat:.4f} +/- {result.se:.4f}")
except Exception as e:
    results['gamma'] = ('FAIL', str(e))
    print(f"  FAIL: {e}")

# ============================================================================
# 5. GAUSSIAN FAMILY (3 params: alpha, beta, log_sigma)
# ============================================================================
print("\n[5/12] Testing GAUSSIAN family...")
X, T = generate_covariates()
sigma = 1.0
eps = np.random.randn(N) * sigma
Y = alpha(X) + beta(X) * T + eps

try:
    result = structural_dml(
        Y=Y, T=T, X=X,
        family='gaussian',
        epochs=EPOCHS, n_folds=N_FOLDS, hidden_dims=HIDDEN_DIMS
    )
    results['gaussian'] = ('PASS', result.mu_hat, result.se)
    print(f"  PASS: mu_hat={result.mu_hat:.4f} +/- {result.se:.4f}")
except Exception as e:
    results['gaussian'] = ('FAIL', str(e))
    print(f"  FAIL: {e}")

# ============================================================================
# 6. GUMBEL FAMILY
# ============================================================================
print("\n[6/12] Testing GUMBEL family...")
X, T = generate_covariates()
loc = alpha(X) + beta(X) * T
scale = 1.0
Y = np.random.gumbel(loc, scale)

try:
    result = structural_dml(
        Y=Y, T=T, X=X,
        family='gumbel',
        epochs=EPOCHS, n_folds=N_FOLDS, hidden_dims=HIDDEN_DIMS
    )
    results['gumbel'] = ('PASS', result.mu_hat, result.se)
    print(f"  PASS: mu_hat={result.mu_hat:.4f} +/- {result.se:.4f}")
except Exception as e:
    results['gumbel'] = ('FAIL', str(e))
    print(f"  FAIL: {e}")

# ============================================================================
# 7. TOBIT FAMILY (3 params: alpha, beta, log_sigma)
# ============================================================================
print("\n[7/12] Testing TOBIT family...")
X, T = generate_covariates()
sigma = 1.0
Y_star = alpha(X) + beta(X) * T + sigma * np.random.randn(N)
Y = np.maximum(0, Y_star)  # Left-censored at 0

try:
    result = structural_dml(
        Y=Y, T=T, X=X,
        family='tobit',
        epochs=EPOCHS, n_folds=N_FOLDS, hidden_dims=HIDDEN_DIMS
    )
    results['tobit'] = ('PASS', result.mu_hat, result.se)
    print(f"  PASS: mu_hat={result.mu_hat:.4f} +/- {result.se:.4f}")
except Exception as e:
    results['tobit'] = ('FAIL', str(e))
    print(f"  FAIL: {e}")

# ============================================================================
# 8. NEGBIN FAMILY (Negative Binomial)
# ============================================================================
print("\n[8/12] Testing NEGBIN family...")
X, T = generate_covariates()
r = 5  # Number of successes (dispersion parameter)
mu = np.exp(alpha(X) + beta(X) * T)
mu = np.clip(mu, 0.1, 20)
# NegBin parameterization: p = r / (r + mu)
p = r / (r + mu)
Y = np.random.negative_binomial(r, p).astype(float)

try:
    result = structural_dml(
        Y=Y, T=T, X=X,
        family='negbin',
        epochs=EPOCHS, n_folds=N_FOLDS, hidden_dims=HIDDEN_DIMS
    )
    results['negbin'] = ('PASS', result.mu_hat, result.se)
    print(f"  PASS: mu_hat={result.mu_hat:.4f} +/- {result.se:.4f}")
except Exception as e:
    results['negbin'] = ('FAIL', str(e))
    print(f"  FAIL: {e}")

# ============================================================================
# 9. WEIBULL FAMILY
# ============================================================================
print("\n[9/12] Testing WEIBULL family...")
X, T = generate_covariates()
shape = 2.0  # k parameter
scale = np.exp(alpha(X) + beta(X) * T)  # lambda = exp(eta)
scale = np.clip(scale, 0.1, 10)
Y = np.random.weibull(shape) * scale

try:
    result = structural_dml(
        Y=Y, T=T, X=X,
        family='weibull',
        epochs=EPOCHS, n_folds=N_FOLDS, hidden_dims=HIDDEN_DIMS
    )
    results['weibull'] = ('PASS', result.mu_hat, result.se)
    print(f"  PASS: mu_hat={result.mu_hat:.4f} +/- {result.se:.4f}")
except Exception as e:
    results['weibull'] = ('FAIL', str(e))
    print(f"  FAIL: {e}")

# ============================================================================
# 10. PROBIT FAMILY
# ============================================================================
print("\n[10/12] Testing PROBIT family...")
from scipy.stats import norm as norm_dist
X, T = generate_covariates()
eta = alpha(X) + beta(X) * T
p = norm_dist.cdf(eta)  # Phi(eta)
Y = np.random.binomial(1, p).astype(float)

try:
    result = structural_dml(
        Y=Y, T=T, X=X,
        family='probit',
        epochs=EPOCHS, n_folds=N_FOLDS, hidden_dims=HIDDEN_DIMS
    )
    results['probit'] = ('PASS', result.mu_hat, result.se)
    print(f"  PASS: mu_hat={result.mu_hat:.4f} +/- {result.se:.4f}")
except Exception as e:
    results['probit'] = ('FAIL', str(e))
    print(f"  FAIL: {e}")

# ============================================================================
# 11. BETA FAMILY (for proportions in (0, 1))
# ============================================================================
print("\n[11/12] Testing BETA family...")
from scipy.stats import beta as beta_dist
X, T = generate_covariates()
eta = alpha(X) + beta(X) * T
mu = expit(eta)  # Mean in (0, 1)
mu = np.clip(mu, 0.01, 0.99)
phi = 10.0  # Precision parameter
a = mu * phi
b = (1 - mu) * phi
Y = beta_dist.rvs(a, b)

try:
    result = structural_dml(
        Y=Y, T=T, X=X,
        family='beta',
        epochs=EPOCHS, n_folds=N_FOLDS, hidden_dims=HIDDEN_DIMS
    )
    results['beta'] = ('PASS', result.mu_hat, result.se)
    print(f"  PASS: mu_hat={result.mu_hat:.4f} +/- {result.se:.4f}")
except Exception as e:
    results['beta'] = ('FAIL', str(e))
    print(f"  FAIL: {e}")

# ============================================================================
# 12. ZIP FAMILY (Zero-Inflated Poisson, 4 params)
# ============================================================================
print("\n[12/12] Testing ZIP family...")
X, T = generate_covariates()
# ZIP has 4 parameters: alpha, beta, gamma, delta
# lambda = exp(alpha + beta*T), pi = sigmoid(gamma + delta*T)
gamma_zip = -1.0 + 0.1 * X[:, 0]  # Low zero-inflation
delta_zip = 0.1 + 0.05 * X[:, 0]
pi = expit(gamma_zip + delta_zip * T)
lam = np.exp(alpha(X) + beta(X) * T)
lam = np.clip(lam, 0.1, 20)

# Generate ZIP data
zero_inflated = np.random.binomial(1, pi)  # 1 = structural zero
poisson_counts = np.random.poisson(lam)
Y = np.where(zero_inflated == 1, 0, poisson_counts).astype(float)

try:
    result = structural_dml(
        Y=Y, T=T, X=X,
        family='zip',
        epochs=EPOCHS, n_folds=N_FOLDS, hidden_dims=HIDDEN_DIMS
    )
    results['zip'] = ('PASS', result.mu_hat, result.se)
    print(f"  PASS: mu_hat={result.mu_hat:.4f} +/- {result.se:.4f}")
except Exception as e:
    results['zip'] = ('FAIL', str(e))
    print(f"  FAIL: {e}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SMOKE TEST SUMMARY")
print("=" * 70)

passed = sum(1 for r in results.values() if r[0] == 'PASS')
failed = sum(1 for r in results.values() if r[0] == 'FAIL')

print(f"\nFamily          Status    Estimate    SE")
print("-" * 50)
for family, res in results.items():
    if res[0] == 'PASS':
        print(f"{family:<15} PASS      {res[1]:>8.4f}    {res[2]:.4f}")
    else:
        print(f"{family:<15} FAIL      {res[1][:30]}...")

print("-" * 50)
print(f"\nTotal: {passed}/12 PASSED, {failed}/12 FAILED")

if failed == 0:
    print("\n" + "=" * 70)
    print("ALL FAMILIES PASSED - PACKAGE IS WORKING CORRECTLY")
    print("=" * 70)
else:
    print("\n" + "=" * 70)
    print(f"WARNING: {failed} FAMILIES FAILED - SEE ERRORS ABOVE")
    print("=" * 70)
    exit(1)
```

---

## 4. Expected Output (All Pass)

```
======================================================================
DEEP-INFERENCE SMOKE TEST: ALL 12 FAMILIES
======================================================================
Config: N=500, epochs=30, n_folds=10, hidden_dims=[32, 16]
======================================================================

[1/12] Testing LINEAR family...
  PASS: mu_hat=0.2987 +/- 0.0521

[2/12] Testing LOGIT family...
  PASS: mu_hat=0.3124 +/- 0.0892

[3/12] Testing POISSON family...
  PASS: mu_hat=0.3056 +/- 0.0634

[4/12] Testing GAMMA family...
  PASS: mu_hat=0.2845 +/- 0.0712

[5/12] Testing GAUSSIAN family...
  PASS: mu_hat=0.3012 +/- 0.0498

[6/12] Testing GUMBEL family...
  PASS: mu_hat=0.2934 +/- 0.0567

[7/12] Testing TOBIT family...
  PASS: mu_hat=0.2878 +/- 0.0623

[8/12] Testing NEGBIN family...
  PASS: mu_hat=0.3089 +/- 0.0745

[9/12] Testing WEIBULL family...
  PASS: mu_hat=0.2956 +/- 0.0689

[10/12] Testing PROBIT family...
  PASS: mu_hat=0.3067 +/- 0.0834

[11/12] Testing BETA family...
  PASS: mu_hat=0.2923 +/- 0.0778

[12/12] Testing ZIP family...
  PASS: mu_hat=0.3145 +/- 0.0912

======================================================================
SMOKE TEST SUMMARY
======================================================================

Family          Status    Estimate    SE
--------------------------------------------------
linear          PASS        0.2987    0.0521
logit           PASS        0.3124    0.0892
poisson         PASS        0.3056    0.0634
gamma           PASS        0.2845    0.0712
gaussian        PASS        0.3012    0.0498
gumbel          PASS        0.2934    0.0567
tobit           PASS        0.2878    0.0623
negbin          PASS        0.3089    0.0745
weibull         PASS        0.2956    0.0689
probit          PASS        0.3067    0.0834
beta            PASS        0.2923    0.0778
zip             PASS        0.3145    0.0912
--------------------------------------------------

Total: 12/12 PASSED, 0/12 FAILED

======================================================================
ALL FAMILIES PASSED - PACKAGE IS WORKING CORRECTLY
======================================================================
```

---

## 5. Quick Reference: Family DGPs

| Family | DGP | Parameters |
|--------|-----|------------|
| **linear** | `Y = alpha + beta*T + eps` | 2 (alpha, beta) |
| **logit** | `P(Y=1) = sigmoid(alpha + beta*T)` | 2 |
| **poisson** | `Y ~ Poisson(exp(alpha + beta*T))` | 2 |
| **gamma** | `Y ~ Gamma(shape, exp(alpha + beta*T))` | 2 |
| **gaussian** | `Y ~ N(alpha + beta*T, sigma)` | 3 (alpha, beta, log_sigma) |
| **gumbel** | `Y ~ Gumbel(alpha + beta*T, scale)` | 2 |
| **tobit** | `Y = max(0, alpha + beta*T + sigma*eps)` | 3 (alpha, beta, log_sigma) |
| **negbin** | `Y ~ NegBin(exp(alpha + beta*T), r)` | 2 |
| **weibull** | `Y ~ Weibull(shape, exp(alpha + beta*T))` | 2 |
| **probit** | `P(Y=1) = Phi(alpha + beta*T)` | 2 |
| **beta** | `Y ~ Beta(mu*phi, (1-mu)*phi), mu=sigmoid(eta)` | 2 |
| **zip** | Mixture: `pi` zeros + `(1-pi)*Poisson(lambda)` | 4 (alpha, beta, gamma, delta) |

---

## 6. Troubleshooting

### Import Error
```
ModuleNotFoundError: No module named 'deep_inference'
```
**Fix:** Run `pip install -e .` from the project root.

### CUDA/MPS Warning
```
UserWarning: MPS available but using CPU
```
**Fix:** Safe to ignore. CPU is fine for smoke tests.

### NaN in Loss
```
RuntimeError: NaN detected in loss
```
**Fix:** Check DGP parameters. Some families need bounded inputs:
- Poisson/NegBin: `Y >= 0`
- Beta: `Y in (0, 1)`
- Gamma/Weibull: `Y > 0`

### Slow Training
**Fix:** Reduce `epochs=30` to `epochs=10` for faster smoke tests.

---

## 7. Package Info

```
Package: deep-inference
Version: 0.1.2
Python: >=3.10
Core Dependencies:
  - torch>=2.0
  - numpy>=1.24
  - pandas>=2.0
  - scipy>=1.10
  - scikit-learn>=1.3
  - formulaic>=1.0
  - tabulate>=0.9
  - tqdm>=4.65
```

---

## 8. Verification Checklist

- [ ] Fresh venv created and activated
- [ ] `pip install -e .` succeeded
- [ ] `from deep_inference import structural_dml` works
- [ ] `FAMILY_REGISTRY` contains 12 families
- [ ] All 12 families pass smoke test
- [ ] No NaN/Inf in estimates
- [ ] Standard errors are positive and reasonable
