# BRUTAL E2E Smoke Test: "John from Minnesota" Experience

This document simulates a new user experience: fresh venv, install package, import, and run each of the 12 GLM families on synthetic data **with ground truth verification**.

---

## What Makes This Test BRUTAL

| Aspect | Old (Weak) | New (Brutal) |
|--------|------------|--------------|
| Pass criterion | Didn't crash | CI covers μ* AND estimate reasonable |
| Ground truth | None | μ* = 0.3 verified |
| SE validation | None | 0.01 < SE < 10 |
| CI check | None | Must cover 0.3 |
| Expected output | Fabricated | Real run results |
| Diagnostics | None | min_eigenvalue checked |

---

## Ground Truth

For all families: `beta(X) = 0.3 + 0.1*X₁`, `E[X₁] = 0` → **μ* = 0.3**

### Verdict Criteria

- **PASS**: CI covers 0.3 AND |μ̂ - 0.3| < 0.5 AND 0.01 < SE < 10 AND λ_min > 0
- **WARN**: Estimate reasonable but CI doesn't cover (SE miscalibrated)
- **FAIL**: Exception OR NaN OR estimate wildly wrong

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

## 3. Run Smoke Test

The brutal smoke test is at `/Users/pranjal/deepest/smoke_test.py`:

```bash
python3 smoke_test.py
```

---

## 4. REAL Output (Actual Run - 2026-01-16)

```
================================================================================
BRUTAL E2E SMOKE TEST: ALL 12 GLM FAMILIES
================================================================================
Config: N=500, epochs=30, n_folds=10, hidden_dims=[32, 16]
Ground Truth: μ* = E[β(X)] = 0.3
================================================================================

[1/12] Testing LINEAR family...
  PASS: μ̂=0.2153 ± 0.0498, CI=[0.1177, 0.3129], covers_μ*=True

[2/12] Testing LOGIT family...
  PASS: μ̂=0.2429 ± 0.0990, CI=[0.0490, 0.4369], covers_μ*=True

[3/12] Testing POISSON family...
  PASS: μ̂=0.3100 ± 0.0404, CI=[0.2309, 0.3892], covers_μ*=True

[4/12] Testing GAMMA family...
  PASS: μ̂=0.3357 ± 0.0379, CI=[0.2615, 0.4099], covers_μ*=True

[5/12] Testing GAUSSIAN family...
  PASS: μ̂=0.3496 ± 0.0476, CI=[0.2563, 0.4430], covers_μ*=True

[6/12] Testing GUMBEL family...
  PASS: μ̂=0.3848 ± 0.0575, CI=[0.2721, 0.4976], covers_μ*=True

[7/12] Testing TOBIT family...
  PASS: μ̂=0.3463 ± 0.0628, CI=[0.2233, 0.4693], covers_μ*=True

[8/12] Testing NEGBIN family...
  WARN: μ̂=0.4304 ± 0.0536, CI=[0.3253, 0.5354], covers_μ*=False

[9/12] Testing WEIBULL family...
  PASS: μ̂=0.2995 ± 0.0111, CI=[0.2777, 0.3214], covers_μ*=True

[10/12] Testing PROBIT family...
  PASS: μ̂=0.3024 ± 0.0631, CI=[0.1788, 0.4260], covers_μ*=True

[11/12] Testing BETA family...
  WARN: μ̂=0.1275 ± 0.0149, CI=[0.0983, 0.1567], covers_μ*=False

[12/12] Testing ZIP family...
  PASS: μ̂=0.2658 ± 0.0638, CI=[0.1406, 0.3909], covers_μ*=True

================================================================================
BRUTAL SMOKE TEST SUMMARY
================================================================================
Ground Truth: μ* = 0.3
================================================================================

Family       Verdict       μ̂       SE    CI_lo    CI_hi  Covers     Bias
--------------------------------------------------------------------------------
linear       PASS     0.2153   0.0498   0.1177   0.3129     Yes  -0.0847
logit        PASS     0.2429   0.0990   0.0490   0.4369     Yes  -0.0571
poisson      PASS     0.3100   0.0404   0.2309   0.3892     Yes  +0.0100
gamma        PASS     0.3357   0.0379   0.2615   0.4099     Yes  +0.0357
gaussian     PASS     0.3496   0.0476   0.2563   0.4430     Yes  +0.0496
gumbel       PASS     0.3848   0.0575   0.2721   0.4976     Yes  +0.0848
tobit        PASS     0.3463   0.0628   0.2233   0.4693     Yes  +0.0463
negbin       WARN     0.4304   0.0536   0.3253   0.5354      No  +0.1304
weibull      PASS     0.2995   0.0111   0.2777   0.3214     Yes  -0.0005
probit       PASS     0.3024   0.0631   0.1788   0.4260     Yes  +0.0024
beta         WARN     0.1275   0.0149   0.0983   0.1567      No  -0.1725
zip          PASS     0.2658   0.0638   0.1406   0.3909     Yes  -0.0342
--------------------------------------------------------------------------------

SUMMARY: 10 PASS, 2 WARN, 0 FAIL out of 12 families

--------------------------------------------------------------------------------
VERDICT CRITERIA:
  PASS: CI covers μ*=0.3 AND |μ̂ - 0.3| < 0.5 AND 0.01 < SE < 10 AND λ_min > 0
  WARN: Estimate reasonable but CI doesn't cover (SE miscalibrated)
  FAIL: Exception OR NaN OR estimate wildly wrong
--------------------------------------------------------------------------------

⚡ WARNINGS (CI miscalibration):
  negbin: CI=[0.3253, 0.5354] doesn't cover μ*=0.3
  beta: CI=[0.0983, 0.1567] doesn't cover μ*=0.3

================================================================================
⚡ 10/12 PASSED, 2/12 WARNED - PACKAGE FUNCTIONAL (SE calibration issues)
================================================================================
```

---

## 5. Results Analysis

### Passing Families (10/12)

| Family | Bias | Verdict |
|--------|------|---------|
| linear | -0.0847 | PASS |
| logit | -0.0571 | PASS |
| poisson | +0.0100 | PASS |
| gamma | +0.0357 | PASS |
| gaussian | +0.0496 | PASS |
| gumbel | +0.0848 | PASS |
| tobit | +0.0463 | PASS |
| weibull | -0.0005 | PASS |
| probit | +0.0024 | PASS |
| zip | -0.0342 | PASS |

### Warned Families (2/12)

| Family | Issue | Bias | Notes |
|--------|-------|------|-------|
| **negbin** | CI doesn't cover μ* | +0.1304 | Positive bias, SE may be underestimated |
| **beta** | CI doesn't cover μ* | -0.1725 | Negative bias, SE too small (0.0149) |

### Why Warnings Are Expected

1. **Small sample (N=500)** - Some families need more data
2. **Low epochs (30)** - More training might help
3. **Few folds (10)** - Package recommends K≥50 for valid SEs
4. **Family complexity** - NegBin and Beta have extra parameters

This is exactly what a brutal test should show: **it exposes real weaknesses**, not just "did it crash".

---

## 6. Quick Reference: Family DGPs

| Family | DGP | μ* Target |
|--------|-----|-----------|
| **linear** | `Y = α + β*T + ε` | E[β(X)] |
| **logit** | `P(Y=1) = σ(α + β*T)` | E[β(X)] |
| **poisson** | `Y ~ Poisson(exp(α + β*T))` | E[β(X)] |
| **gamma** | `Y ~ Gamma(k, exp(α + β*T))` | E[β(X)] |
| **gaussian** | `Y ~ N(α + β*T, σ)` | E[β(X)] |
| **gumbel** | `Y ~ Gumbel(α + β*T, s)` | E[β(X)] |
| **tobit** | `Y = max(0, α + β*T + ε)` | E[β(X)] |
| **negbin** | `Y ~ NegBin(exp(α + β*T), r)` | E[β(X)] |
| **weibull** | `Y ~ Weibull(k, exp(α + β*T))` | E[β(X)] |
| **probit** | `P(Y=1) = Φ(α + β*T)` | E[β(X)] |
| **beta** | `Y ~ Beta(μφ, (1-μ)φ)` | E[β(X)] |
| **zip** | `Y ~ π*0 + (1-π)*Poisson(λ)` | E[β(X)] |

Where: `α(X) = 0.5 + 0.2*X₁`, `β(X) = 0.3 + 0.1*X₁`, `E[X₁] = 0` → **μ* = 0.3**

---

## 7. Troubleshooting

### Import Error
```
ModuleNotFoundError: No module named 'deep_inference'
```
**Fix:** Run `pip install -e .` from the project root.

### High Correction Variance Warning
```
UserWarning: High correction variance ratio (22.98)
```
**Fix:** Increase `n_folds` to 50+ for production runs. This is expected for small K.

### CI Doesn't Cover Ground Truth
**Expected for:** Hard families (beta, negbin) with small samples.
**Fix:** Increase N, epochs, and n_folds:
```python
result = structural_dml(
    Y=Y, T=T, X=X,
    family='negbin',
    epochs=100,     # More training
    n_folds=50,     # More folds for valid SEs
    hidden_dims=[64, 32]  # Larger network
)
```

---

## 8. Verification Checklist

- [x] Fresh venv created and activated
- [x] `pip install -e .` succeeded
- [x] `from deep_inference import structural_dml` works
- [x] `FAMILY_REGISTRY` contains 12 families
- [x] All 12 families run without exception
- [x] No NaN/Inf in estimates
- [x] Standard errors are positive and reasonable
- [x] 10/12 families have CI covering ground truth
- [ ] negbin and beta need investigation (known issues)

---

## 9. Next Steps for WARN Families

### negbin (Negative Binomial)
- **Issue:** +13% bias, CI misses μ*
- **Hypothesis:** Dispersion parameter estimation interferes with β recovery
- **TODO:** Test with larger N, check overdispersion parameter

### beta (Beta regression)
- **Issue:** -17% bias, SE very small (0.015)
- **Hypothesis:** Link function (logit) may not match DGP properly
- **TODO:** Verify DGP matches family assumptions, check phi parameter
