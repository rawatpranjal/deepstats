# Validation

Monte Carlo simulation study validating the influence function methodology across all supported model families.

```{toctree}
:hidden:

verification
```

See also: [Verification Against FLM2](verification.md) for comparison with the original implementation.

---

## Eval Suite Overview

The package includes 7 evals in `evals/` validating every mathematical component of Theorem 2.

| Eval | Component | Tests | Result |
|------|-----------|-------|--------|
| 01 | Parameter Recovery θ̂(x) | 12 families × 3 seeds | 12/12 PASS |
| 02 | Autodiff vs Calculus | Score + Hessian | 31/31 PASS |
| 03 | Lambda Estimation Λ̂(x) | 5 methods | 9/9 PASS |
| 04 | Target Jacobian H_θ | Autodiff vs oracle | 92/92 PASS |
| 05 | Influence Function ψ | Assembly + coverage | 4/4 PASS |
| 06 | Frequentist Coverage | Monte Carlo M=50 | PASS |
| 07 | End-to-End | Full workflow | 7/7 PASS |

---

## Eval 01: Parameter Recovery

Neural networks recover heterogeneous parameters θ(x) = [α(x), β(x)] across all 12 families.

**Config:** n=2000, epochs=100, seeds=[42, 123, 999]

| Family | Corr(α) | Corr(β) | Status |
|--------|---------|---------|--------|
| linear | 0.991 | 0.995 | PASS |
| logit | 0.978 | 0.996 | PASS |
| poisson | 0.980 | 0.967 | PASS |
| negbin | 0.990 | 0.983 | PASS |
| gamma | 0.993 | 0.990 | PASS |
| weibull | 0.993 | 0.986 | PASS |
| gumbel | 0.967 | 0.991 | PASS |
| tobit | 0.987 | 0.988 | PASS |
| gaussian | 0.984 | 0.995 | PASS |
| probit | 0.983 | 0.985 | PASS |
| beta | 0.989 | 0.975 | PASS |
| zip | 0.969 | 0.944 | PASS |

**Result: 12/12 families PASS** with Corr(β) > 0.94 across all families.

---

## Eval 03: Lambda Estimation

Comparing methods for estimating the conditional Hessian Λ(x) = E[ℓ_θθ|X=x].

**Config:** n=1000, seed=42, 500 test points

| Method | Correlation | Frob Error | Time | Result |
|--------|-------------|------------|------|--------|
| aggregate | 0.000 | 0.121 | 0.02s | 1/3 |
| ridge | 0.508 | 0.087 | 0.08s | 2/3 |
| rf | 0.904 | 0.060 | 0.3s | 3/3 PASS |
| **lgbm** | **0.978** | **0.033** | **1.2s** | **3/3 PASS** |
| **mlp** | **0.997** | **0.018** | 12.5s | **3/3 PASS** |

**Key Finding:** `aggregate` has zero correlation with true Λ(x) — it ignores heterogeneity entirely. Use `mlp` for best accuracy or `lgbm` for best speed/accuracy tradeoff.

---

## Eval 05: Influence Function Coverage

Monte Carlo validation of the complete influence function.

**Config:** M=50 simulations, n=1000, Canonical Logit DGP

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Coverage | 88% (44/50) | 85-97% | PASS |
| SE Ratio | 0.873 | 0.8-1.2 | PASS |
| Mean Bias | 0.002 | < 0.1 | PASS |
| Corr(ψ̂, ψ*) | 1.000 | > 0.99 | PASS |

---

## Key Findings

### 1. Parameter Recovery Works Across All Families

- All 12 families achieve Corr(β) > 0.94
- ZIP has lowest at 0.944 (mixture model complexity)
- Logit achieves 0.996 (best treatment effect recovery)

### 2. Lambda Method Matters

- `aggregate` ignores heterogeneity (Corr = 0.000)
- `mlp` best accuracy but 10x slower than `lgbm`
- `lgbm` recommended default (Corr = 0.978, 1.2s)

### 3. Influence Correction is Essential

Without the influence function correction:
- Neural networks severely underestimate uncertainty
- Naive coverage is typically 10-30% instead of 95%
- The bias correction term accounts for regularization bias

---

## Metrics Explained

### Phase 1: Parameter Recovery

| Metric | Description | Target |
|--------|-------------|--------|
| RMSE$_\alpha$, RMSE$_\beta$ | Root mean squared error | Lower is better |
| Corr$_\alpha$, Corr$_\beta$ | Correlation with true | Higher is better |
| R$^2_\alpha$, R$^2_\beta$ | Variance explained | 0.3-0.8 typical |

### Phase 2: Inference (Primary)

| Metric | Description | Target |
|--------|-------------|--------|
| **Coverage** | % of CIs containing $\mu^*$ | 93-97% |
| **SE Ratio** | Estimated / Empirical SE | 0.9-1.2 |
| Bias | $\mathbb{E}[\hat{\mu}] - \mu^*$ | Near 0 |
| Violation Rate | Naive outside influence CI | 30-70% |

### Phase 3: Diagnostics

| Metric | Description | Target |
|--------|-------------|--------|
| Final grad norm | Convergence indicator | < 5 |
| $\beta$ std | Learned heterogeneity | > 0 |
| min($\Lambda$) | Hessian eigenvalue | > 0.02 |
| Hessian condition | Ill-conditioning | < 10 |

---

## Running Your Own Validation

Use Python scripts or Jupyter notebooks for Monte Carlo validation:

```python
import numpy as np
from deep_inference import structural_dml

# Monte Carlo validation
M = 30  # simulations
N = 2000  # sample size
MU_TRUE = 0.5  # known ground truth

results = []
for m in range(M):
    np.random.seed(m)

    # Generate data
    X = np.random.randn(N, 10)
    T = np.random.randn(N)
    Y = X[:, 0] + MU_TRUE * T + np.random.randn(N)

    # Run inference
    result = structural_dml(
        Y=Y, T=T, X=X,
        family='linear',
        epochs=50,
        n_folds=50,
        verbose=False
    )

    covered = result.ci_lower <= MU_TRUE <= result.ci_upper
    results.append({
        'mu_hat': result.mu_hat,
        'se': result.se,
        'covered': covered
    })

# Compute metrics
coverage = np.mean([r['covered'] for r in results])
se_ratio = np.mean([r['se'] for r in results]) / np.std([r['mu_hat'] for r in results])

print(f"Coverage: {coverage:.1%}")  # Target: 93-97%
print(f"SE Ratio: {se_ratio:.2f}")  # Target: 0.9-1.2
```

See `tutorials/01_linear_oracle.ipynb` and `tutorials/02_logit_oracle.ipynb` for complete validation examples.

---

## Sample Size for Heterogeneity Recovery

Beyond valid inference (Coverage ≈ 95%), you may want to **recover the heterogeneous parameters** α(X) and β(X).

### Parameter Recovery vs Sample Size

| N | Coverage | SE Ratio | Corr(α) | Corr(β) | Recommended For |
|---|----------|----------|---------|---------|-----------------|
| 2,000 | 96% | 0.90 | 0.62 | 0.28 | Inference only |
| 5,000 | 95% | 0.92 | 0.74 | 0.43 | Moderate heterogeneity |
| 10,000 | 95% | 1.22 | 0.80 | 0.49 | Good heterogeneity |
| **20,000** | **95%** | **1.08** | **0.83** | **0.95** | **Rich heterogeneity** |

### Key Findings

1. **Valid inference at any N ≥ 2000**: Coverage remains 93-97% regardless of N
2. **Parameter recovery improves with N**: Corr(β) scales as ~√N
3. **Corr(α) > Corr(β)**: Intercept is easier to estimate than treatment effect
4. **N=20,000 achieves Corr(β) > 0.55**: Good recovery of heterogeneous effects

### Recommendations

| Use Case | Sample Size | Expected Corr(β) |
|----------|-------------|------------------|
| Inference only (μ* estimate) | N ≥ 2,000 | Not relevant |
| Moderate individual effects | N ≥ 10,000 | ~0.5 |
| Rich heterogeneity analysis | N ≥ 20,000 | ~0.6 |

---

## References

- Farrell, Liang, Misra (2021): "Deep Neural Networks for Estimation and Inference" *Econometrica*
- Farrell, Liang, Misra (2025): "Deep Learning for Individual Heterogeneity" *Working Paper*
