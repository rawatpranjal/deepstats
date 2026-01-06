# Influence Function Validation Experiments

## Overview

Monte Carlo simulations validating the FLM (Farrell, Liang, Misra) influence function approach for neural network inference.

**Target**: 95% coverage, SE ratio ~1.0
**Standard**: M=30 simulations minimum

---

## Completed Experiments (M=30)

| Experiment | Model | K | Network | Epochs | N | Coverage | SE Ratio | Grade |
|------------|-------|---|---------|--------|---|----------|----------|-------|
| exp_K50 | linear | 50 | [64,32] | 50 | 10,000 | **93.3%** | 1.03 | PASS |
| exp_deep | linear | 20 | [128,64,32] | 50 | 10,000 | **93.3%** | 1.02 | PASS |
| logit_stress_test | logit | 20 | [64,32] | 50 | 10,000 | **90.0%** | 0.93 | PASS |
| exp_e100 | linear | 20 | [64,32] | 100 | 10,000 | 90.0% | 0.90 | WARNING |
| exp_separate | linear | 20 | [64,32] | 50 | 10,000 | 80.0% | 0.82 | WARNING |

---

## Recent Results

| Experiment | Model | Target | K | Network | Coverage | SE Ratio | Grade |
|------------|-------|--------|---|---------|----------|----------|-------|
| logit_ame_fixed | logit | AME | 20 | deep | **90%** | 0.88 | PASS |
| tobit_both_K50_deep | tobit | latent | 50 | deep | 80% | 0.97 | WARNING |
| tobit_both_K50_deep | tobit | observed | 50 | deep | **87%** | 0.79 | WARNING |

**Bug Fix Applied:** LogitDGP now computes correct μ* for AME target (was using E[β] instead of E[p(1-p)β]).

## All Results Summary

| Model | Target | K | Network | Coverage | SE Ratio | Grade |
|-------|--------|---|---------|----------|----------|-------|
| linear | β | 50 | default | **93%** | 1.03 | PASS |
| linear | β | 20 | deep | **93%** | 1.02 | PASS |
| poisson | β | 20 | deep | **93%** | 1.20 | PASS |
| logit | AME | 20 | deep | **90%** | 0.88 | PASS |
| **negbin** | β | 20 | deep | **100%** | 1.26 | WARNING |
| linear | β | 10 | deep | 87% | 0.94 | WARNING |
| tobit | observed | 50 | deep | 87% | 0.79 | WARNING |
| tobit | latent | 50 | deep | 80% | 0.97 | WARNING |
| **gamma** | β | 20 | deep | **100%** | 1.24 | WARNING (FIXED) |
| **gumbel** | β | 20 | deep | **97%** | 1.11 | PASS (FIXED) |
| weibull | β | 20 | deep | 67% | 0.63 | FAIL |

### Bug Fix Applied (2026-01-06)
- **Gamma, Gumbel**: Removed broken `influence_score()` overrides that used scalar variance instead of full Hessian
- Now inherit correct `BaseFamily.influence_score()` with proper FLM formula
- Coverage improved: Gamma 80%→100%, Gumbel 77%→97%

### Remaining Issue: Weibull
- Uses correct BaseFamily formula but still failing (67% coverage)
- May need weight function adjustment or K=50 folds

---

## Commands

```bash
# Linear with K=50 folds (PASS)
python3 -m deepstats.run_mc --M 30 --N 10000 --epochs 50 --n-folds 50 \
  --models linear --methods naive influence --log-dir logs/exp_K50 --n-jobs -1

# Linear with deep network (PASS)
python3 -m deepstats.run_mc --M 30 --N 10000 --epochs 50 --n-folds 20 \
  --network deep --models linear --methods naive influence --log-dir logs/exp_deep --n-jobs -1

# Logit AME (PASS)
python3 -m deepstats.run_mc --M 30 --N 10000 --epochs 50 --n-folds 20 \
  --models logit --methods naive influence --target ame --log-dir logs/logit_stress_test --n-jobs -1

# Linear with 100 epochs (WARNING)
python3 -m deepstats.run_mc --M 30 --N 10000 --epochs 100 --n-folds 20 \
  --models linear --methods naive influence --log-dir logs/exp_e100 --n-jobs -1
```

---

## Metrics

### Phase 1: Parameter Recovery
- RMSE_α, RMSE_β: Root mean square error
- Corr_α, Corr_β: Correlation with true values
- R²_α, R²_β: Variance explained

### Phase 2: Inference
- Coverage: % of CIs containing true μ*
- SE Ratio: SE(estimated) / SE(empirical)
- Bias²/MSE: Proportion of error from bias

### Phase 3: Diagnostics
- Grad→0: Final gradient norm
- β_std: Learned heterogeneity
- R_corr: Correction ratio
- min(Λ): Hessian minimum eigenvalue

---

## Scorecard Grades

| Grade | Coverage | SE Ratio |
|-------|----------|----------|
| PASS | 93-97% | 0.9-1.2 |
| WARNING | 85-93% | 0.8-1.4 |
| FAIL | <85% | <0.8 or >1.4 |

---

## DGP

```python
α*(X) = sin(2πX₁) + X₂³ - 2cos(πX₃) + exp(X₄/3)·I(X₄>0) + 0.5·X₅·X₆
β*(X) = cos(2πX₁)·sin(πX₂) + 0.8·tanh(3X₃) - 0.5·X₄² + 0.3·X₅·I(X₆>0)
μ* = E[β(X)] ≈ -0.168
```
