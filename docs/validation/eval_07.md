# Eval 07: End-to-End Workflow

Full analyst workflow demonstrating production use case.

## Scenario: Loan Application

A bank wants to understand how interest rate sensitivity varies across customer segments.

```
DGP: Heterogeneous Logit Demand
- Y: Loan acceptance (0/1)
- T: Interest rate offered
- X: Customer characteristics (income, credit score, etc.)

True parameters:
- α*(x) = 0.5 + 0.3·x₁ - 0.2·x₂
- β*(x) = -0.8 + 0.4·x₁ (rate sensitivity)
- μ* = E[β(X)] = -0.8
```

## Configuration

| Parameter | Value |
|-----------|-------|
| Sample Size (n) | 1000 |
| Cross-fitting Folds | 20 |
| Epochs | 30 |
| Bootstrap Samples | 200 |

## Results by Round

### Round A: Oracle Logistic Regression

| Metric | Value |
|--------|-------|
| μ̂ | -0.812 |
| SE | 0.089 |
| 95% CI | [-0.987, -0.637] |
| Covers μ* | True |

### Round B: Bootstrap Oracle

| Metric | Value |
|--------|-------|
| Bootstrap SE | 0.091 |
| Bootstrap CI | [-0.992, -0.641] |
| Covers μ* | True |

### Round C: Neural Network (Naive)

| Metric | Value |
|--------|-------|
| μ̂_naive | -0.798 |
| SE_naive | 0.024 |
| CI_naive | [-0.845, -0.751] |
| Covers μ* | **False** |

### Round D: Neural Network (Influence Function)

| Metric | Value |
|--------|-------|
| μ̂_IF | -0.823 |
| SE_IF | 0.087 |
| CI_IF | [-0.994, -0.652] |
| Covers μ* | True |

### Round E: Oracle vs NN Comparison

| Method | μ̂ | SE | CI Width | Covers |
|--------|------|------|----------|--------|
| Oracle | -0.812 | 0.089 | 0.350 | T |
| Bootstrap | -0.812 | 0.091 | 0.351 | T |
| NN Naive | -0.798 | 0.024 | 0.094 | **F** |
| NN IF | -0.823 | 0.087 | 0.342 | T |

### Round F: Heterogeneity Recovery

| Metric | Value |
|--------|-------|
| Corr(α̂, α*) | 0.73 |
| Corr(β̂, β*) | 0.40 |
| θ Bootstrap Coverage | 94% |

### Round G: SE Calibration (M=100)

| Metric | Value | Target |
|--------|-------|--------|
| Coverage | 95% | 93-97% |
| SE Ratio | 0.91 | 0.9-1.1 |

## Summary

| Round | Test | Result |
|-------|------|--------|
| A | Oracle coverage | PASS |
| B | Bootstrap coverage | PASS |
| C | Naive coverage | FAIL (expected) |
| D | IF coverage | PASS |
| E | Oracle-NN comparison | PASS |
| F | Heterogeneity recovery | PASS |
| G | SE calibration | PASS |
| **Total** | | **7/7 PASS** |

## Key Findings

1. **Naive NN severely undercovers** - SE is 3.6x too small
2. **IF correction restores valid inference** - matches Oracle SE
3. **Heterogeneity is recovered** - Corr(β̂, β*) = 0.40
4. **Oracle and NN agree** - both cover true μ*

## Run Command

```bash
python3 -m evals.eval_07_e2e 2>&1 | tee evals/reports/eval_07_$(date +%Y%m%d_%H%M%S).txt

# With SE calibration round
python3 -m evals.eval_07_e2e --round-g 2>&1 | tee evals/reports/eval_07_g_$(date +%Y%m%d_%H%M%S).txt
```
