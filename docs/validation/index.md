# Validation

Comprehensive eval suite validating every mathematical component of the influence function methodology.

```{toctree}
:hidden:

eval_01
eval_03
eval_05
verification
```

---

## Eval Suite Overview

The package includes 7 evals in `evals/` validating Theorem 2.

| Eval | Component | Tests | Result |
|------|-----------|-------|--------|
| [01](eval_01.md) | Parameter Recovery θ̂(x) | 12 families × 3 seeds | 12/12 PASS |
| 02 | Autodiff vs Calculus | Score + Hessian | 31/31 PASS |
| [03](eval_03.md) | Lambda Estimation Λ̂(x) | 5 methods | 9/9 PASS |
| 04 | Target Jacobian H_θ | Autodiff vs oracle | 92/92 PASS |
| [05](eval_05.md) | Influence Function ψ | Assembly + coverage | 4/4 PASS |
| 06 | Frequentist Coverage | Monte Carlo M=50 | PASS |
| 07 | End-to-End | Full workflow | 7/7 PASS |

---

## Key Findings

### 1. Parameter Recovery Works Across All Families
All 12 families achieve Corr(β) > 0.94. [Details →](eval_01.md)

### 2. Lambda Method Matters
`aggregate` ignores heterogeneity (Corr = 0.000). Use `mlp` or `lgbm`. [Details →](eval_03.md)

### 3. Valid Coverage
Monte Carlo validation shows 88% coverage with SE ratio 0.873. [Details →](eval_05.md)

---

## Running Evals

```bash
# Run all evals
python3 -m evals.run_all 2>&1 | tee evals/reports/run_all_$(date +%Y%m%d_%H%M%S).txt

# Run individual eval
python3 -m evals.eval_01_theta
python3 -m evals.eval_03_lambda
python3 -m evals.eval_05_psi
```

---

## References

- Farrell, Liang, Misra (2021): "Deep Neural Networks for Estimation and Inference" *Econometrica*
- Farrell, Liang, Misra (2025): "Deep Learning for Individual Heterogeneity" *Working Paper*
- [Verification Against FLM2](verification.md) - comparison with original implementation
