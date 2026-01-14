# Validation

Comprehensive eval suite validating every mathematical component of the influence function methodology.

```{toctree}
:hidden:

eval_01
eval_02
eval_03
eval_04
eval_05
eval_06
eval_07
verification
```

---

## Eval Suite Overview

The package includes 7 evals in `evals/` validating Theorem 2.

| Eval | Component | Tests | Result | Details |
|------|-----------|-------|--------|---------|
| [01](eval_01.md) | Parameter Recovery θ̂(x) | 12 families × 3 seeds | 12/12 PASS | [→](eval_01.md) |
| [02](eval_02.md) | Autodiff vs Calculus | Score + Hessian | 31/31 PASS | [→](eval_02.md) |
| [03](eval_03.md) | Lambda Estimation Λ̂(x) | 5 methods | 9/9 PASS | [→](eval_03.md) |
| [04](eval_04.md) | Target Jacobian H_θ | Autodiff vs oracle | 92/92 PASS | [→](eval_04.md) |
| [05](eval_05.md) | Influence Function ψ | Assembly + coverage | 4/4 PASS | [→](eval_05.md) |
| [06](eval_06.md) | Frequentist Coverage | Monte Carlo M=50 | PASS | [→](eval_06.md) |
| [07](eval_07.md) | End-to-End | Full workflow | 7/7 PASS | [→](eval_07.md) |

**Total: 224+ individual checks, all passing.**

---

## Quick Summary

### Eval 01: Parameter Recovery
Neural networks recover θ(x) = [α(x), β(x)] across all 12 families with Corr(β) > 0.94. [Details →](eval_01.md)

### Eval 02: Autodiff Accuracy
PyTorch autodiff matches calculus formulas to machine precision (error < 1e-14). [Details →](eval_02.md)

### Eval 03: Lambda Estimation
MLP achieves Corr=0.997 with true Λ(x); aggregate ignores heterogeneity (Corr=0.000). [Details →](eval_03.md)

### Eval 04: Target Jacobian
∂H/∂θ computed correctly for all targets and families (92/92 tests). [Details →](eval_04.md)

### Eval 05: Influence Functions
Complete ψ assembly validated with 88% coverage, SE ratio 0.87. [Details →](eval_05.md)

### Eval 06: Frequentist Coverage
Monte Carlo (M=50, n=5000) confirms valid CIs with z-scores ~ N(0,1). [Details →](eval_06.md)

### Eval 07: End-to-End
Full analyst workflow: Oracle vs Bootstrap vs NN comparison shows IF correction is essential. [Details →](eval_07.md)

---

## Running Evals

```bash
# Run all evals
python3 -m evals.run_all 2>&1 | tee evals/reports/run_all_$(date +%Y%m%d_%H%M%S).txt

# Run individual evals
python3 -m evals.eval_01_theta
python3 -m evals.eval_02_autodiff
python3 -m evals.eval_03_lambda
python3 -m evals.eval_04_jacobian
python3 -m evals.eval_05_psi
python3 -m evals.eval_06_coverage
python3 -m evals.eval_07_e2e
```

---

## References

- Farrell, Liang, Misra (2021): "Deep Neural Networks for Estimation and Inference" *Econometrica*
- Farrell, Liang, Misra (2025): "Deep Learning for Individual Heterogeneity" *Working Paper*
- [Verification Against FLM2](verification.md) - comparison with original implementation
