# Eval 03: Lambda Estimation

Validates Lambda estimation Λ(x) = E[ℓ_θθ | X=x] across all three regimes.

## Configuration

| Parameter | Value |
|-----------|-------|
| n | 5000 |
| Oracle MC | 5000 samples |
| Methods | aggregate, mlp, ridge, rf, lgbm |

## Results by Regime

### Part A: Regime A (RCT) - ComputeLambda

| Test | Description | Result |
|------|-------------|--------|
| A1 | Quadrature vs MC | PASS (0.03% error) |
| A2 | MC convergence rate | PASS (rate=0.43) |
| A3 | Y-independence | PASS (diff=0.00) |
| A4 | Package integration | PASS (0.21% error) |

**Part A: 4/4 PASS**

### Part B: Regime B (Linear) - AnalyticLambda

| Test | Description | Result |
|------|-------------|--------|
| B1 | Lambda = E[TT'|X] | PASS (error=0.00) |
| B2 | theta-independence | PASS (diff=0.00) |
| B3 | Confounded T | PASS (4.6% error) |
| B4 | Package integration | PASS (3.4% error) |

**Part B: 4/4 PASS**

### Part C: Regime C (Observational) - EstimateLambda

| Method | Corr(λ₁) | Mean Frob | Min Eig | PSD% | Result |
|--------|----------|-----------|---------|------|--------|
| aggregate | 0.000 | 0.121 | 0.041 | 100% | 1/3 |
| **mlp** | **0.997** | **0.018** | 0.000 | 100% | **3/3 PASS** |
| ridge | 0.508 | 0.087 | 0.000 | 100% | 2/3 |
| rf | 0.904 | 0.060 | 0.000 | 100% | 3/3 PASS |
| lgbm | 0.978 | 0.033 | 0.000 | 100% | 3/3 PASS |

**Best Method: MLP** (Corr=0.997, lowest Frobenius error)

## Summary

| Part | Tests | Result |
|------|-------|--------|
| Part A (RCT) | 4 | 4/4 PASS |
| Part B (Linear) | 4 | 4/4 PASS |
| Part C (Observational) | 3 | 3/3 PASS |
| **Total** | **11** | **11/11 PASS** |

## Key Findings

- **Regime A**: ComputeLambda works when treatment is randomized (Y-independent Hessian)
- **Regime B**: AnalyticLambda = E[TT'|X] is exact for linear models
- **Regime C**: MLP achieves Corr=0.997 with oracle; aggregate ignores heterogeneity (Corr=0.000)

## Run Command

```bash
python3 -m evals.eval_03_lambda 2>&1 | tee evals/reports/eval_03_$(date +%Y%m%d_%H%M%S).txt
```
