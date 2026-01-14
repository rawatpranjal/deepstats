# Eval 01: Parameter Recovery

Neural networks recover heterogeneous parameters θ(x) = [α(x), β(x)] across all 12 families.

## Configuration

| Parameter | Value |
|-----------|-------|
| Sample Size | n = 2000 |
| Epochs | 100 |
| Seeds | [42, 123, 999] |
| Network | [64, 32] |

## Results

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

## Key Findings

- All 12 families achieve high correlation with true parameters
- ZIP has lowest Corr(β) at 0.944 (mixture model complexity)
- Logit achieves highest Corr(β) at 0.996
- Corr(α) generally higher than Corr(β) (intercept easier than treatment effect)

## Run Command

```bash
python3 -m evals.eval_01_theta 2>&1 | tee evals/reports/eval_01_$(date +%Y%m%d_%H%M%S).txt
```
