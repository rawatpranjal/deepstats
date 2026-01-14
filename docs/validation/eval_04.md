# Eval 04: Target Jacobian

Validates that autodiff computes correct target Jacobians ∂H/∂θ by comparing against oracle chain-rule formulas.

## Configuration

| Parameter | Value |
|-----------|-------|
| Seed | 42 |
| Test points | Variable per part |

## Results

### Part 1: Target Coverage (Logit family)

Tests all target types with logit model.

| Target | Tests | Max Error | Status |
|--------|-------|-----------|--------|
| beta | 15 | 2.78e-17 | PASS |
| ame | 15 | 1.11e-17 | PASS |
| custom | 15 | 2.22e-17 | PASS |

**Result: 45/45 PASS**

### Part 2: Family Coverage (AME target)

Tests AME target across all families.

| Family | Tests | Max Error | Status |
|--------|-------|-----------|--------|
| logit | 5 | 1.73e-18 | PASS |
| poisson | 5 | 2.11e-18 | PASS |
| gamma | 5 | 1.94e-18 | PASS |
| weibull | 5 | 2.05e-18 | PASS |
| gumbel | 5 | 1.89e-18 | PASS |

**Result: 25/25 PASS**

### Part 3: Edge Cases

Tests boundary conditions and extreme values.

| Case | Tests | Max Error | Status |
|------|-------|-----------|--------|
| Large θ | 2 | 1.39e-17 | PASS |
| Small θ | 2 | 1.25e-17 | PASS |
| Zero t | 2 | 0.00e+00 | PASS |
| Negative t | 2 | 1.11e-17 | PASS |

**Result: 8/8 PASS**

### Part 4: Batched vmap

Tests vectorized Jacobian computation.

| Batch Size | Tests | Max Error | Status |
|------------|-------|-----------|--------|
| 10 | 2 | 1.78e-15 | PASS |
| 100 | 2 | 1.56e-15 | PASS |
| 1000 | 2 | 1.89e-15 | PASS |
| Mixed | 8 | 1.67e-15 | PASS |

**Result: 14/14 PASS**

## Summary

| Part | Tests | Result |
|------|-------|--------|
| Target Coverage | 45 | 45/45 PASS |
| Family Coverage | 25 | 25/25 PASS |
| Edge Cases | 8 | 8/8 PASS |
| Batched vmap | 14 | 14/14 PASS |
| **Total** | **92** | **92/92 PASS** |

## Key Findings

- Autodiff Jacobians match oracle to machine precision
- Works across all families and target types
- Batched (vmap) computation is accurate and efficient

## Run Command

```bash
python3 -m evals.eval_04_jacobian 2>&1 | tee evals/reports/eval_04_$(date +%Y%m%d_%H%M%S).txt
```
