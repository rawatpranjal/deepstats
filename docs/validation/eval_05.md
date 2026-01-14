# Eval 05: Influence Function Coverage

Monte Carlo validation of the complete influence function assembly.

## Configuration

| Parameter | Value |
|-----------|-------|
| Simulations | M = 50 |
| Sample Size | n = 1000 |
| DGP | Canonical Logit |
| Seed | 42 |

## Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Coverage | 88% (44/50) | 85-97% | PASS |
| SE Ratio | 0.873 | 0.8-1.2 | PASS |
| Mean Bias | 0.002 | < 0.1 | PASS |
| Corr(ψ̂, ψ*) | 1.000 | > 0.99 | PASS |

**Result: 4/4 rounds PASS**

## Rounds

### Round A: Mechanical Assembly
- Validates ψ̂ = score - Λ⁻¹·H_θ·(θ̂ - θ̃) matches oracle
- Corr(ψ̂, ψ*) = 1.000

### Round B: Neyman Orthogonality
- Validates ψ is orthogonal to nuisance perturbations
- δ = 0.01: bias = 0.000415
- δ = 0.05: bias = 0.000934
- δ = 0.10: bias = 0.001479

### Round C: Variance Formula
- Validates SE = √(Var(ψ)/n)
- 95% CI covers true μ*

### Round D: Multi-Seed Coverage
- M = 50 independent simulations
- Coverage = 88% (within 85-97% target)

## Run Command

```bash
python3 -m evals.eval_05_psi 2>&1 | tee evals/reports/eval_05_$(date +%Y%m%d_%H%M%S).txt
```
