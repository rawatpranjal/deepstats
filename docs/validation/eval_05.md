# Eval 05: Influence Function Assembly

Validates the complete influence function ψ assembly from Theorem 2.

## Formula

$$\psi_i = H(\theta_i) - H_\theta(\theta_i) \cdot \Lambda(x_i)^{-1} \cdot \ell_\theta(y_i, t_i, \theta_i)$$

## Configuration

| Parameter | Value |
|-----------|-------|
| n | 1000 |
| Seed | 42 |
| True μ* | 0.241 |

## Results

### Assembly Comparison

| Metric | Package | Oracle |
|--------|---------|--------|
| Mean(ψ) | 0.241 | 0.240 |
| Std(ψ) | 1.050 | 0.997 |

### Assembly Quality

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Corr(ψ̂, ψ*) | 0.995 | > 0.9 | PASS |
| Bias | 0.001 | < 0.1 | PASS |
| RMSE | 0.114 | < 0.5 | PASS |

### Inference Check

| Quantity | Value |
|----------|-------|
| True μ* | 0.241 |
| Mean(ψ_oracle) | 0.240 |
| Mean(ψ_package) | 0.241 |
| Oracle bias from true | -0.001 |
| Package bias from true | -0.000 |

## Summary

| Test | Result |
|------|--------|
| Corr(ψ̂, ψ*) > 0.9 | PASS |
| \|Bias\| < 0.1 | PASS |
| RMSE < 0.5 | PASS |
| **Overall** | **PASS** |

## Key Findings

- Package ψ correlates 0.995 with oracle ψ
- Bias is negligible (< 0.01)
- Standard deviation within 5% of oracle
- Assembly correctly combines H, H_θ, Λ⁻¹, and ℓ_θ

## Run Command

```bash
python3 -m evals.eval_05_psi 2>&1 | tee evals/reports/eval_05_$(date +%Y%m%d_%H%M%S).txt
```
