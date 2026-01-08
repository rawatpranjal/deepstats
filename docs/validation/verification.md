# Verification Against FLM2

This page documents how `deepstats` validates against the original implementation by Farrell, Liang, and Misra.

---

## Original Implementation

**Repository:** [maxhfarrell/FLM2](https://github.com/maxhfarrell/FLM2)

The FLM2 repository contains replication code for:
- Farrell, Liang, Misra (2021): "Deep Neural Networks for Estimation and Inference" *Econometrica*
- Farrell, Liang, Misra (2025): "Deep Learning for Individual Heterogeneity" *Working Paper*

---

## Implementation Comparison

| Component | FLM2 (R) | deepstats (Python) |
|-----------|----------|-------------------|
| Framework | R + PyTorch via reticulate | PyTorch native |
| Cross-fitting folds | K=50 | K=50 |
| Lambda regularization | λ=1e-8 | λ=1e-8 |
| Optimizer | Adam, lr=0.01 | Adam, lr=0.01 |
| Architecture | 2 layers, 10-20 units | 2-3 layers, 32-64 units |
| Epochs | 5000 | 100-500 |

### Influence Function Formula

Both implementations compute:

$$\psi_i = \beta(X_i) - \Lambda^{-1} \nabla \ell_i \cdot e_\beta$$

Where:
- $\Lambda = \mathbb{E}[W_i \tilde{T}_i \otimes \tilde{T}_i]$ (Hessian)
- $\nabla \ell_i = -r_i \cdot [1, \tilde{T}_i]$ (score)
- $e_\beta = [0, 1]$ (targeting vector)

---

## Validation Results

### Monte Carlo Study (M=100, N=20,000, K=50)

| Metric | FLM Target | deepstats | Status |
|--------|------------|-----------|--------|
| Coverage | 93-97% | 95% | PASS |
| SE Ratio | 0.9-1.2 | 1.08 | PASS |
| Bias | ~0 | -0.001 | PASS |

### Parameter Recovery

| Metric | Value |
|--------|-------|
| Corr(β) | 0.953 ± 0.004 |
| Corr(α) | 0.830 ± 0.005 |
| RMSE(β) | 0.105 ± 0.004 |

### Diagnostics

| Check | Result |
|-------|--------|
| Min eigenvalue(Λ) | 1.81 (stable) |
| Regularization rate | 0.0% |
| Naive coverage | 8% (confirms IF needed) |

---

## Key Alignment Points

1. **Cross-fitting structure**: 50-fold splitting ensures each observation's inference uses a model trained without it

2. **Hessian regularization**: Ridge penalty λ=1e-8 prevents numerical instability in Λ⁻¹

3. **Theorem 1 compliance**: Influence function formula matches the asymptotic expansion in FLM (2021)

4. **Coverage validation**: 95% coverage confirms valid confidence intervals

---

## Alternative Implementations

Other implementations of the FLM framework:

- [PopovicMilica/causal_nets](https://github.com/PopovicMilica/causal_nets) - HTE and propensity scores
- [rmmomin/causal-ml-auto-inference](https://github.com/rmmomin/causal-ml-auto-inference) - Causal ML framework

---

## References

- Farrell, M.H., Liang, T., Misra, S. (2021). "Deep Neural Networks for Estimation and Inference." *Econometrica*, 89(1), 181-213.
- Farrell, M.H., Liang, T., Misra, S. (2025). "Deep Learning for Individual Heterogeneity: An Automatic Inference Framework." *Working Paper*.
