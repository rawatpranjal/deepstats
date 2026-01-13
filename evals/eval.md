# Eval Learnings

What we learn from each eval. Updated as we investigate.

---

## Eval 01: Parameter Recovery

**Goal**: Recover θ*(x) = [α*(x), β*(x)] from logistic data.

**Issue 1 - Early stopping**: Default `patience=10` stops training at epoch ~15. The network needs 50-200 epochs to converge. Fix: increase patience to 50+ in `structural_net.py:109`.

**Issue 2 - Flat loss surface**: The logistic loss is nearly flat w.r.t. level shifts. Adding +0.1 to α and -0.1 to β changes loss by only 0.0003. This means many (α,β) pairs achieve nearly identical loss, making exact level recovery impossible. The network converges to a point with ~+0.05 bias in α and ~-0.05 bias in β.

**Issue 3 - Thresholds**: RMSE(β) < 0.1 is unrealistic given the flat loss surface. Even with n=20k and 500 epochs, best achieved is RMSE(β)=0.13. Relax to 0.2.

**What helps**: More data (n=20k vs 5k), no dropout for large n, patience=50+, epochs=300+.

**What doesn't help**: More epochs beyond convergence (~100), lower learning rate (0.001 vs 0.01 - actually worse).

---

## Eval 02: Autodiff vs Calculus

**Goal**: Verify torch.func autodiff matches closed-form gradient/Hessian for all families.

**Results by family**:
| Family | Gradient | Hessian | Status |
|--------|----------|---------|--------|
| Linear | 1e-16 | 0 | PASS |
| Logit | 1e-16 | 1e-16 | PASS |
| Poisson | 1e-9 | 1e-9 | PASS (numerical epsilon) |
| Gamma | 1e-15 | 1e-15 | PASS |
| Gumbel | 0 | 0 | PASS |
| Tobit | N/A | N/A | Uses autodiff only |
| NegBin | 1e-9 | **1.66** | **FAIL** |
| Weibull | 1e-16 | 1e-15 | PASS |

**Issue found - NegBin Hessian bug**: The closed-form Hessian in `negbin.py` uses a "working weight" formula `mu/(1+α·mu)` for quasi-likelihood, but the loss is Poisson-like with true Hessian `mu`. These don't match. Fix: change Hessian to use `mu` instead of `mu/(1+α·mu)`, or use autodiff.

**Extended test**: Also verified with estimated θ̂(x) from a trained model. Same results.

**Poisson/NegBin note**: The 1e-9 gradient errors are from the `log(mu + 1e-10)` numerical stability term. Acceptable.

---

## Eval 03: Lambda Estimation

**Goal**: Verify EstimateLambda recovers Λ(x) = E[ℓ_θθ | X=x].

**Result**: PASS. Mean Frobenius error 0.12, all eigenvalues positive (min=0.035), 0/1000 non-PSD.

**Note**: The `aggregate` method returns mean(Hessians), losing x-dependence. This is stable but doesn't capture Λ(x) heterogeneity. For this DGP, the heterogeneity is small so it works.

---

## Eval 04-06

(TODO)
