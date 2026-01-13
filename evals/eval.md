# Eval Learnings

What we learn from each eval. Updated as we investigate.

---

## Eval 01: Parameter Recovery

**Goal**: Recover θ*(x) = [α*(x), β*(x)] for all families.

**Results by family** (n=2000, epochs=100):
| Family | RMSE(α) | RMSE(β) | Corr(α) | Corr(β) | Status |
|--------|---------|---------|---------|---------|--------|
| Linear | 0.10 | 0.10 | 0.96 | 1.00 | PASS |
| Gaussian | 0.09 | 0.08 | 0.97 | 1.00 | PASS |
| Logit | 0.09 | 0.03 | 0.98 | 1.00 | PASS |
| Poisson | 0.08 | 0.02 | 0.98 | 1.00 | PASS |
| NegBin | 0.12 | 0.04 | 0.96 | 0.99 | PASS |
| Gamma | 0.06 | 0.11 | 1.00 | 0.90 | PASS |
| Weibull | 0.21 | 0.01 | 1.00 | 1.00 | PASS |
| Gumbel | 0.05 | 0.03 | 0.99 | 1.00 | PASS |
| Tobit | 0.03 | 0.04 | 1.00 | 1.00 | PASS |

**Overall**: 9/9 PASS with relaxed thresholds (RMSE < 0.3, Corr > 0.7).

**Legacy issues (Logit-specific)**: Early stopping and flat loss surface were previously problematic with stricter thresholds. These are now mitigated with patience=50 default.

**What helps**: More data (n=5k+), patience=50+, family-specific DGP coefficients (smaller for Poisson/NegBin to avoid overflow).

---

## Eval 02: Autodiff vs Calculus

**Goal**: Verify torch.func autodiff matches closed-form gradient/Hessian for all families.

**Results by family** (9 families):
| Family | Gradient | Hessian | Status |
|--------|----------|---------|--------|
| Linear | 1e-16 | 0 | PASS |
| Logit | 1e-16 | 1e-16 | PASS |
| Poisson | 1e-9 | 1e-9 | PASS |
| Gamma | 1e-15 | 1e-15 | PASS |
| Gaussian | 1e-6 | 1e-6 | PASS (now theta_dim=3 with MLE for sigma) |
| Gumbel | 0 | 0 | PASS |
| Tobit | N/A | N/A | Uses autodiff only (verified against Tobias/Purdue notes) |
| NegBin | 1e-9 | 1e-9 | PASS |
| Weibull | 1e-15 | 1e-15 | PASS |

**Fixed (2026-01-13)**:
- NegBin: Now uses true NegBin NLL with `lgamma` terms (was using Poisson-like loss)
- Gaussian: Now estimates sigma via MLE (theta_dim=3, gamma=log(sigma)), distinct from Linear
- NegBin Hessian: `w = r * mu * (r + y) / (r + mu)²` (observed information)

**Extended test**: Also verified with estimated θ̂(x) from a trained model. Same results.

**Poisson/NegBin note**: The 1e-9 gradient errors are from the `log(mu + 1e-10)` numerical stability term. Acceptable.

**Tobit verification** (against Tobias, Purdue Econ 674 notes): Our loss function matches the standard formulation exactly. Uncensored: `log(σ) + (y-μ)²/(2σ²)`. Censored: `-log(Φ(-z))` which equals `-log(1-Φ(z))` from the textbook. Our `target='observed'` marginal effect `β·Φ(z)` matches the PDF's `∂E(y|x)/∂x_j = β_j·Φ(xβ/σ)`. We use `γ=log(σ)` parameterization (ensures σ>0) rather than Olsen's `(δ=1/σ, θ=β/σ)` which provides global concavity - both valid, ours is more standard for neural nets.

---

## Eval 03: Lambda Estimation

**Goal**: Verify EstimateLambda recovers Λ(x) = E[ℓ_θθ | X=x].

**Result**: PASS. Mean Frobenius error 0.12, all eigenvalues positive (min=0.035), 0/1000 non-PSD.

**Note**: The `aggregate` method returns mean(Hessians), losing x-dependence. This is stable but doesn't capture Λ(x) heterogeneity. For this DGP, the heterogeneity is small so it works.

---

## Eval 04-06

(TODO)
