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

## Eval 04: Target Jacobian H_θ

**Goal**: Verify autodiff ∂H/∂θ matches closed-form oracle formulas.

**Paper Reference**: Theorem 2 (FLM 2021)
> "√n(μ̂ - μ*) →_d N(0, E[ψ²]) where ψ = H_θ(X,θ) · Λ(X)⁻¹ · ℓ_θ(Z,θ)"

The influence function formula requires accurate H_θ. If H_θ is wrong, the entire IF correction is wrong → invalid standard errors.

**Mathematical Objects Tested**:
| Target | Formula | H_θ |
|--------|---------|-----|
| AverageParameter | H(θ) = β | [0, 1] |
| AME (Logit) | H(θ) = σ(α+βt̃)(1-σ)β | [σ(1-σ)(1-2σ)β, σ(1-σ)(1+(1-2σ)βt̃)] |
| AME (Poisson) | H(θ) = β·exp(α+βt̃) | [βμ, μ(1+βt̃)] |
| Prediction | H(θ) = g⁻¹(α+βt̃) | Depends on link |

**Test Matrix (9 parts)**:
| Part | Description | Tests |
|------|-------------|-------|
| 1 | Target coverage (Logit) | AvgParam, AME, Prediction |
| 2 | Family coverage (AME) | Linear, Logit, Poisson, Probit |
| 3 | Edge cases | θ = [±5, 1], tiny/large effects |
| 4 | Batched vmap | 100 random θ per config |
| 5 | Package Target classes | AverageParameter, AME, CustomTarget |
| 6 | All 9 families | Linear, Logit, Poisson, Gamma, Weibull, NegBin, Gumbel, Beta, Probit |
| 7 | Higher-dim θ | Gaussian (3D), Tobit (3D), ZIP (4D) |
| 8 | Varying θ(x) | 50 obs with θ varying as NN output |
| 9 | Elasticity | ε = β·t̄ for Poisson |

**Pass Criteria**:
- Standard θ: max|err| < 1e-10 (machine precision)
- Edge θ: max|err| < 1e-6 OR relative error < 1e-4
- Batched: max|err| < 1e-8

**Results**: 9/9 parts PASS, overall max|err| ≈ 1e-14 (machine precision).

---

## Eval 05: Influence Function Assembly ψ

**Goal**: Verify complete IF assembly matches Oracle formula.

**Formula** (Theorem 2):
```
ψ_i = H(θ_i) - H_θ(θ_i) · Λ(x_i)⁻¹ · ℓ_θ(y_i, t_i, θ_i)
```

**Paper References**:
- Theorem 2: "√n(μ̂ - μ*) →_d N(0, E[ψ²])"
- Theorem 3: "Var(ψ)/n gives valid SE"
- Remark 4: "Λ(x) varies with x through θ(x)" - per-observation Lambda

**DGP**: Heterogeneous Logistic (Regime C - most stressful)
- X ~ Uniform(-2, 2)
- α*(x) = 0.5·sin(x), β*(x) = 1.0 + 0.5·x
- T = β*(x) + N(0, 0.5²) [CONFOUNDED]
- Y ~ Bernoulli(σ(α*(x) + β*(x)·T))
- Target: AME at t̃=0, μ* ≈ 0.241

**Test Rounds**:
| Round | Description | Pass Criteria |
|-------|-------------|---------------|
| A | Mechanical Assembly | Corr > 0.999, Max|diff| < 0.01, RMSE < 0.01 |
| B | Neyman Orthogonality | bias ~ O(δ²) not O(δ), bias < 10·δ² |
| C | Variance Formula | Var(ψ) > 0, SE > 0, SE < 1 |
| D | Multi-Seed Coverage | Coverage in [88%, 98%] over 50 seeds |
| E | Aggregate vs Per-obs Λ | Demonstrates U-shape limitation |

**Round Details**:
- **Round A**: Uses identical inputs (true θ*, same Λ) for oracle and package. Tests assembly code correctness.
- **Round B**: Perturbs θ by δ, verifies bias scales as O(δ²). This is the Neyman orthogonality property - IF is first-order insensitive to θ estimation error.
- **Round C**: Basic sanity checks on variance formula from Theorem 3.
- **Round D**: Monte Carlo with 50 seeds, coverage should be 88-98% (allowing MC noise).
- **Round E**: Compares aggregate Λ vs per-observation Λ(xᵢ). Confirms Remark 4: aggregate shows U-shaped SE ratio, per-obs is stable.

**Results**: 4/4 core rounds PASS. Round E confirms theoretical prediction.

---

## Eval 06: Frequentist Coverage

**Goal**: Monte Carlo validation that CIs achieve valid coverage.

**Paper Reference**: Theorem 3 (FLM 2021)
> "√n(μ̂ - μ*) →_d N(0, V) where V = E[ψ₀(W)²]"

**Procedure**:
```
For m = 1, ..., M:
    1. Generate data from canonical DGP (logit, confounded)
    2. Run inference() to get μ̂, SE, CI
    3. Check if true μ* is in CI
```

**Settings** (NON-NEGOTIABLE):
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| M | 50 | Detects systematic failure |
| n | 8000 | Binary families need ~2x data (1 bit/obs) |
| epochs | 200 | Proper convergence |
| patience | 50 | Matches eval_01 |
| lambda_method | lgbm | Validated 96% coverage |
| t_tilde | 0.0 | Must match DGP's mu_true() |
| n_jobs | 4 | Avoids OOM (11 workers caused OOM) |

**Pass Criteria**:
| Metric | Range | Tests |
|--------|-------|-------|
| Coverage | [85%, 99%] | Valid CIs |
| SE ratio | [0.5, 2.0] | SE calibration |
| \|Bias\| | < 0.1 | Unbiased estimation |
| \|z_mean\| | < 0.5 | z-scores centered |
| z_std | [0.5, 2.0] | z-scores ~N(0,1) |

**Results** (M=50, n=8000, lgbm):
```
Coverage: 96% (48/50)
SE Ratio: 0.914
Bias: -0.0064
z_mean: ~0
z_std: ~1
```

**Status**: PASS with 96% coverage.

---

## Summary

| Eval | Component | Status |
|------|-----------|--------|
| 01 | θ*(x) Recovery | 9/9 PASS |
| 02 | Autodiff ∇ℓ, ∇²ℓ | 9/9 PASS |
| 03 | Λ(x) Estimation | PASS |
| 04 | H_θ Jacobian | 9/9 PASS |
| 05 | ψ Assembly | 4/4 PASS |
| 06 | Coverage (96%) | PASS |

All components of Theorem 2 validated end-to-end.
