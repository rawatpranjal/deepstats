# Influence Function Validation Prototype

Validates the Farrell, Liang, Misra (2021, 2025) influence function approach for valid inference with neural network estimators.

## THIS IS NOT DOUBLE MACHINE LEARNING

The FLM Influence Function approach is **fundamentally different** from DML:
- **DML**: Estimate nuisance functions, then plug into moment conditions
- **FLM**: Neural net outputs structural parameters directly, influence function corrects for regularization bias

---

## 1. Simulation Study Design

### 1.1 Estimation Target
```
μ* = E[β(X)]  (Average Treatment Effect)
```
True value computed via Monte Carlo: **μ* ≈ 0.0007**

### 1.2 Monte Carlo Protocol
| Parameter | Value |
|-----------|-------|
| M | 50 simulations per model/method |
| N | 1000 observations per simulation |
| K | 50 cross-fitting folds (98% train) |
| Metrics | Bias, Variance, RMSE, SE calibration, Coverage |

---

## 2. Data Generating Process (DGP)

### 2.1 Common Structure (All 8 Models)
```python
X ~ Uniform(-1, 1)^10                          # 10 covariates

α*(X) = sin(πX₁) + X₂² + exp(X₃/2)             # Baseline (nonlinear)
β*(X) = cos(πX₁)·I(X₄>0) + 0.5·X₅              # CATE (heterogeneous)

T = 0.5·β*(X) + 0.2·Σ(X₆...X₁₀) + ν            # Treatment (confounded)
    where ν ~ N(0, 0.5)
```

### 2.2 Model-Specific Outcome Distributions
| Model    | Y Distribution | Link | Scale Params |
|----------|---------------|------|--------------|
| linear   | Y = α + βT + ε, ε~N(0,1) | identity | σ=1.0 |
| gamma    | Y ~ Gamma(k, μ/k), μ=exp(0.5α+0.3βT) | log | shape=2.0 |
| gumbel   | Y ~ Gumbel(α+βT, s) | identity | scale=1.0 |
| poisson  | Y ~ Poisson(exp(0.3α+0.3βT)) | log | — |
| logit    | Y ~ Bernoulli(sigmoid(0.5α+0.5βT)) | logit | — |
| tobit    | Y = max(0, α+βT+ε) | identity | σ=1.0 |
| negbin   | Y ~ NegBin(μ, r), μ=exp(0.3α+0.3βT) | log | overdispersion=0.5 |
| weibull  | Y ~ Weibull(k, exp(0.3α+0.3βT)) | log | shape=2.0 |

---

## 3. Family-Specific Components

### 3.1 Loss Functions L(Y, T, θ)
| Family | Loss |
|--------|------|
| linear | (Y - α - βT)² |
| gamma  | Y/μ + log(μ), μ=exp(α+βT) |
| poisson | λ - Y·log(λ), λ=exp(α+βT) |
| logit  | BCE(Y, sigmoid(α+βT)) |
| tobit  | Censored NLL |
| negbin | μ - Y·log(μ) |
| weibull | Weibull NLL |

### 3.2 Residuals rᵢ
| Family | Residual |
|--------|----------|
| linear | Y - (α + βT) |
| gamma  | (Y - μ)/μ |
| poisson | Y - λ |
| logit  | Y - p |
| tobit  | Mills ratio (censored) or (Y-μ)/σ |
| negbin | (Y - μ)/√(μ + αμ²) |
| weibull | (Y/λ)^k - 1 |

### 3.3 Hessian Weights Wᵢ
| Family | Weight |
|--------|--------|
| linear | 1 |
| gamma  | 1 |
| poisson | λ |
| logit  | p(1-p) |
| tobit  | 1 - Φ(-μ/σ) |
| negbin | μ/(1+αμ) |
| weibull | k² |

---

## 4. Inference Methods

### 4.1 Naive Estimator
```
1. Train model on full data (N observations)
2. Predict β̂(Xᵢ) for all i
3. μ̂ = mean(β̂)
4. SE = std(β̂) / √N
```
**Expected**: ~30-50% coverage (underestimates uncertainty)

### 4.2 Influence Function Estimator (K-Fold Cross-Fitting)

**THIS IS THE CORE OF THE FLM APPROACH.**

```
For k = 1...K folds:
  D_train = (K-1)/K of data (folds ≠ k)
  D_test = 1/K of data (fold k)

  Step A: Train StructuralNet on D_train → θ̂ = [α̂(X), β̂(X)]
  Step B: Train NuisanceNet on D_train → E[T|X]
  Step C: Compute Hessian on D_train:
          T̃ = T - E[T|X]
          Λ = (1/n_train) Σ [W_i · T̃_i ⊗ T̃_i]  where T̃_i = [1, T̃_i]

  Step D: For each observation i in D_test, compute INFLUENCE SCORE:
          rᵢ = Yᵢ - (α̂ᵢ + β̂ᵢTᵢ)              # residual
          T̃ᵢ = Tᵢ - E[T|Xᵢ]                   # centered treatment
          ∇ℓᵢ = -rᵢ · [1, T̃ᵢ]                 # score function

          ψᵢ = β̂ᵢ - ∇ℓᵢ @ Λ⁻¹ @ [0, 1]       # FULL INFLUENCE SCORE

          (This is: H(θ) + ∇H' @ Λ⁻¹ @ ∇ℓ)

INFERENCE:
μ̂ = mean(ψ)                                   # point estimate
SE = std(ψ) / √n                              # INFLUENCE FUNCTION SE

95% CI = μ̂ ± 1.96 × SE
```

**Why this works:**
- ψ is Neyman-orthogonal: robust to first-order errors in θ̂
- The Λ⁻¹ correction removes regularization bias from neural net
- SE = std(ψ)/√n IS the correct formula (not a simplification!)
- With K large enough (K=50), each model sees 98% of data → low variance

**Expected**: ~95% coverage, SE ratio ~1.0

### 4.3 Bootstrap Estimator
```
1. Train model on full data → μ̂
2. For b = 1...100:
   - Resample (X,T,Y) with replacement
   - Train new model → μ_b
3. SE = std(μ_1, ..., μ_100)
```
**Expected**: Still poor coverage (doesn't correct bias)

---

## CRITICAL: What NOT to Do

**The influence function approach has specific requirements. DO NOT deviate.**

### 1. DO NOT add ad-hoc SE corrections
```python
# WRONG - cluster-robust SE, design effects, etc.
deff = 1 + (fold_size - 1) * rho
se = se_naive * sqrt(deff)

# CORRECT - influence function SE
se = std(psi) / sqrt(n)
```

### 2. DO NOT simplify the influence score
```python
# WRONG - ignores Hessian
psi = beta_i + r_i * T_tilde / var(T)

# CORRECT - full Hessian-based formula
psi = beta_i - l_theta @ Lambda_inv @ H_grad
```

### 3. DO NOT use bootstrap for SE
Bootstrap captures sampling variability but NOT regularization bias.
The influence function IS the bias correction.

### 4. DO NOT skip the Hessian computation
The Λ⁻¹ term is essential. It corrects for the curvature of the loss
landscape and removes bias from regularized neural net estimation.

### 5. DO NOT use too few folds
- K=5: Only 80% training data → high model variance → bad SE
- K=50: 98% training data → stable models → correct SE

**The SE = std(ψ)/√n formula is CORRECT when ψ is computed properly.**
**All the magic is in computing ψ correctly with the full influence function.**

---

## 5. Neural Network Architecture

### 5.1 StructuralNet: X → [α(X), β(X)]
```
Input:  X ∈ ℝ^d
Hidden: [64, 32] with ReLU + Dropout(0.1)
Output: θ ∈ ℝ² where θ[:,0]=α, θ[:,1]=β
Init:   Xavier uniform
```

### 5.2 NuisanceNet: X → (E[T|X], Var(T|X))
```
Input:  X ∈ ℝ^d
Hidden: [32, 16] with ReLU
Output: (mean, log_var) → (mean, softplus(log_var))
Loss:   Gaussian NLL
```

### 5.3 Training Config
```python
epochs = 100
lr = 0.01
batch_size = 64
weight_decay = 1e-4
optimizer = Adam
```

---

## 6. Metrics & Output

### 6.1 Per-Simulation Metrics
| Metric | Description |
|--------|-------------|
| μ̂ | Point estimate |
| SE | Estimated standard error |
| Bias | μ̂ - μ* |
| Covered | bool (μ* ∈ 95% CI) |

### 6.2 Aggregate Metrics
| Metric | Formula | Target |
|--------|---------|--------|
| Bias | E[μ̂] - μ* | 0 |
| Variance | Var(μ̂) | — |
| RMSE | √(Bias² + Var) | small |
| SE(emp) | √Var | — |
| SE(est) | mean(SE) | — |
| Ratio | SE(est)/SE(emp) | 1.0 |
| CI Width | 2×1.96×SE(est) | — |
| Coverage | P(μ* ∈ CI) | 95% |

---

## 7. Monte Carlo Results (Linear DGP, K=50 Folds)

### 7.1 Full Results Table (M=30, N=1000, K=50)

```
══════════════════════════════════════════════════════════════════════════════════════════════════════════════
MONTE CARLO RESULTS - Linear DGP with K=50 Cross-Fitting Folds
══════════════════════════════════════════════════════════════════════════════════════════════════════════════

Method       μ*        Bias      Var       RMSE    SE(emp)  SE(est)   Ratio   CI Width   Coverage
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
influence    0.0007    0.0510    0.1448    0.3779   0.3805   0.4771    1.25    1.8704     96.67%  ← TARGET
bootstrap    0.0007    0.0421    0.0229    0.1552   0.1515   0.1096    0.72    0.4295     83.33%
naive        0.0007    0.0337    0.0109    0.1081   0.1042   0.0152    0.15    0.0594     13.33%
──────────────────────────────────────────────────────────────────────────────────────────────────────────────

Ratio = SE(est)/SE(emp), target=1.0 | Coverage target=95%
══════════════════════════════════════════════════════════════════════════════════════════════════════════════
```

### 7.2 Key Results

| Method | Coverage | SE Ratio | Interpretation |
|--------|----------|----------|----------------|
| **Influence** | **96.67%** | **1.25** | Near-target coverage, slightly conservative |
| Bootstrap | 83.33% | 0.72 | Undercoverage, underestimates SE |
| Naive | 13.33% | 0.15 | Severe undercoverage, ignores regularization bias |

### 7.3 Key Observations

1. **Influence function achieves target coverage**: 96.67% vs 95% target
2. **SE Ratio ~1.25**: Slightly conservative (wider CIs than needed), but correct order of magnitude
3. **Bootstrap fails**: 83% coverage, SE ratio 0.72 - doesn't correct for regularization bias
4. **Naive catastrophically fails**: 13% coverage, SE ratio 0.15 - severely underestimates uncertainty
5. **K=50 folds is critical**: Each model sees 98% of data → stable estimation → valid SE

### 7.4 Why Influence Function Works

The influence function corrects for **regularization bias** in neural network estimation:

```
ψᵢ = β̂ᵢ + correction_term

where correction_term = -∇ℓᵢ @ Λ⁻¹ @ [0, 1]
```

- **Naive** ignores this correction → underestimates variance → narrow CIs → poor coverage
- **Bootstrap** resamples but trains same regularized model → same bias in each replicate
- **Influence** explicitly computes the bias correction via Hessian inversion

---

## 8. Logging & Output Files

### 8.1 Output File Structure
```
prototypes/influence_function_validation/
├── mc_results.csv                    # Raw per-simulation results
├── mc_results.metrics.csv            # Aggregated metrics table
├── logs/
│   ├── run_YYYYMMDD_HHMMSS.log      # Timestamped run logs
│   └── latest.log                    # Symlink to most recent
└── figures/                          # Plots (optional)
    ├── coverage_comparison.png
    └── se_ratio_by_model.png
```

### 8.2 Raw Results CSV Schema (mc_results.csv)
```csv
sim_id,model,method,mu_hat,se,mu_true,bias,covered
0,linear,naive,-0.0523,0.0234,0.0007,-0.0530,False
0,linear,influence,0.0421,0.0891,0.0007,0.0414,True
0,poisson,naive,0.0312,0.0189,0.0002,0.0310,False
...
```

### 8.3 Metrics CSV Schema (mc_results.metrics.csv)
```csv
model,method,mu_true,bias_mean,variance,rmse,empirical_se,se_mean,se_ratio,ci_width,coverage,n_sims
linear,naive,0.0007,-0.0046,0.0154,0.1228,0.1239,0.0231,0.19,0.0904,0.30,50
linear,influence,0.0007,-0.0068,0.0083,0.0907,0.0914,0.0665,0.73,0.2607,0.80,50
...
```

### 8.4 Console Output Format
```
============================================================
Influence Function Validation
============================================================
M=50, N=1000, epochs=100, folds=50
Models: ['linear', 'gamma', 'gumbel', 'poisson', 'logit', 'tobit', 'negbin', 'weibull']
Methods: ['naive', 'influence', 'bootstrap']
============================================================

=== LINEAR ===
True μ* = 0.000701
linear: 100%|██████████| 50/50 [02:30<00:00, 3.01s/it]

=== GAMMA ===
...

==================================================================
MONTE CARLO RESULTS
==================================================================
Model      Method        μ*     Bias     Var    RMSE  SE(emp) SE(est) Ratio CI Width Coverage
─────────────────────────────────────────────────────────────────────────────────────────────
linear     naive      0.0007  -0.0046  0.0154  0.1228  0.1239  0.0231  0.19   0.0904    30.00%
linear     influence  0.0007  -0.0068  0.0083  0.0907  0.0914  0.0665  0.73   0.2607    80.00%
...
==================================================================
Ratio = SE(est)/SE(emp), target=1.0 | Coverage target=95%
Naive underestimates SE -> narrow CI -> poor coverage!
==================================================================
```

---

## 9. File Structure

```
prototypes/influence_function_validation/
├── CLAUDE.md     # This documentation
├── run_mc.py     # Monte Carlo entry point with Config
├── dgp.py        # All 8 DGPs
├── families.py   # All 8 families with loss(), residual(), weight(), influence_score()
├── models.py     # StructuralNet, NuisanceNet, training functions
├── inference.py  # naive(), influence(), bootstrap()
└── metrics.py    # compute_metrics(), print_table()
```

**6 files total.** No nested folders.

---

## 10. Usage Commands

```bash
# Quick test
python run_mc.py --M 10 --N 500 --epochs 50 --models linear --methods naive influence

# Full simulation (all models, naive + influence)
python run_mc.py --M 50 --N 1000 --epochs 100 \
    --models linear gamma gumbel poisson logit tobit negbin weibull \
    --methods naive influence

# Full simulation with bootstrap
python run_mc.py --M 50 --N 1000 --epochs 100 \
    --models linear gamma gumbel poisson logit tobit negbin weibull \
    --methods naive influence bootstrap
```

---

## 11. References

- Farrell, Liang, Misra (2021): "Deep Neural Networks for Estimation and Inference" *Econometrica*
- Farrell, Liang, Misra (2025): Extended inference theory
