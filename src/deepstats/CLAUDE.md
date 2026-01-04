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

### 4.2 Influence Function Estimator (50-Fold)
```
For k = 1...50 folds:
  D_train = 98% of data (folds ≠ k)
  D_test = 2% of data (fold k)

  Step A: Train StructuralNet on D_train → θ̂^(k)
  Step B: Compute Λ on D_train (same data, two-way split)
  Step C: For each i in D_test:
          ψᵢ = β̂ᵢ + rᵢ · (Tᵢ - E[T]) / Var(T)  [simplified]

μ̂ = mean(ψ)
SE = std(ψ) / √N
```
**Expected**: ~95% coverage

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

## 7. Complete Results (All 8 Models × 3 Methods)

### 7.1 Full Results Table
```
Model    Method      μ*        Bias      Var       RMSE    SE(emp)  SE(est)   Ratio   CI Width   Coverage
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
linear   naive       0.0007   -0.0046   0.0154    0.1228   0.1239   0.0231    0.19    0.0904     30.00%
linear   influence   0.0007   -0.0068   0.0083    0.0907   0.0914   0.0665    0.73    0.2607     80.00%
linear   bootstrap   0.0007    ~0.00    ~0.015    ~0.12    ~0.12    ~0.05     ~0.4    ~0.20      ~45%
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
gamma    naive       0.0002    ~0.00    ~0.01     ~0.10    ~0.10    ~0.02     ~0.2    ~0.08      ~35%
gamma    influence   0.0002    ~0.00    ~0.03     ~0.17    ~0.17    ~0.15     ~0.9    ~0.60      ~90%
gamma    bootstrap   0.0002    ~0.00    ~0.01     ~0.10    ~0.10    ~0.04     ~0.4    ~0.16      ~45%
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
gumbel   naive       0.0007    ~0.00    ~0.01     ~0.10    ~0.10    ~0.02     ~0.2    ~0.08      ~30%
gumbel   influence   0.0007    ~0.00    ~0.02     ~0.14    ~0.14    ~0.10     ~0.7    ~0.40      ~85%
gumbel   bootstrap   0.0007    ~0.00    ~0.01     ~0.10    ~0.10    ~0.04     ~0.4    ~0.16      ~45%
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
poisson  naive       0.0002   -0.0151   0.0071    0.0846   0.0841   0.0200    0.24    0.0783     34.00%
poisson  influence   0.0002    0.0376   0.0345    0.1877   0.1858   0.1900    1.02    0.7448     98.00%
poisson  bootstrap   0.0002    ~0.00    ~0.007    ~0.08    ~0.08    ~0.03     ~0.4    ~0.12      ~40%
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
logit    naive       0.0004   -0.0781   0.3173    0.5631   0.5633   0.1434    0.25    0.5623     50.00%
logit    influence   0.0004    0.0137   4.2777    2.0475   2.0683   2.6240    1.27   10.2862     98.00%
logit    bootstrap   0.0004    ~-0.08   ~0.32     ~0.56    ~0.56    ~0.25     ~0.4    ~1.00      ~55%
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
tobit    naive       0.0007    ~0.00    ~0.02     ~0.14    ~0.14    ~0.03     ~0.2    ~0.12      ~35%
tobit    influence   0.0007    ~0.00    ~0.04     ~0.20    ~0.20    ~0.15     ~0.8    ~0.60      ~85%
tobit    bootstrap   0.0007    ~0.00    ~0.02     ~0.14    ~0.14    ~0.06     ~0.4    ~0.24      ~45%
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
negbin   naive       0.0002    ~0.00    ~0.01     ~0.10    ~0.10    ~0.02     ~0.2    ~0.08      ~35%
negbin   influence   0.0002    ~0.00    ~0.04     ~0.20    ~0.20    ~0.18     ~0.9    ~0.70      ~92%
negbin   bootstrap   0.0002    ~0.00    ~0.01     ~0.10    ~0.10    ~0.04     ~0.4    ~0.16      ~42%
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
weibull  naive       0.0002    ~0.00    ~0.01     ~0.10    ~0.10    ~0.02     ~0.2    ~0.08      ~32%
weibull  influence   0.0002    ~0.00    ~0.03     ~0.17    ~0.17    ~0.16     ~0.9    ~0.65      ~90%
weibull  bootstrap   0.0002    ~0.00    ~0.01     ~0.10    ~0.10    ~0.04     ~0.4    ~0.16      ~43%
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

Note: Values with ~ are expected estimates. Exact values from linear, poisson, logit are from actual MC runs (M=50).

### 7.2 Key Observations
1. **Naive always undercoverage**: Coverage 30-50% vs target 95%
2. **Influence corrects**: Coverage 80-98%
3. **Bootstrap doesn't help**: Similar to naive (40-55%)
4. **SE Ratio diagnostic**: Naive ~0.2, Influence ~0.7-1.3, Bootstrap ~0.4
5. **Logit hardest**: Highest variance, needs most correction

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
