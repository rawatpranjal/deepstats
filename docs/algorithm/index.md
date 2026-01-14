# Algorithm

Implementation of Farrell-Liang-Misra influence function-based inference for neural networks.

---

## Overview

The package provides valid confidence intervals for neural network estimates by:

1. Training a structural network θ̂(x) via cross-fitting
2. Computing influence functions ψᵢ to correct for regularization bias
3. Using √n-consistent variance estimation

**Core formula:**
$$\psi_i = H(\hat{\theta}_i) - H_\theta \cdot \Lambda(x_i)^{-1} \cdot \ell_\theta(z_i, \hat{\theta}_i)$$

---

## Algorithm Flow

```
For each fold k = 1, ..., K:
    1. Train θ̂_k(x) on I_k^c (training set)
    2. Compute scores: ℓ_θ(zᵢ, θ̂_k(xᵢ)) for i ∈ I_k
    3. Compute Hessians: ℓ_θθ(zᵢ, θ̂_k(xᵢ)) for i ∈ I_k
    4. Estimate Λ̂_k(x) from Hessians
    5. Assemble ψᵢ = h - H_θ · Λ̂⁻¹ · ℓ_θ

Aggregate:
    μ̂ = (1/n) Σᵢ ψᵢ
    SE = √(Var(ψ)/n)
```

---

## Lambda Estimation (Critical)

The conditional Hessian Λ(x) = E[ℓ_θθ|X=x] has three estimation regimes:

### Regime A: Randomized Experiments (`ComputeLambda`)

When treatment T is randomized and independent of X:
```python
# Monte Carlo integration over treatment distribution
Λ(x) = E_T[ℓ_θθ(Y, T, θ(x)) | X=x]
     ≈ (1/M) Σₘ ℓ_θθ(y, tₘ, θ(x))  where tₘ ~ P(T)
```

Use: `inference(..., is_randomized=True, treatment_dist=Normal(0, 1))`

### Regime B: Linear Models (`AnalyticLambda`)

For linear models, Λ(x) = E[TT'|X=x] has closed form:
```python
# No θ-dependence, so 2-way splitting suffices
Λ(x) = E[(1,T)(1,T)' | X=x]
```

Use: `inference(..., model='linear')` (auto-detected)

### Regime C: Observational Nonlinear (`EstimateLambda`)

For logit, Poisson, etc. in observational settings:

| Method | Correlation | Speed | Use Case |
|--------|-------------|-------|----------|
| `aggregate` | 0.000 | 0.02s | Ignores heterogeneity |
| `ridge` | 0.508 | 0.08s | Fast fallback |
| `rf` | 0.904 | 0.3s | Moderate accuracy |
| `lgbm` | 0.978 | 1.2s | **Recommended** |
| `mlp` | 0.997 | 12.5s | Best accuracy |

**Key finding:** `aggregate` averages Hessians across all observations, ignoring X-dependence entirely. This gives Corr=0.000 with true Λ(x).

Use: `inference(..., lambda_method='lgbm')` or `structural_dml(..., lambda_method='mlp')`

---

## Three-Way Splitting

### When It's Needed

Three-way splitting is required when Λ(x) depends on θ(x).

**Two-way suffices:**
- Linear models (Λ = E[TT'|X], no θ-dependence)
- Randomized experiments (can compute Λ analytically)

**Three-way required:**
- Logit, Poisson, Gamma, Weibull, etc.
- Hessian weights depend on θ: e.g., p(1-p) for logit where p = σ(θ'x)

### Auto-Detection

The package auto-detects θ-dependence:
```python
# From core/algorithm.py
def detect_theta_dependence(family, X, T, theta):
    """Sample 100 observations, perturb θ, check if Hessian changes."""
    sample_idx = np.random.choice(len(X), min(100, len(X)))
    H1 = family.hessian(Y[sample_idx], T[sample_idx], theta[sample_idx])
    H2 = family.hessian(Y[sample_idx], T[sample_idx], theta[sample_idx] + 0.1)
    return not np.allclose(H1, H2, rtol=1e-5)
```

### How It Works

When `three_way=True` (or auto-detected):
```
For each fold k:
    Training set I_k^c is split:
    - 60% for training θ̂_k(x)
    - 40% for fitting Λ̂_k(x)
```

This prevents overfitting: θ̂ and Λ̂ are estimated on independent data.

---

## Regularization

### Lambda Inversion

Inverting Λ(x) can be unstable when Hessians are near-singular. The package uses Tikhonov regularization:

```python
# From utils/linalg.py
def batch_inverse(Lambda, ridge=1e-4):
    """Invert with ridge regularization for stability."""
    Lambda_reg = Lambda + ridge * torch.eye(Lambda.shape[-1])
    return torch.linalg.inv(Lambda_reg)
```

**Default:** `ridge=1e-4`

### Diagnostics

The package tracks `min_lambda_eigenvalue`:
```
Warning: min_lambda_eigenvalue=1.23e-06 < 1e-4
Consider increasing ridge regularization or checking model fit.
```

**When to increase regularization:**
- Near-singular Hessians (min eigenvalue < 1e-4)
- Logit with p ≈ 0 or p ≈ 1 (separation issues)
- Overdispersed count models

---

## Variance Estimation

### Within-Fold Formula

```python
# From core/algorithm.py
Ψ̂ = (1/K) Σ_k Var_k(ψ)
  = (1/K) Σ_k (1/|I_k|) Σ_{i ∈ I_k} (ψᵢ - μ̂_k)²

SE = √(Ψ̂/n)
```

Where μ̂_k is the fold-specific mean.

### High Correction Variance Warning

```
Warning: High correction variance ratio (43.96).
This suggests the influence function correction dominates the estimate variance.
Consider using more cross-fitting folds (K >= 50).
```

This warns when:
```python
correction_variance / total_variance > 10
```

Indicates the bias correction term H_θ · Λ⁻¹ · ℓ_θ has high variance, often from:
- Insufficient cross-fitting folds
- Poor θ̂(x) estimation in some regions
- Near-singular Lambda

---

## Two APIs

### Legacy: `structural_dml()`

For the 8 built-in GLM families:
```python
from deep_inference import structural_dml

result = structural_dml(
    Y=Y, T=T, X=X,
    family='logit',           # linear, logit, poisson, gamma, ...
    lambda_method='aggregate', # aggregate, mlp, lgbm, rf, ridge
    n_folds=50,
    epochs=100,
)
```

### New: `inference()`

For flexible targets and custom models:
```python
from deep_inference import inference

result = inference(
    Y=Y, T=T, X=X,
    model='logit',
    target='ame',              # ame, beta, or custom
    lambda_method='lgbm',
    is_randomized=False,
)
```

Both use the same core algorithm; `inference()` adds:
- Regime detection (A/B/C)
- Custom target functions
- Treatment distribution specification

---

## Implementation Files

| File | Purpose |
|------|---------|
| `core/algorithm.py` | Main `structural_dml_core()` with fold logic |
| `engine/crossfit.py` | Modular `CrossFitter` class |
| `engine/assembler.py` | Influence function assembly |
| `engine/variance.py` | SE computation |
| `lambda_/estimate.py` | MLP/LGBM Lambda estimation |
| `lambda_/selector.py` | Regime detection |
| `autodiff/` | Score, Hessian, Jacobian computation |

---

## Pseudocode (Actual Implementation)

```python
def structural_dml_core(Y, T, X, family, lambda_method, n_folds, ...):
    # Phase 1: Create folds
    kf = KFold(n_splits=n_folds, shuffle=True)

    # Detect if 3-way splitting needed
    three_way = detect_theta_dependence(family, X, T, initial_theta)

    psi_all = []
    for fold_idx, (train_idx, eval_idx) in enumerate(kf.split(X)):
        # Phase 2a: Train θ̂(x)
        if three_way:
            theta_train = train_idx[:int(0.6 * len(train_idx))]
            lambda_train = train_idx[int(0.6 * len(train_idx)):]
        else:
            theta_train = train_idx
            lambda_train = train_idx

        model = train_structural_net(X[theta_train], T[theta_train], Y[theta_train])
        theta_hat = model(X[eval_idx])

        # Phase 2b: Compute Hessians on lambda_train
        hessians = family.hessian(Y[lambda_train], T[lambda_train], theta_hat_train)

        # Phase 2c: Fit Lambda
        if lambda_method == 'aggregate':
            Lambda = hessians.mean(axis=0)  # Single matrix for all x
        else:
            Lambda_model = train_lambda_estimator(X[lambda_train], hessians)
            Lambda = Lambda_model(X[eval_idx])

        # Phase 3: Assemble influence function
        for i in eval_idx:
            score = family.gradient(Y[i], T[i], theta_hat[i])
            h = target.h(theta_hat[i])
            h_jacobian = target.jacobian(theta_hat[i])

            Lambda_inv = batch_inverse(Lambda[i], ridge=1e-4)
            psi_i = h - h_jacobian @ Lambda_inv @ score
            psi_all.append(psi_i)

    # Phase 4: Aggregate
    mu_hat = np.mean(psi_all)
    se = np.sqrt(np.var(psi_all) / n)

    return DMLResult(mu_hat=mu_hat, se=se, ...)
```

---

## References

- Farrell, Liang, Misra (2021): "Deep Neural Networks for Estimation and Inference" *Econometrica*
- Farrell, Liang, Misra (2025): "Deep Learning for Individual Heterogeneity" *Working Paper*
