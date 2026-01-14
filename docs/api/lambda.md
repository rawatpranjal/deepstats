# Lambda Strategies

The `lambda_` module handles estimation of Λ(x) = E[∂²ℓ/∂θ² | X=x], the conditional Hessian of the loss function. This is critical for valid influence function inference.

## Overview

The package supports three **regimes** based on your data and model:

| Regime | Condition | Lambda Method | Cross-Fitting |
|--------|-----------|---------------|---------------|
| **A** | RCT with known F_T | Compute via MC | 2-way |
| **B** | Linear model | Analytic closed-form | 2-way |
| **C** | Observational + nonlinear | Estimate via neural net | 3-way |

---

## Regime Detection

The package automatically detects the appropriate regime:

```python
from deep_inference.lambda_ import detect_regime, Regime

# Returns Regime.A, Regime.B, or Regime.C
regime = detect_regime(
    model=my_model,
    is_randomized=True,
    has_treatment_dist=True
)
```

### Decision Logic

```
if is_randomized AND treatment_dist provided:
    → Regime A (ComputeLambda)
elif model is linear:
    → Regime B (AnalyticLambda)
else:
    → Regime C (EstimateLambda)
```

---

## Lambda Strategies

### ComputeLambda (Regime A)

For randomized experiments where T is independent of X and has known distribution F_T.

```python
from deep_inference import inference
from deep_inference.lambda_.compute import Normal, Bernoulli, Uniform

# Normal treatment: T ~ N(μ, σ²)
result = inference(
    Y, T, X,
    model='logit',
    is_randomized=True,
    treatment_dist=Normal(mean=0.0, std=1.0)
)

# Binary treatment: T ~ Bernoulli(p)
result = inference(
    Y, T, X,
    model='logit',
    is_randomized=True,
    treatment_dist=Bernoulli(p=0.5)
)

# Uniform treatment: T ~ Uniform(a, b)
result = inference(
    Y, T, X,
    model='logit',
    is_randomized=True,
    treatment_dist=Uniform(low=-1.0, high=1.0)
)
```

**How it works:**
```
Λ(x, θ) = E_T[∂²ℓ/∂θ² | θ]
        ≈ (1/M) Σ ∂²ℓ(y, t_m, θ)/∂θ²
```

Where t_m ~ F_T are Monte Carlo samples.

**Advantages:**
- No neural network for Λ needed
- 2-way cross-fitting (faster)
- More stable

### AnalyticLambda (Regime B)

For linear models, Λ has a closed-form solution.

```python
# Automatically selected for linear models
result = inference(Y, T, X, model='linear')
# Uses AnalyticLambda internally
```

**Formula (linear):**
```
Λ = E[[1, T]ᵀ[1, T]] = [[1, E[T]], [E[T], E[T²]]]
```

### EstimateLambda (Regime C)

For observational data with nonlinear models, Λ(x) must be estimated.

```python
# Default for observational logit
result = inference(Y, T, X, model='logit')
# Uses EstimateLambda internally (3-way cross-fitting)
```

**How it works:**
1. Compute sample Hessians ∂²ℓ/∂θ² for training data
2. Train a neural network to predict Λ̂(x) from covariates
3. Predict Λ̂ for evaluation fold

**Cross-fitting:**
3-way split is required to avoid bias:
- Fold A: Train θ̂(x)
- Fold B: Train Λ̂(x) using θ̂ from Fold A
- Fold C: Evaluate ψ using both

---

## Treatment Distributions

### Available Distributions

```python
from deep_inference.lambda_.compute import Normal, Bernoulli, Uniform

# Continuous
Normal(mean=0.0, std=1.0)      # Gaussian
Uniform(low=-1.0, high=1.0)    # Uniform

# Discrete
Bernoulli(p=0.5)               # Binary
```

### Custom Distributions

```python
from deep_inference.lambda_.compute import TreatmentDistribution
import torch

class MyDistribution(TreatmentDistribution):
    def sample(self, n: int) -> torch.Tensor:
        # Return n samples from your distribution
        return torch.randn(n) * 2 + 1  # Example: N(1, 4)
```

---

## Strategy Selection API

### Automatic Selection

```python
from deep_inference.lambda_ import select_lambda_strategy

strategy = select_lambda_strategy(
    model=my_model,
    is_randomized=True,
    treatment_dist=Normal(0, 1),
    lambda_method=None  # Auto-detect
)
```

### Manual Override

```python
result = inference(
    Y, T, X,
    model='logit',
    lambda_method='estimate'  # Force estimation even if RCT
)
```

---

## Lambda Strategy Protocol

All strategies implement:

```python
class LambdaStrategy(Protocol):
    requires_theta: bool        # Does fit() need θ̂?
    requires_separate_fold: bool  # 3-way cross-fitting?

    def fit(self, X, T, Y, theta_hat, model) -> None:
        """Fit the strategy (if needed)."""
        ...

    def predict(self, X, theta_hat) -> Tensor:
        """Return Λ̂(x) matrices of shape (n, d_θ, d_θ)."""
        ...
```

---

## Diagnostics

Check which regime was used:

```python
result = inference(Y, T, X, model='logit', verbose=True)

print(result.diagnostics['regime'])        # 'A', 'B', or 'C'
print(result.diagnostics['lambda_method']) # 'ComputeLambda', etc.
```

---

## When to Use Each Regime

| Scenario | Regime | Example |
|----------|--------|---------|
| A/B test with known assignment | A | Marketing experiment with 50/50 split |
| Survey experiment with normal dosage | A | Price sensitivity study with T ~ N(0,1) |
| Linear regression | B | Simple OLS with heterogeneous coefficients |
| Observational logit | C | Health outcomes from medical records |
| Complex nonlinear model | C | Custom loss with unknown treatment selection |
