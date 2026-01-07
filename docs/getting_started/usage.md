# Usage Guide

A practical guide to using `deepstats` for structural deep learning with valid inference.

---

## The Fundamental Workflow

```
1. SPECIFY your economic model (what's the structure?)
2. DEFINE your target (what do you want to learn?)
3. FIT the model (let the package do its work)
4. VALIDATE the results (does this make sense?)
5. INTERPRET and USE (economics, not statistics)
```

---

## Step 1: Specify Your Economic Model

**The question you must answer:** What is the structural relationship between T and Y?

This is economics, not statistics. You're encoding domain knowledge about how the world works.

| If your outcome is... | And you believe... | Use this family |
|-----------------------|-------------------|-----------------|
| Continuous, unbounded | Linear response | `linear` |
| Binary (0/1) | Latent utility model | `logit` |
| Continuous, censored at 0 | Corner solutions | `tobit` |
| Count (0, 1, 2, ...) | Poisson process | `poisson` |
| Positive, skewed | Multiplicative errors | `gamma` |
| Duration/survival | Hazard model | `weibull` |

**Example thought process:**

> "I'm studying labor supply. The outcome Y is hours worked. Many people work zero hours (corner solution), and hours can't be negative. This sounds like a Tobit model."

```python
family = get_family("tobit")
```

---

## Step 2: Define Your Target

**The question you must answer:** What economic quantity do you want to estimate and do inference on?

Common targets:

| Target | Mathematical Form | Economic Meaning |
|--------|------------------|------------------|
| Average effect | $E[\beta(X)]$ | Mean treatment effect across population |
| Effect at a point | $\beta(x_0)$ | Effect for a specific type of person |
| Average marginal effect | $E[\partial P / \partial T]$ | Mean change in probability (nonlinear) |
| Optimal policy | $\arg\max_t \Pi(t, x)$ | Personalized optimal treatment |
| Welfare | $E[CS(\theta(X))]$ | Consumer surplus |

**For most applications, the default target $E[\beta(X)]$ is what you want.**

```python
# Default: average treatment effect
result = influence(X, T, Y, family, config)
```

---

## Step 3: Fit the Model

### Network Architecture

```python
# Wider/deeper = more flexibility, but needs more data
config = {
    "hidden_dims": [64, 32],  # Default: reasonable for most cases
    "epochs": 100,
    "lr": 0.01
}
```

**Rules of thumb:**

| Sample Size | Architecture |
|-------------|-------------|
| n < 1,000 | `[32, 16]` or simpler |
| 1,000 < n < 10,000 | `[64, 32]` |
| 10,000 < n < 100,000 | `[128, 64, 32]` |
| n > 100,000 | `[256, 128, 64]` or experiment |

### Number of Folds

```python
# More folds = more stable, but slower
n_folds = 20  # Default
n_folds = 50  # Better for final results
```

Monte Carlo validation shows K=50 gives best coverage. Use K=20 for exploration, K=50 for final results.

### The Actual Call

```python
from deepstats import get_dgp, get_family, influence

# Generate or load data
dgp = get_dgp("linear", seed=42)
data = dgp.generate(n=2000)

# Run influence function inference
family = get_family("linear")
result = influence(
    X=data.X,
    T=data.T,
    Y=data.Y,
    family=family,
    config={
        "hidden_dims": [64, 32],
        "epochs": 100,
        "n_folds": 20
    }
)
```

---

## Step 4: Validate the Results

**Before trusting your estimates, check these things:**

### Did the Neural Network Converge?

Check training loss - it should decrease and flatten:
- If still decreasing: need more epochs
- If erratic: learning rate too high

### Are the Parameter Functions Sensible?

Ask yourself:
- Does the pattern make economic sense?
- Are there extreme values that seem wrong?
- Is there enough variation to matter?

### Is $\Lambda(x)$ Well-Behaved?

Check condition numbers of estimated Hessians:
- If very large (>1000): near-singularity problems
- May need more regularization or simpler model

### Compare to Naive Estimate

```python
print(f"Naive estimate: {result.mu_naive:.4f}")
print(f"Debiased estimate: {result.mu_hat:.4f}")
print(f"Difference: {result.mu_hat - result.mu_naive:.4f}")

# If very different: the correction matters a lot
# If similar: either correction is small or something may be wrong
```

---

## Step 5: Interpret and Use

### Report Results

```python
print(f"Average Treatment Effect: {result.mu_hat:.3f}")
print(f"Standard Error: {result.se:.3f}")
print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
```

### Explore Heterogeneity

The whole point of this framework is heterogeneity. Use it!

```python
# Get theta(x) for all observations
theta_all = result.predict_theta(X)
alpha_all = theta_all[:, 0]
beta_all = theta_all[:, 1]

# Who has the largest treatment effects?
top_responders = np.argsort(beta_all)[-100:]

# What characterizes them?
print(X[top_responders].mean(axis=0))
```

---

## Complete Example

```python
import numpy as np
from deepstats import get_family, influence

# ======================
# 1. LOAD AND PREPARE DATA
# ======================
# Y: binary purchase decision
# T: price offered (continuous)
# X: customer characteristics

Y = data["purchased"].values
T = data["price"].values
X = data[["age", "income", "tenure", "segment"]].values

# ======================
# 2. EXPLORATORY ANALYSIS
# ======================
print(f"N = {len(Y)}")
print(f"Purchase rate: {Y.mean():.2%}")
print(f"Price range: [{T.min():.2f}, {T.max():.2f}]")
print(f"X dimensions: {X.shape[1]}")

# Check for obvious issues
assert not np.isnan(Y).any(), "Missing outcomes"
assert not np.isnan(T).any(), "Missing treatments"
assert not np.isnan(X).any(), "Missing covariates"

# ======================
# 3. FIT THE MODEL
# ======================
family = get_family("logit")

# First pass: quick exploration
result_v1 = influence(
    X=X, T=T, Y=Y,
    family=family,
    config={"hidden_dims": [32, 16], "epochs": 50, "n_folds": 10}
)
print(f"Quick estimate: {result_v1.mu_hat:.4f} +/- {result_v1.se:.4f}")

# Second pass: production quality
result = influence(
    X=X, T=T, Y=Y,
    family=family,
    config={"hidden_dims": [64, 32], "epochs": 100, "n_folds": 50}
)

# ======================
# 4. VALIDATE
# ======================
# Compare naive vs debiased
print(f"Naive: {result.mu_naive:.4f}")
print(f"Debiased: {result.mu_hat:.4f}")
print(f"Bias correction: {result.mu_hat - result.mu_naive:.4f}")

# ======================
# 5. INTERPRET
# ======================
print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"Average Price Sensitivity: {result.mu_hat:.4f}")
print(f"Standard Error: {result.se:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")

if result.ci_upper < 0:
    print("Conclusion: Price significantly reduces purchase probability")
elif result.ci_lower > 0:
    print("Conclusion: Price significantly increases purchase probability")
else:
    print("Conclusion: Cannot reject zero price effect")
```

---

## Common Mistakes to Avoid

| Mistake | Why It's Wrong | What to Do Instead |
|---------|---------------|-------------------|
| Using linear family for binary Y | Predictions outside [0,1] | Use logit |
| Too few folds (K < 10) | Unstable estimates | Use K >= 20, ideally 50 |
| Not checking convergence | May have bad estimates | Always check training loss |
| Ignoring naive vs debiased | Miss whether correction matters | Always report both |
| Over-complicated architecture | Overfitting, slow | Start simple, add complexity if needed |
| Not exploring heterogeneity | Miss the whole point | Always look at parameter distribution |
| Treating estimates as ground truth | They have error | Use confidence intervals for decisions |

---

## When to Use What

| Situation | Recommendation |
|-----------|---------------|
| Quick exploration | K=10, small network, 50 epochs |
| Paper/publication | K=50, validated architecture, full diagnostics |
| Binary outcome | `family="logit"` |
| Censored outcome | `family="tobit"` |
| Count outcome | `family="poisson"` |
| Continuous outcome | `family="linear"` |
| Small data (n < 1000) | Simpler network, more regularization |
| Large data (n > 100k) | Larger network, can use more folds |

---

## The Mindset

**Remember what this framework is for:**

1. **Learning heterogeneity** - The parameter functions $\theta(x)$ show how effects vary across people

2. **Valid inference** - The influence function correction gives you trustworthy confidence intervals

3. **Economic interpretation** - The structure ensures parameters mean something

4. **Policy decisions** - Personalization and optimization require structure + heterogeneity

**It is NOT for:**

- Pure prediction (use standard ML instead)
- Discovering causal structure (you impose it)
- Replacing domain knowledge (you still need to specify the model)

The package automates the statistics. The economics is still your job.
