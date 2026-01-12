# Metrics Module

Helper functions for computing inference quality metrics.

## Main Functions

### compute_coverage

```python
from deep_inference import compute_coverage

# Check if true value falls within CI
covered = compute_coverage(mu_true, ci_lower, ci_upper)
```

### compute_se_ratio

```python
from deep_inference import compute_se_ratio

# Compare estimated SE to empirical SE
se_ratio = compute_se_ratio(estimated_se, empirical_se)
```

## Usage Example

```python
from deep_inference import structural_dml
import numpy as np

# Run multiple simulations
results = []
for seed in range(100):
    np.random.seed(seed)
    # Generate data...
    result = structural_dml(Y, T, X, family='linear')
    results.append({
        'mu_hat': result.mu_hat,
        'se': result.se,
        'ci_lower': result.ci_lower,
        'ci_upper': result.ci_upper
    })

# Compute metrics
mu_true = 0.5  # known ground truth
mu_hats = [r['mu_hat'] for r in results]
ses = [r['se'] for r in results]

# Coverage
covered = [(r['ci_lower'] <= mu_true <= r['ci_upper']) for r in results]
coverage = np.mean(covered)
print(f"Coverage: {coverage:.1%}")  # Target: 95%

# SE ratio
empirical_se = np.std(mu_hats)
mean_estimated_se = np.mean(ses)
se_ratio = mean_estimated_se / empirical_se
print(f"SE Ratio: {se_ratio:.2f}")  # Target: 1.0
```

## Key Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| `bias` | $E[\hat\mu] - \mu^*$ | 0 |
| `variance` | $\text{Var}(\hat\mu)$ | - |
| `rmse` | $\sqrt{\text{Bias}^2 + \text{Var}}$ | Small |
| `empirical_se` | $\sqrt{\text{Var}}$ | - |
| `se_ratio` | $\hat{SE} / SE_{emp}$ | 1.0 |
| `coverage` | $P(\mu^* \in CI)$ | 95% |

## Validation Targets

| Metric | Valid Range | Interpretation |
|--------|-------------|----------------|
| Coverage | 93-97% | CI contains true value |
| SE Ratio | 0.9-1.2 | SE is properly calibrated |
| min(lambda) | > 1e-4 | Hessian is well-conditioned |

## Interpreting Results

### Good Results

```
Coverage: 95.0%   [PASS - in 93-97% range]
SE Ratio: 1.02    [PASS - close to 1.0]
RMSE: 0.032       [Low bias and variance]
```

### Warning Signs

```
Coverage: 30%     [FAIL - severe undercoverage]
SE Ratio: 0.27    [FAIL - SE underestimated 4x]
```

Common causes of poor coverage:
- Naive method (no IF correction)
- Too few folds (K < 20)
- Insufficient training epochs
- Model misspecification

## Monte Carlo Validation

For rigorous validation, run Monte Carlo simulations:

```python
import numpy as np
from deep_inference import structural_dml

M = 100  # number of simulations
N = 2000  # sample size
MU_TRUE = 0.5

results = []
for m in range(M):
    np.random.seed(m)

    # Generate data with known DGP
    X = np.random.randn(N, 10)
    T = np.random.randn(N)
    Y = X[:, 0] + MU_TRUE * T + np.random.randn(N)

    result = structural_dml(Y, T, X, family='linear', verbose=False)

    covered = result.ci_lower <= MU_TRUE <= result.ci_upper
    results.append({
        'mu_hat': result.mu_hat,
        'se': result.se,
        'covered': covered
    })

# Summary
coverage = np.mean([r['covered'] for r in results])
se_ratio = np.mean([r['se'] for r in results]) / np.std([r['mu_hat'] for r in results])

print(f"Coverage: {coverage:.1%}")
print(f"SE Ratio: {se_ratio:.2f}")
```
