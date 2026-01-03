# deepstats

**Production-grade econometrics with neural networks in PyTorch.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

`deepstats` bridges the **predictive flexibility** of machine learning (scikit-learn) with the **statistical rigor** of traditional econometrics (Stata/statsmodels). It provides neural network-based estimators with proper statistical inference.

## Key Features

- **sklearn-compatible estimators** - Works with `GridSearchCV`, `cross_val_score`, pipelines
- **Robust standard errors** - HC0, HC1, HC2, HC3, and clustered SEs
- **Statistical inference** - Summary tables, confidence intervals, p-values
- **Causal inference** - Double Machine Learning for treatment effect estimation
- **Meta-estimator pattern** - Plug in any sklearn-compatible learner

## Installation

```bash
pip install deepstats
```

For development:
```bash
git clone https://github.com/rawatpranjal/deepstats.git
cd deepstats
pip install -e ".[dev]"
```

## Quick Start

### Deep OLS Regression

```python
import deepstats as ds
import numpy as np

# Generate data
X = np.random.randn(1000, 3)
y = 2 + 0.5*X[:, 0] - 0.3*X[:, 1] + 0.8*X[:, 2] + np.random.randn(1000) * 0.5

# Fit model with robust standard errors
model = ds.DeepOLS(epochs=100, robust_se="HC1", random_state=42)
result = model.fit(X, y)

# Print Stata-style summary
print(result.summary())

# Access components
print(f"Coefficients: {result.params}")
print(f"Std Errors: {result.std_errors}")
print(f"R-squared: {result.r_squared:.4f}")

# Confidence intervals
print(result.confint(alpha=0.05))
```

### Using Formulas

```python
import pandas as pd

df = pd.DataFrame({
    'wage': y,
    'education': X[:, 0],
    'experience': X[:, 1],
    'ability': X[:, 2]
})

model = ds.DeepOLS(formula="wage ~ education + experience + ability")
result = model.fit(df)
print(result.summary())
```

### Causal Inference with Double Machine Learning

```python
from deepstats import DoubleMachineLearning

# Generate causal data with known treatment effect
X = np.random.randn(1000, 5)  # Confounders
T = np.random.binomial(1, 0.5, size=1000)  # Treatment
Y = 2.0 * T + X[:, 0] + np.random.randn(1000) * 0.5  # True ATE = 2.0

# Estimate ATE
dml = DoubleMachineLearning(n_folds=5, random_state=42)
result = dml.fit(Y=Y, T=T, X=X)

print(result.summary())
print(f"ATE: {result.ate:.4f} ({result.ate_se:.4f})")
print(f"95% CI: {result.confint()}")
```

### Custom Learners

```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

dml = DoubleMachineLearning(
    outcome_learner=RandomForestRegressor(n_estimators=100),
    treatment_learner=RandomForestClassifier(n_estimators=100),
    n_folds=5
)
result = dml.fit(Y=Y, T=T, X=X)
```

## API Reference

### Estimators

| Estimator | Description |
|-----------|-------------|
| `DeepOLS` | Deep neural network regression with robust SEs |
| `DoubleMachineLearning` | DML for causal effect estimation |

### Standard Error Types

| Type | Description |
|------|-------------|
| `"iid"` | Homoskedastic (classical OLS) |
| `"HC0"` | White (1980) heteroskedasticity-robust |
| `"HC1"` | HC0 with degrees of freedom correction (Stata default) |
| `"HC2"` | Leverage-adjusted |
| `"HC3"` | Jackknife estimator (most conservative) |
| `"cluster"` | Cluster-robust standard errors |

## Design Philosophy

### Hybrid Pattern

`deepstats` follows the **hybrid pattern** that combines sklearn's `fit/predict` API with statsmodels-style `Results` objects:

```python
# sklearn-compatible
model = DeepOLS()
result = model.fit(X, y)  # Returns DeepResults, not self
predictions = model.predict(X_new)

# statsmodels-like inference
print(result.summary())
print(result.confint())
print(result.vcov())
```

### Meta-Estimator Pattern

The `DoubleMachineLearning` estimator accepts arbitrary sklearn-compatible learners:

```python
# Use any sklearn model for nuisance estimation
dml = DoubleMachineLearning(
    outcome_learner=XGBRegressor(),
    treatment_learner=LogisticRegression()
)
```

### Config/State Separation

- `__init__`: Only stores hyperparameters (no computation)
- `fit()`: All computation happens here
- Enables `clone()` for cross-validation and grid search

## Testing

Run tests:
```bash
pytest tests/ -v
```

Run simulation tests (slow):
```bash
pytest tests/simulations/ -v -m slow
```

## References

### Papers

- Farrell, M. H., Liang, T., & Misra, S. (2021). "Deep Neural Networks for Estimation and Inference." *Econometrica*, 89(1), 181-213. [Link](https://www.econometricsociety.org/publications/econometrica/2021/01/01/deep-neural-networks-estimation-and-inference)

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). "Double/Debiased Machine Learning for Treatment and Structural Parameters." *The Econometrics Journal*, 21(1), C1-C68. [Link](https://academic.oup.com/ectj/article/21/1/C1/5056401)

### Related Software

- [pyfixest](https://github.com/py-econometrics/pyfixest) - Fast high-dimensional fixed effects regression
- [EconML](https://github.com/microsoft/EconML) - Microsoft's ML-based causal inference
- [DoWhy](https://github.com/py-why/dowhy) - Causal inference framework
- [statsmodels](https://www.statsmodels.org/) - Statistical models in Python
- [scikit-learn](https://scikit-learn.org/) - Machine learning in Python

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
