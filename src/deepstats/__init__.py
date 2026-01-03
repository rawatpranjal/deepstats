"""
deepstats: Production-grade econometrics with neural networks.

This package bridges the predictive flexibility of machine learning (scikit-learn)
with the statistical rigor of traditional econometrics (Stata/statsmodels).

Key Features
------------
- sklearn-compatible estimators (fit/predict interface)
- Robust standard errors (HC0, HC1, HC2, HC3, cluster)
- Statistical inference (summary tables, confidence intervals, p-values)
- Causal inference with Double Machine Learning
- Cross-fitting for valid inference

Basic Usage
-----------
>>> import deepstats as ds
>>> from deepstats.estimators import DeepOLS, DoubleMachineLearning
>>>
>>> # Deep regression with robust SEs
>>> model = DeepOLS(robust_se="HC1", epochs=100)
>>> result = model.fit(X, y)
>>> print(result.summary())
>>>
>>> # Causal inference
>>> dml = DoubleMachineLearning(n_folds=5)
>>> result = dml.fit(Y=Y, T=treatment, X=confounders)
>>> print(f"ATE: {result.ate:.4f} ({result.ate_se:.4f})")

References
----------
- Farrell, Liang, Misra (2021). "Deep Neural Networks for Estimation and Inference"
- Chernozhukov et al. (2018). "Double/Debiased Machine Learning"
"""

__version__ = "0.1.0"

# Main estimators
from .estimators.deep_ols import DeepOLS
from .estimators.dml import CausalResults, DoubleMachineLearning

# Results
from .results.deep_results import DeepResults

# Networks
from .networks.mlp import MLP, MLPClassifier, create_network

# Inference
from .inference.standard_errors import (
    compute_vcov,
    compute_vcov_hc0,
    compute_vcov_hc1,
    compute_vcov_hc2,
    compute_vcov_hc3,
)

__all__ = [
    # Version
    "__version__",
    # Estimators
    "DeepOLS",
    "DoubleMachineLearning",
    # Results
    "DeepResults",
    "CausalResults",
    # Networks
    "MLP",
    "MLPClassifier",
    "create_network",
    # Inference
    "compute_vcov",
    "compute_vcov_hc0",
    "compute_vcov_hc1",
    "compute_vcov_hc2",
    "compute_vcov_hc3",
]
