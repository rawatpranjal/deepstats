"""Estimator classes for deepstats.

Available Estimators
--------------------
LinearOLS : Classical OLS with robust standard errors
DeepOLS : Deep neural network regression with robust SEs
DeepGLM : Deep GLM with proper MLE for various distributions
DeepHTE : Heterogeneous Treatment Effects via enriched structural models
DeepPoisson : Poisson regression with inference on rate parameter functionals
"""

from .base import DeepEstimatorBase
from .deep_ols import DeepOLS
from .deep_glm import DeepGLM
from .deep_poisson import DeepPoisson
from .linear_ols import LinearOLS
from .hte import DeepHTE, HTEResults

__all__ = [
    "DeepEstimatorBase",
    "LinearOLS",
    "DeepOLS",
    "DeepGLM",
    "DeepPoisson",
    "DeepHTE",
    "HTEResults",
]
