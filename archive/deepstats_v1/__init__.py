"""
deepstats: Enriched structural models with neural networks.

This package implements the Farrell-Liang-Misra (2021, 2023) paradigm where
parameters become functions: Y = a(X) + b(X)*T + ε. Neural networks output
parameter functions θ(X), enabling flexible heterogeneous treatment effects.

Key Features
------------
- DeepHTE: Heterogeneous treatment effects with neural networks
- R-formula interface: "Y ~ a(X1 + X2) + b(X1 + X2) * T"
- Multiple architectures: MLP, Transformer, LSTM
- Doubly robust ATE with influence function inference
- Quantile treatment effects for heterogeneity analysis
- GLM families (Normal, Bernoulli) with structural loss
- sklearn-compatible estimators (fit/predict interface)

Basic Usage
-----------
>>> import deepstats as ds
>>> import pandas as pd
>>>
>>> # Heterogeneous treatment effects (main use case)
>>> model = ds.DeepHTE(
...     formula="Y ~ a(X1 + X2 + X3) + b(X1 + X2 + X3) * T",
...     family="normal",
...     backbone="transformer"  # or "mlp", "lstm"
... )
>>> result = model.fit(data)
>>> print(result.summary())
>>>
>>> # Access individual treatment effects
>>> print(f"ATE: {result.ate:.3f} (SE: {result.ate_se:.3f})")
>>> print(f"ITE range: [{result.ite.min():.2f}, {result.ite.max():.2f}]")

References
----------
- Farrell, Liang, Misra (2021). "Deep Neural Networks for Estimation and Inference"
- Farrell, Liang, Misra (2023). "Deep Learning for Individual Heterogeneity"
- MisraLab course: github.com/MisraLab/cml.github.io
"""

__version__ = "0.1.0"

# Main estimators
from .estimators.linear_ols import LinearOLS
from .estimators.deep_ols import DeepOLS
from .estimators.deep_glm import DeepGLM
from .estimators.deep_poisson import DeepPoisson
from .estimators.hte import DeepHTE, HTEResults

# Results
from .results.deep_results import DeepResults
from .results.glm_results import GLMResults
from .results.poisson_results import PoissonResults

# Families
from .families.base import ExponentialFamily
from .families.normal import Normal
from .families.poisson import Poisson
from .families.bernoulli import Bernoulli
from .families.gamma import Gamma
from .families.exponential import Exponential

# Networks
from .networks.mlp import MLP, MLPClassifier, MLPBackbone, create_network
from .networks.transformer import TabularTransformer
from .networks.lstm import LSTMBackbone
from .networks.registry import ArchitectureRegistry
from .networks.base import NetworkArchitecture, BackboneNetwork, ParameterNetwork

# Formula
from .formula.parser import FormulaParser, ParsedFormula

# Inference
from .inference.standard_errors import (
    compute_vcov,
    compute_vcov_hc0,
    compute_vcov_hc1,
    compute_vcov_hc2,
    compute_vcov_hc3,
)
from .inference.bootstrap import bootstrap_pairs, bootstrap_wild
from .inference.influence import compute_influence_function_se
from .inference.validation import validate_standard_errors

# Datasets
from .datasets.generators import (
    make_linear_highdim,
    make_poisson_highdim,
    make_binary_highdim,
    make_nonlinear_highdim,
)
from .datasets.ab_test import (
    ABTestData,
    make_ab_test,
    make_ab_test_binary,
    make_ab_test_highdim,
)
from .datasets.benchmarks import (
    BenchmarkData,
    load_ihdp,
    load_jobs,
    load_twins,
    load_oj,
    load_acic,
    list_benchmarks,
    load_benchmark,
)
from .datasets.download import clear_cache

# Simulations
from .simulations import (
    SimulationStudy,
    SimulationResult,
    SimulationSummary,
    compute_simulation_metrics,
    diagnose_fitting,
    FittingDiagnosis,
    make_overfit_scenario,
    make_underfit_scenario,
    make_balanced_scenario,
)

__all__ = [
    # Version
    "__version__",
    # Estimators
    "LinearOLS",
    "DeepOLS",
    "DeepGLM",
    "DeepPoisson",
    "DeepHTE",
    # Results
    "DeepResults",
    "GLMResults",
    "PoissonResults",
    "HTEResults",
    # Families
    "ExponentialFamily",
    "Normal",
    "Poisson",
    "Bernoulli",
    "Gamma",
    "Exponential",
    # Networks
    "MLP",
    "MLPClassifier",
    "MLPBackbone",
    "TabularTransformer",
    "LSTMBackbone",
    "ArchitectureRegistry",
    "NetworkArchitecture",
    "BackboneNetwork",
    "ParameterNetwork",
    "create_network",
    # Formula
    "FormulaParser",
    "ParsedFormula",
    # Inference
    "compute_vcov",
    "compute_vcov_hc0",
    "compute_vcov_hc1",
    "compute_vcov_hc2",
    "compute_vcov_hc3",
    "bootstrap_pairs",
    "bootstrap_wild",
    "compute_influence_function_se",
    "validate_standard_errors",
    # Datasets
    "make_linear_highdim",
    "make_poisson_highdim",
    "make_binary_highdim",
    "make_nonlinear_highdim",
    # A/B test data
    "ABTestData",
    "make_ab_test",
    "make_ab_test_binary",
    "make_ab_test_highdim",
    # Benchmark datasets
    "BenchmarkData",
    "load_ihdp",
    "load_jobs",
    "load_twins",
    "load_oj",
    "load_acic",
    "list_benchmarks",
    "load_benchmark",
    "clear_cache",
    # Simulations
    "SimulationStudy",
    "SimulationResult",
    "SimulationSummary",
    "compute_simulation_metrics",
    "diagnose_fitting",
    "FittingDiagnosis",
    "make_overfit_scenario",
    "make_underfit_scenario",
    "make_balanced_scenario",
]
